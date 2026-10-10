import os, secrets
import modal
from fastapi import Header, HTTPException
from typing import Literal
from pydantic import BaseModel, Field

app = modal.App("fine-serve")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "tiktoken", "numpy", "fastapi[standard]")
		.env({
			"TORCHINDUCTOR_CACHE_DIR": "/data/inductor_cache",
			"TORCHINDUCTOR_COMPILE_THREADS": "1",
		})
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("fine")

CONTEXT_SIZE = 512
N_VOCAB = 50304
C, H, L = (1024, 16, 20)  # "r" config

# model_version -> checkpoint path (adjust to your actual filenames)
CHECKPOINTS = {
	"fine-r": "/data/fine-r-slim.pt",
	"fine-r-sft": "/data/fine-r-sft-slim.pt",
	"fine-r-best-sft": "/data/fine-r-best-sft-slim.pt",
}
ModelVersion = Literal["fine-r", "fine-r-sft", "fine-r-best-sft"]
DEFAULT_VERSION = "fine-r-best-sft"

class GenerateRequest(BaseModel):
	prompt: str = Field(min_length=1, max_length=2000)
	model_version: ModelVersion = DEFAULT_VERSION
	max_tokens: int = Field(200, ge=1, le=512)
	temperature: float = Field(0.8, gt=0, le=2.0)
	topk: int = Field(50, ge=5, le=N_VOCAB)


@app.cls(image=image, gpu="A10G", volumes={"/data": vol}, scaledown_window=300,
			secrets=[modal.Secret.from_name("models-gateway")], enable_memory_snapshot=True,
			experimental_options={"enable_gpu_snapshot": True})
class Model:

	# cold start snapshot...skip cuda init, compile already on gpu
	@modal.enter(snap=True)
	def load(self):
		import os, torch, tiktoken
		from model import MegaGPT

		self.enc = tiktoken.get_encoding("gpt2")
		self.eot = self.enc.eot_token
		self.models = {}

		for name, path in CHECKPOINTS.items():
			m = MegaGPT(context_size=CONTEXT_SIZE, vocab_size=N_VOCAB,
						embedding_size=C, nheads=H, nblocks=L).to("cuda").eval()
			m.load_state_dict(torch.load(path, map_location="cuda")["model"])
			m.setup_cache(1, dtype=torch.bfloat16)
			self.models[name] = {
				"prefill": m, # varable prompts
				"decode": torch.compile(m, mode="reduce-overhead", dynamic=False) # fixed shape one token
			}
		for name in self.models:  # warmup → compile happens before snapshot
			self._generate("Hello", name, 16, 0.8, 50)
		torch.cuda.synchronize()
	
	@modal.enter(snap=False)
	def reseed(self):
		import torch
		torch.manual_seed(secrets.randbits(63))

	def _generate(self, prompt, model_version, max_tokens, temperature, top_k):
		import torch

		model = self.models[model_version]
		ids = (self.enc.encode_ordinary(prompt) or [self.eot])[-CONTEXT_SIZE:]
		zero = torch.zeros((), dtype=torch.long, device="cuda")
		out = []

		with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
			x = torch.tensor([ids], dtype=torch.long, device="cuda")
			logits = model["prefill"](x, pos=zero)
			n = len(ids) # initial tokens in cache

			for _ in range(max_tokens):
				logit = logits[:, -1, :].float()
				logit[:, self.enc.n_vocab:] = -float("inf")  # mask padded vocab cuz multiple of 64

				logit /= temperature
				if top_k:
					v, _ = torch.topk(logit, top_k)
					logit[logit < v[:, [-1]]] = -float("inf")
				nxt = torch.multinomial(torch.softmax(logit, -1), 1)

				tok = nxt.item() # forces gpu sync
				if tok == self.eot:
					break

				out.append(tok)
				ids.append(tok)

				if n >= CONTEXT_SIZE:
					# cache full, prefill last half reset to 0
					keep = ids[-(CONTEXT_SIZE // 2):]
					n = len(keep)
					keep = torch.tensor([keep], device="cuda")
					logits = model["prefill"](keep, pos=zero)
				else:
					pos = torch.tensor(n, dtype=torch.long, device="cuda")
					torch.compiler.cudagraph_mark_step_begin()
					logits = model["decode"](nxt, pos)
					n += 1

		return self.enc.decode(out)

	@modal.method()
	def generate(self, prompt: str, model_version: str = DEFAULT_VERSION,
				 max_tokens: int = 200, temperature: float = 0.8, top_k: int = 50):
		if model_version not in self.models:
			raise ValueError(f"unknown model_version {model_version}")
		return self._generate(prompt, model_version, max_tokens, temperature, top_k)

	@modal.method()
	def throughput(self, prompt: str, model_version: str = DEFAULT_VERSION,
				 max_tokens: int = 200, temperature: float = 0.8, top_k: int = 50):
		if model_version not in self.models:
			raise ValueError(f"unknown model_version {model_version}")
		import time
		import torch
		torch.cuda.synchronize()
		t = time.time()
		text = self._generate(prompt, model_version, max_tokens, temperature, top_k)
		torch.cuda.synchronize()
		print(f"{max_tokens/(time.time()-t):.0f} tok/s")
		return text

	@modal.fastapi_endpoint(method="POST")
	def web(self,
		req: GenerateRequest,
		authorization: str | None = Header(default=None),
	):
		expected = os.environ["API_TOKEN"]
		if not authorization or not secrets.compare_digest(authorization, expected):
			raise HTTPException(status_code=401)
		text = self._generate(req.prompt, req.model_version, req.max_tokens,
							  req.temperature, req.topk)
		return {"text": text, "model_version": req.model_version}


@app.local_entrypoint()
def main(prompt: str = "The meaning of life is", model_version: str = DEFAULT_VERSION,
		 max_tokens: int = 200, temperature: float = 0.8, top_k: int = 50):
	print(prompt + Model().throughput.remote(prompt, model_version, max_tokens, temperature, top_k))