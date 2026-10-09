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
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("fine")

CONTEXT_SIZE = 512
N_VOCAB = 50304

# version -> (embedding_size, nheads, nblocks); must match train config
CONFIGS = {
	"n": (896, 14, 14),    # ~135M
	"o": (960, 15, 15),    # ~166M
	"p": (1024, 16, 16),   # ~201M
	"q": (1024, 16, 18),   # ~226M
	"r": (1024, 16, 20),   # ~250M
	"s": (1280, 20, 20),   # ~393M
	"t": (1280, 20, 24),   # ~472M
	"u": (1536, 24, 24),   # ~679M
	"v": (1600, 25, 28),   # ~860M
	"w": (1792, 28, 28),   # ~1.08B
	"x": (2048, 32, 28),   # ~1.41B
	"y": (2048, 32, 32),   # ~1.61B
	"z": (2304, 36, 32),   # ~2.04B
}
VERSION = "r"

class GenerateRequest(BaseModel):
	prompt: str = Field(min_length=1, max_length=2000)
	model_version: Literal["r"] = "r"
	max_tokens: int = Field(200, ge=1, le=512)
	temperature: float = Field(0.8, gt=0, le=2.0)
	topk: int = Field(50, ge=5, le=N_VOCAB)


@app.cls(image=image, gpu="A10G", volumes={"/data": vol}, scaledown_window=300, 
			secrets=[modal.Secret.from_name("models-gateway")])
class Model:
	version: str = modal.parameter(default=VERSION)

	@modal.enter()
	def load(self):
		import os, torch, tiktoken
		from model import MegaGPT

		C, H, L = CONFIGS[self.version]
		self.model = MegaGPT(
			context_size=CONTEXT_SIZE, vocab_size=N_VOCAB,
			embedding_size=C, nheads=H, nblocks=L,
		).to("cuda")

		path = f"/data/fine-{self.version}-best.pt"
		if not os.path.exists(path):
			path = f"/data/fine-{self.version}.pt"
		ck = torch.load(path, map_location="cuda")
		self.model.load_state_dict(ck["model"])
		self.model.eval()
		print(f"loaded {path} (step {ck['step']}, best_val {ck['best_val']:.4f})")

		self.enc = tiktoken.get_encoding("gpt2")
		self.eot = self.enc.eot_token

	def _generate(self, prompt, max_tokens, temperature, top_k):
		import torch

		ids = self.enc.encode_ordinary(prompt) or [self.eot]
		x = torch.tensor([ids], dtype=torch.long, device="cuda")
		out = []
		with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
			for _ in range(max_tokens):
				logits = self.model(x[:, -CONTEXT_SIZE:])[:, -1, :].float()
				logits[:, self.enc.n_vocab:] = -float("inf")  # mask padded vocab
				if temperature <= 0:
					nxt = logits.argmax(-1, keepdim=True)
				else:
					logits /= temperature
					if top_k:
						v, _ = torch.topk(logits, top_k)
						logits[logits < v[:, [-1]]] = -float("inf")
					nxt = torch.multinomial(torch.softmax(logits, -1), 1)
				if nxt.item() == self.eot:
					break
				out.append(nxt.item())
				x = torch.cat([x, nxt], dim=1)
		return self.enc.decode(out)

	@modal.method()
	def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8, top_k: int = 50):
		return self._generate(prompt, max_tokens, temperature, top_k)

	@modal.fastapi_endpoint(method="POST")
	def web(self,
		req: GenerateRequest,
		authorization: str | None = Header(default=None),
	):
		expected = os.environ["API_TOKEN"]
		if not authorization or not secrets.compare_digest(authorization, expected):
			raise HTTPException(status_code=401)
		text = self._generate(
			req.prompt,
			# req.model_version,
			req.max_tokens,
			req.temperature,
			req.topk
		)
		return {"text": text}


@app.local_entrypoint()
def main(prompt: str = "The meaning of life is", max_tokens: int = 200,
         temperature: float = 0.8, top_k: int = 50, version: str = VERSION):
	print(prompt + Model(version=version).generate.remote(prompt, max_tokens, temperature, top_k))