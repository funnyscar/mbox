import os
import secrets

import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
	prompt: str = Field(min_length=1, max_length=2000)
	model_version: str = Field("l", pattern="^[a-s]$")
	max_tokens: int = Field(200, ge=1, le=512)
	temperature: float = Field(0.8, gt=0, le=2.0)


MODEL_CONFIGS = {
	"a": (128, 2, 4),
	"b": (192, 3, 4),
	"c": (192, 3, 6),
	"d": (256, 4, 6),
	"e": (256, 4, 8),
	"f": (320, 5, 8),
	"g": (384, 6, 8),
	"h.01": (448, 7, 9),
	"i.01": (512, 8, 10),
	"j": (576, 9, 11),
	"k": (640, 10, 12),
	"l.01": (768, 12, 12),
	"m.01": (832, 13, 13),
	"n.01": (896, 14, 14),
	"o": (960, 15, 15),
	"p": (1024, 16, 16),
	"q": (1024, 16, 18),
	"r": (1024, 16, 20),
	"s": (1280, 20, 20),
}
DEFAULT_MODEL_VERSION = "l"

app = modal.App("chinesegpt-serve")
image = (
	modal.Image.debian_slim()
	.pip_install("torch", "transformers", "sentencepiece", "fastapi[standard]")
	.add_local_python_source("model")
)
vol = modal.Volume.from_name("chinese")


@app.cls(
	image=image,
	gpu="T4",
	volumes={"/data": vol},
	scaledown_window=300,
	enable_memory_snapshot=True,
	secrets=[modal.Secret.from_name("models-gateway")],
)
class ChineseModel:
	@modal.enter(snap=True)
	def load(self):
		import torch
		from transformers import AutoTokenizer

		self.torch = torch
		self.device = "cuda"
		self.tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B")
		self.model_version = None
		self.model = None
		self.model_device = None
		if os.path.isfile(f"/data/chinese-{DEFAULT_MODEL_VERSION}.pt"):
			self._load_model(DEFAULT_MODEL_VERSION, device="cpu")

	def _load_model(self, model_version: str, device: str):
		from model import ChineseGPT

		torch = self.torch
		path = f"/data/chinese-{model_version}.pt"
		checkpoint = torch.load(
			path,
			map_location="cpu",
			weights_only=True,
			mmap=True,
		)
		if "model" not in checkpoint:
			raise ValueError(f"Checkpoint {path} does not contain model weights")

		embedding_size, nheads, nblocks = MODEL_CONFIGS[model_version]
		vocab_size = (-len(self.tokenizer) % 64) + len(self.tokenizer)
		model = ChineseGPT(
			context_size=512,
			vocab_size=vocab_size,
			embedding_size=embedding_size,
			nheads=nheads,
			nblocks=nblocks,
		)
		model.load_state_dict(checkpoint["model"])
		del checkpoint

		if self.model is not None:
			del self.model
			torch.cuda.empty_cache()

		self.model = model.half().to(device).eval()
		self.model_version = model_version
		self.model_device = device

	@modal.method()
	def generate(
		self,
		prompt: str,
		model_version: str = "l",
		max_tokens: int = 200,
		temperature: float = 0.8,
	):
		torch = self.torch
		if model_version not in MODEL_CONFIGS:
			raise ValueError(f"Unknown model version: {model_version}")
		if self.model_version != model_version:
			self._load_model(model_version, device=self.device)
		elif self.model_device != self.device:
			self.model = self.model.to(self.device)
			self.model_device = self.device

		ids = torch.tensor(
			[self.tokenizer.encode(prompt, add_special_tokens=False)],
			device=self.device,
		)
		eos_token_id = self.tokenizer.eos_token_id

		with torch.inference_mode():
			for _ in range(max_tokens):
				logits = self.model(ids[:, -self.model.Tmax :])[:, -1, :]
				next_id = torch.multinomial(
					torch.softmax(logits / temperature, dim=-1),
					num_samples=1,
				)
				ids = torch.cat([ids, next_id], dim=1)
				if eos_token_id is not None and next_id.item() == eos_token_id:
					break

		output_ids = ids[0].tolist()
		if eos_token_id is not None and eos_token_id in output_ids:
			output_ids = output_ids[: output_ids.index(eos_token_id)]
		return self.tokenizer.decode(output_ids, skip_special_tokens=True)

	@modal.fastapi_endpoint(method="POST")
	def web(
		self,
		req: GenerateRequest,
		authorization: str | None = Header(default=None),
	):
		expected = os.environ["API_TOKEN"]
		if not authorization or not secrets.compare_digest(authorization, expected):
			raise HTTPException(status_code=401)
		text = self.generate.local(
			req.prompt,
			req.model_version,
			req.max_tokens,
			req.temperature,
		)
		return {"text": text, "model_version": req.model_version}


@app.local_entrypoint()
def main(
	prompt: str = "你好",
	model_version: str = DEFAULT_MODEL_VERSION,
	max_tokens: int = 200,
	temperature: float = 0.8,
):
	if model_version not in MODEL_CONFIGS:
		raise ValueError(f"model_version must be one of {', '.join(MODEL_CONFIGS)}")
	print(
		ChineseModel().generate.remote(
			prompt,
			model_version,
			max_tokens,
			temperature,
		)
	)
