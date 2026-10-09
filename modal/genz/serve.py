import os
import secrets
from typing import Literal

import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
	prompt: str = Field(min_length=1, max_length=2000)
	model_version: Literal["genz_00", "genz_l", "genz_l.01"] = "genz_l.01"
	max_tokens: int = Field(200, ge=1, le=512)
	temperature: float = Field(0.8, gt=0, le=2.0)


MODEL_SPECS = {
	"genz_00": {
		"checkpoint": "/data/genz-0.1.pt",
		"context_size": 128,
		"embedding_size": 128,
		"nheads": 4,
		"nblocks": 4,
	},
	"genz_l": {
		"checkpoint": "/data/genz-l.pt",
		"context_size": 512,
		"embedding_size": 768,
		"nheads": 12,
		"nblocks": 12,
	},
	"genz_l.01": {
		"checkpoint": "/data/genz-l.01.pt",
		"context_size": 512,
		"embedding_size": 768,
		"nheads": 12,
		"nblocks": 12,
	},
}

app = modal.App("genz-serve")
image = (
	modal.Image.debian_slim()
	.pip_install("torch", "tiktoken", "numpy", "fastapi[standard]")
	.add_local_python_source("genz_00", "genz_10")
)
vol = modal.Volume.from_name("genz")


@app.cls(
	image=image,
	gpu="A10G",
	volumes={"/data": vol},
	scaledown_window=300,
	enable_memory_snapshot=True,
	secrets=[modal.Secret.from_name("models-gateway")],
)
class GenZModel:
	@modal.enter(snap=True)
	def load_cpu_models(self):
		import torch
		import tiktoken
		from genz_00 import GenZGPT as GenZ00
		from genz_10 import GenZGPT as GenZ10

		torch.set_num_threads(1)
		self.torch = torch
		self.encoder = tiktoken.get_encoding("gpt2")
		self.models = {}

		for version, model_class in (("genz_00", GenZ00), ("genz_l", GenZ10), ("genz_l.01", GenZ10)):
			spec = MODEL_SPECS[version]
			checkpoint = torch.load(
				spec["checkpoint"],
				map_location="cpu",
				weights_only=True,
				mmap=True,
			)
			if "model" not in checkpoint:
				raise ValueError(f"Checkpoint {spec['checkpoint']} does not contain model weights")

			state_dict = {
				key.removeprefix("_orig_mod."): value
				for key, value in checkpoint["model"].items()
			}
			model = model_class(
				context_size=spec["context_size"],
				vocab_size=50304,
				embedding_size=spec["embedding_size"],
				nheads=spec["nheads"],
				nblocks=spec["nblocks"],
			)
			model.load_state_dict(state_dict)
			del state_dict, checkpoint
			self.models[version] = model.half().eval()

	@modal.enter(snap=False)
	def move_models_to_gpu(self):
		device = "cuda"
		with self.torch.inference_mode():
			for model in self.models.values():
				model.to(device)
				model(self.torch.zeros(1, 8, dtype=self.torch.long, device=device))
		self.torch.cuda.synchronize()

	@modal.method()
	def generate(
		self,
		prompt: str,
		model_version: str = "genz_l",
		max_tokens: int = 200,
		temperature: float = 0.8,
	):
		if model_version not in MODEL_SPECS:
			raise ValueError(f"Unknown model version: {model_version}")

		torch = self.torch
		model = self.models[model_version]
		ids = torch.tensor(
			[self.encoder.encode(prompt)],
			device="cuda",
			dtype=torch.long,
		)
		eot_token = self.encoder.eot_token

		max_phrases = 5

		prompt_len = ids.shape[1]
		n_eot = 0
		with torch.inference_mode():
			for _ in range(max_tokens):
				logits = model(ids[:, -model.Tmax:])[:, -1, :].float()
				logits[:, self.encoder.n_vocab:] = -float("inf")  # mask padded vocab
				next_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
				ids = torch.cat([ids, next_id], dim=1)
				if next_id.item() == eot_token:
					n_eot += 1
					if n_eot >= max_phrases:
						break

		# split generated tokens on EOT; first segment continues the prompt
		segments, cur = [], []
		for t in ids[0, prompt_len:].tolist():
			if t == eot_token:
				segments.append(self.encoder.decode(cur))
				cur = []
			else:
				cur.append(t)
		if cur:
			segments.append(self.encoder.decode(cur))  # last one may be cut off by max_tokens
		return "\n---\n".join(segments)


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
	prompt: str = "whats good homie?",
	model_version: str = "genz_l.01",
	max_tokens: int = 200,
	temperature: float = 0.8,
):
	if model_version not in MODEL_SPECS:
		raise ValueError(f"model_version must be one of {', '.join(MODEL_SPECS)}")
	print(
		GenZModel().generate.remote(
			prompt,
			model_version,
			max_tokens,
			temperature,
		)
	)
