import os
import secrets
from typing import Literal

import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
	prompt: str = Field(min_length=1, max_length=2000)
	model_version: Literal["gen_00", "gen_10"] = "gen_10"
	max_tokens: int = Field(200, ge=1, le=512)
	temperature: float = Field(0.8, gt=0, le=2.0)


MODEL_SPECS = {
	"gen_00": {
		"checkpoint": "/data/genz-0.1.pt",
		"context_size": 128,
		"embedding_size": 128,
		"nheads": 4,
		"nblocks": 4,
	},
	"gen_10": {
		"checkpoint": "/data/genz-l.pt",
		"context_size": 512,
		"embedding_size": 768,
		"nheads": 12,
		"nblocks": 12,
	},
}

app = modal.App("genz-serve")
image = (
	modal.Image.debian_slim()
	.pip_install("torch", "tiktoken", "fastapi[standard]")
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

		for version, model_class in (("gen_00", GenZ00), ("gen_10", GenZ10)):
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
		model_version: str = "gen_10",
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

		with torch.inference_mode():
			for token_index in range(max_tokens):
				logits = model(ids[:, -model.Tmax :])[:, -1, :]
				next_id = torch.multinomial(
					torch.softmax(logits.float() / temperature, dim=-1),
					num_samples=1,
				)
				ids = torch.cat([ids, next_id], dim=1)
				if token_index % 16 == 15 and (ids[0, -16:] == eot_token).any().item():
					break

		output_ids = ids[0].tolist()
		if eot_token in output_ids:
			output_ids = output_ids[: output_ids.index(eot_token)]
		return self.encoder.decode(output_ids)

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
	prompt: str = "whats good",
	model_version: str = "gen_10",
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
