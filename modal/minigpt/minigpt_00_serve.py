import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

class GenRequest(BaseModel):
	prompt: str = Field(min_length=1, max_length=2000)
	max_tokens: int = Field(200, ge=1, le=500)
	temperature: float = Field(0.8, gt=0, le=2.0)

app = modal.App("minigpt_serve")
image = (
	modal.Image.debian_slim()
	.pip_install("torch", "tiktoken", "fastapi[standard]", "numpy")
	.add_local_python_source("minigpt_00")
)
vol = modal.Volume.from_name("minigpt")


@app.cls(image=image, gpu="T4", volumes={"/ckpts": vol}, scaledown_window=300, secrets=[modal.Secret.from_name("minigpt-auth")])
class Model:
	@modal.enter()
	def load(self):
		import torch, tiktoken
		from minigpt_00 import MiniGPT, context_size

		self.torch = torch
		self.enc = tiktoken.get_encoding("gpt2")
		self.context_size = context_size

		self.model = MiniGPT(vocab_size=self.enc.n_vocab).to("cuda").eval()
		ckpt = torch.load("/ckpts/minigpt-0.0.pt", map_location="cuda")
		self.model.load_state_dict(ckpt)

	@modal.method()
	def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8):
		torch = self.torch

		ids = torch.tensor([self.enc.encode(prompt)], device="cuda")
		with torch.no_grad():
			for _ in range(max_tokens):
				logits = self.model(ids[:, -self.context_size:])[:, -1, :] / temperature
				next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
				ids = torch.cat([ids, next_id], dim=1)
		return self.enc.decode(ids[0].tolist())


	@modal.fastapi_endpoint(method="POST")
	def web(self, req: GenRequest, authorization: str = Header(None)):
		import os, secrets
		expected = os.environ["API_TOKEN"]
		if not authorization or not secrets.compare_digest(authorization, expected):
			raise HTTPException(401)
		return {"text": self.generate.local(req.prompt, req.max_tokens, req.temperature)}

@app.local_entrypoint()
def main(prompt: str = "ROMEO:"):
	print(Model().generate.remote(prompt))