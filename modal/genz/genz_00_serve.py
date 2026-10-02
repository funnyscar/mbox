import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

class GenRequest(BaseModel):
	prompt: str = Field(min_length=1, max_length=2000)
	max_tokens: int = Field(200, ge=1, le=500)
	temperature: float = Field(0.8, gt=0, le=2.0)

app = modal.App("genz00_serve")
image = (
	modal.Image.debian_slim()
	.pip_install("torch", "tiktoken", "fastapi[standard]", "numpy")
	.add_local_python_source("genz_00")
)
vol = modal.Volume.from_name("genz")


@app.cls(image=image, gpu="T4", volumes={"/data": vol}, scaledown_window=300, enable_memory_snapshot=True, secrets=[modal.Secret.from_name("models-gateway")])
class Model:
	@modal.enter(snap=True) # load on cpu
	def load(self):
		import torch, tiktoken
		from genz_00 import GenZGPT

		self.torch = torch
		self.enc = tiktoken.get_encoding("gpt2")

		ckpt = torch.load("/data/genz-0.1.pt", map_location="cuda")
		sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

		self.model = GenZGPT()
		self.model.load_state_dict(sd)
		self.model = self.model.half().to("cuda").eval() # do it after loaded weights

		with torch.inference_mode():
			self.model(torch.zeros(1, 8, dtype=torch.long, device="cuda"))

	@modal.method()
	def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8):
		torch = self.torch
		ids = torch.tensor([self.enc.encode(prompt)], device="cuda")
		EOT = self.enc.eot_token

		with torch.inference_mode():
			for i in range(max_tokens):
				logits = self.model(ids[:, -self.model.Tmax:])[:, -1, :] / temperature
				next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
				ids = torch.cat([ids, next_id], dim=1)
				if i % 16 == 15 and (ids[0, -16:] == EOT).any():
					break
		out = ids[0].tolist()
		if EOT in out:
			out = out[:out.index(EOT)]
		return self.enc.decode(out)


	@modal.fastapi_endpoint(method="POST")
	def web(self, req: GenRequest, authorization: str = Header(None)):
		import os, secrets
		expected = os.environ["API_TOKEN"]
		if not authorization or not secrets.compare_digest(authorization, expected):
			raise HTTPException(401)
		return {"text": self.generate.local(req.prompt, req.max_tokens, req.temperature)}

@app.local_entrypoint()
def main(prompt: str = "whats good"):
	print(Model().generate.remote(prompt))