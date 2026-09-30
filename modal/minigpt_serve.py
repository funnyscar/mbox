# serve.py
import modal

app = modal.App("minigpt_serve")
image = (
	modal.Image.debian_slim()
	.pip_install("torch", "tiktoken", "fastapi[standard]")
	.add_local_python_source("minigpt")
)
vol = modal.Volume.from_name("minigpt")


@app.cls(image=image, gpu="T4", volumes={"/ckpts": vol}, scaledown_window=300)
class Model:
	@modal.enter()
	def load(self):
		import torch, tiktoken
		from minigpt import MiniGPT

		self.torch = torch
		self.enc = tiktoken.get_encoding("gpt2")
		self.model = MiniGPT().to("cuda").eval()
		ckpt = torch.load("/ckpts/minigpt.pt", map_location="cuda")
		self.model.load_state_dict(ckpt)  # or ckpt if you saved only weights

	@modal.method()
	def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8):
		torch = self.torch
		from minigpt import context_size

		ids = torch.tensor([self.enc.encode(prompt)], device="cuda")
		with torch.no_grad():
			for _ in range(max_tokens):
				logits = self.model(ids[:, -context_size:])[:, -1, :] / temperature
				next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
				ids = torch.cat([ids, next_id], dim=1)
		return self.enc.decode(ids[0].tolist())

	@modal.fastapi_endpoint(method="POST")
	def web(self, body: dict):
		return {"text": self.generate.local(body["prompt"], body.get("max_tokens", 200))}


@app.local_entrypoint()
def main(prompt: str = "ROMEO:"):
	print(Model().generate.remote(prompt))