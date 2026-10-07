import os
import modal
import torch
import torch.nn as nn
import json
import math

app = modal.App("imagenet")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "datasets", "transformers", "torchvision", "lpips", "tiktoken", "sentencepiece")
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("imagenet", create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")
CACHE = "/data/hf_cache"
CKPT = "/data/vae_cifar10.pt"


@app.function(image=image, volumes={"/data": vol}, secrets=[hf], gpu="A10G")
def generate(prompts: list[str], idx: int = -1):
	import random
	from datasets import load_dataset
	from transformers import AutoTokenizer, AutoImageProcessor

	IMG, TXT = "google/vit-base-patch16-224", "distilbert-base-uncased"
	proc, tok = AutoImageProcessor.from_pretrained(IMG, cache_dir=CACHE), AutoTokenizer.from_pretrained(TXT, cache_dir=CACHE)

	model = CLIP(VIT_MODEL=IMG, LLM_MODEL=TXT).cuda().eval()
	model.load_state_dict(torch.load("/data/clip.pt", map_location="cuda"))

	ds = load_dataset("lmms-lab/flickr30k", split="test", cache_dir=CACHE)
	ex = ds[random.randrange(len(ds)) if idx < 0 else idx]

	pix = proc(ex["image"].convert("RGB"), return_tensors="pt").pixel_values.cuda()
	txt = tok(prompts, padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")

	with torch.no_grad():
		i = F.normalize(model.img_proj(model.img(pixel_values=pix).last_hidden_state[:, 0]), dim=-1)
		t = F.normalize(model.txt_proj(model.txt(**txt).last_hidden_state[:, 0]), dim=-1)
		probs = (model.logit_scale.exp() * i @ t.T).softmax(-1)[0]

	print("true caption:", ex["caption"][0])
	for p, s in sorted(zip(prompts, probs.tolist()), key=lambda x: -x[1]):
		print(f"{s:.3f}  {p}")


@app.function(image=image, gpu="T4", volumes={"/data": vol}, timeout=60 * 60)
def train(epochs: int = 30, workers: int = 2, bs: int = 128):
	import torch
	import lpips
	from datasets import load_dataset
	from torch.utils.data import DataLoader
	from torchvision.transforms import v2
	from model import VAE

	dev = "cuda"
	tf = v2.Compose([v2.ToImage(), v2.RandomHorizontalFlip(),
		v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5] * 3, [0.5] * 3)])
	ds = load_dataset("uoft-cs/cifar10", split="train", cache_dir=CACHE)
	ds = ds.with_transform(lambda b: {"x": [tf(im.convert("RGB")) for im in b["img"]]})
	dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, drop_last=True,
		pin_memory=True, collate_fn=lambda b: torch.stack([e["x"] for e in b]))

	vae = VAE().to(dev)
	perc = lpips.LPIPS(net="vgg").to(dev).eval().requires_grad_(False)
	opt = torch.optim.AdamW(vae.parameters(), lr=2e-4, betas=(0.5, 0.9))
	scaler = torch.amp.GradScaler("cuda")

	for epoch in range(epochs):
		for x in dl:
			x = x.to(dev, non_blocking=True)
			with torch.autocast("cuda", dtype=torch.float16):
				rec, mu, logvar = vae(x)
			rec, mu, logvar = rec.float(), mu.float(), logvar.float()

			l1_loss = (rec - x).abs().mean() + 0.5
			kl = 0.5 * (mu**2 + logvar.exp() - 1 - logvar).sum(dim=[1, 2, 3]).mean()
			perc_loss = perc(rec, x).mean()
			loss = l1_loss + 0.5 * perc_loss + 1e-6 * kl

			opt.zero_grad(set_to_none=True)
			scaler.scale(loss).backward()
			scaler.unscale_(opt)
			torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
			scaler.step(opt)
			scaler.update()

		print(f"epoch {epoch} loss {loss.item():.4f} kl {kl.item():.1f}")
		torch.save(vae.state_dict(), CKPT)
		vol.commit()


@app.local_entrypoint()
def main(workers: int = 2):
	train.remote(workers=workers)