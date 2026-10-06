# reimplementing my toy version in folder minimal. it just needs more compute to actually give reasonable results

import io
import modal
import torch
import torch.nn as nn

app = modal.App("imagenet")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "datasets", "torchvision")
		.add_local_python_source("model_cifar10")
)

vol = modal.Volume.from_name("imagenet", create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")
CACHE = "/data/hf"
CKPT = "/data/dit_cifar10.pt"

@app.function(image=image, volumes={"/data": vol}, secrets=[hf],
              cpu=8, timeout=24 * 60 * 60)
def download():
    from datasets import load_dataset
    load_dataset("uoft-cs/cifar10", cache_dir=CACHE, num_proc=8)
    vol.commit()


@app.function(image=image, gpu="A100", cpu=8, volumes={"/data": vol}, timeout=60 * 60 * 15, secrets=[hf])
def train(epochs: int = 200):
	import torch
	import json
	from datasets import load_dataset
	from torchvision.transforms import v2
	from model_cifar10 import DiT
	import torch.nn.functional as F
	from torch.utils.data import DataLoader

	ds = load_dataset("uoft-cs/cifar10", split="train", cache_dir=CACHE)
	tf = v2.Compose([
		v2.ToImage(),
		v2.RandomHorizontalFlip(),
		v2.ToDtype(torch.float32, scale=True), # [0, 1]
		v2.Normalize([0.5] * 3, [0.5] * 3), # [-1, 1]
	])

	def apply_tf(batch):
		return {
			"x": [tf(img.convert("RGB")) for img in batch["img"]],
			"y": batch["label"],
		}

	ds = ds.with_transform(apply_tf)

	def collate(batch):
		x = torch.stack([b["x"] for b in batch])
		y = torch.tensor([b["y"] for b in batch])
		return x, y

	train_dl = DataLoader(
		ds, batch_size=128, shuffle=True, num_workers=8,
		pin_memory=True, drop_last=True, persistent_workers=True,
		collate_fn=collate,
	)

	device = "cuda"
	mymodel = DiT().to(device)
	cmodel = torch.compile(mymodel)

	T = 1000
	beta = torch.linspace(1e-4, 0.02, T, device=device)
	abar = torch.cumprod(1 - beta, dim=0)
	opt = torch.optim.AdamW(mymodel.parameters(), lr=1e-4, weight_decay=0) 
	sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000))

	for epoch in range(epochs):
		for x, y in train_dl:

			x = x.to(device)
			y = y.to(device)

			B, C, H, W = x.shape
			noise = torch.randn_like(x)

			# using t to approx the noise
			t = torch.randint(0, T, (B,), device=device)
			a = abar[t].view(B, 1, 1, 1)
			x_t = torch.sqrt(a) * x + torch.sqrt(1-a) * noise

			# Add "dropout" for label. for unconditional mode (aka don't need it to generate image)
			drop_mask = torch.rand(B, device=device) < 0.1
			y = torch.where(drop_mask, torch.full_like(y, 10), y)

			with torch.autocast("cuda", dtype=torch.bfloat16):
				pred_noise = cmodel(x_t, t, y)
			loss = F.mse_loss(pred_noise.float(), noise)


			opt.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
			opt.step()
			sched.step()

		print(loss.item())

		torch.save(mymodel.state_dict(), CKPT)
		vol.commit()
	
	print("done")



@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60 * 3)
def generate(noise_t: int = 500, steps: int = 50, seed: int = 0) -> bytes:
	import math
	import torch
	from datasets import load_dataset
	from torchvision import transforms
	from torchvision.utils import make_grid
	from PIL import Image
	from model_cifar10 import DiT

	dev = "cuda"
	g = torch.Generator(device="cpu").manual_seed(seed)

	# --- load model (prefer EMA weights, strip torch.compile prefix) ---
	model = DiT().to(dev).eval()
	sd = torch.load(CKPT, map_location=dev)
	sd = sd.get("ema", sd.get("model", sd))
	model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})

	# --- random real image ---
	ds = load_dataset("uoft-cs/cifar10", split="test", cache_dir=CACHE)
	idx = torch.randint(len(ds), (1,), generator=g).item()
	ex = ds[idx]
	tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
	x0 = tf(ex["img"].convert("RGB"))[None].to(dev)                # (1,3,32,32) in [-1,1]
	y = torch.tensor([ex["label"]], device=dev)

	# --- schedule: must match training ---
	N = 1000
	beta = torch.linspace(1e-4, 0.02, N, device=dev)
	abar = torch.cumprod(1 - beta, dim=0)

	# --- forward: add noise up to step noise_t ---
	noise_t = min(max(noise_t, 1), N - 1)
	a = abar[noise_t]
	noise = torch.randn(x0.shape, generator=g).to(dev)
	x_noisy = a.sqrt() * x0 + (1 - a).sqrt() * noise

	# --- reverse: deterministic DDIM from noise_t down to 0 ---
	x = x_noisy
	ts = torch.linspace(noise_t, 0, steps + 1, device=dev).round().long()
	with torch.no_grad():
		for i in range(steps):
			t, s = ts[i], ts[i + 1]
			if t == s:
				continue
			a_t = abar[t]
			a_s = abar[s] if i < steps - 1 else torch.tensor(1.0, device=dev)
			with torch.autocast("cuda", dtype=torch.bfloat16):
				eps = model(x, t.expand(1), y).float()
			x0_hat = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
			x = a_s.sqrt() * x0_hat + (1 - a_s).sqrt() * eps

	mse = ((x - x0) ** 2).mean().item()
	print(f"idx={idx} label={ex['label']} noise_t={noise_t} steps={steps} recon MSE={mse:.4f}")

	# --- [original | noised | denoised] ---
	row = (torch.cat([x0, x_noisy.clamp(-1, 1), x.clamp(-1, 1)]) + 1) / 2
	grid = make_grid(row, nrow=3, padding=2)
	arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
	buf = io.BytesIO()
	Image.fromarray(arr).resize((arr.shape[1] * 6, arr.shape[0] * 6), Image.NEAREST).save(buf, "PNG")
	vol.commit()   # persist the HF cache for next run
	return buf.getvalue()

 
@app.local_entrypoint()
def main(noise_t: int = 500, steps: int = 50, seed: int = 0):
	# png = generate.remote(T, steps, seed)

	# out = f"denoise_T{T}_seed{seed}.png"
	# with open(out, "wb") as f:
	# 	f.write(png)
	# print(f"saved {out}")
	train.remote()
 
