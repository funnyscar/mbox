import os
import modal
import torch
import torch.nn as nn

app = modal.App("imagenet")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "datasets", "torchvision")
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("imagenet", create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")
CACHE = "/data/hf"

@app.function(image=image, volumes={"/data": vol}, secrets=[hf],
              cpu=8, timeout=24 * 60 * 60)
def download():
    from datasets import load_dataset
    load_dataset("ILSVRC/imagenet-1k", cache_dir=CACHE, num_proc=8)
    vol.commit()


@app.function(image=image, gpu="A100", volumes={"/data": vol}, timeout=60 * 60 * 24, secrets=[hf])
def train():
	import torch
	import json
	from datasets import load_dataset
	from torchvision.transforms import v2
	mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

	train_tf = v2.Compose([
        v2.RandomResizedCrop(224, antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])

	val_tf = v2.Compose([
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])

	def apply(tf):
		def f(batch):
			batch["image"] = [tf(im.convert("RGB")) for im in batch["image"]]
			return batch
		return f

	ds = load_dataset("ILSVRC/imagenet-1k", cache_dir=CACHE)
	train_ds = ds["train"].with_transform(apply(train_tf))
	val_ds = ds["validation"].with_transform(apply(val_tf))

	def collate(batch):
		x = torch.stack([b["image"] for b in batch]) 
		y = torch.tensor([b["label"] for b in batch])
		return x, y

	train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, num_workers=16, pin_memory=True,
           persistent_workers=True, prefetch_factor=4, collate_fn=collate)
	val_dl = torch.utils.data.DataLoader(val_ds, batch_size=256, collate_fn=collate)

	mix = v2.RandomChoice([v2.MixUp(num_classes=1000), v2.CutMix(num_classes=1000)])

	device = "cuda"

	from model import ViT

	mymodel = ViT().to(device)
	cmymodel = torch.compile(mymodel)

	torch.set_float32_matmul_precision("high")

	epochs = 300
	steps_per_epoch = len(train_dl) // 128
	total = epochs * steps_per_epoch
	warmup = 5 * steps_per_epoch

	criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
	opt = torch.optim.AdamW(mymodel.parameters(), lr=1e-3, weight_decay=0.05, fused=True) 

	sched = torch.optim.lr_scheduler.SequentialLR(opt, [
    	torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup),
    	torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - warmup),
	], milestones=[warmup])

	lossi = []
	vlossi = []
	accui = []
	start = 0
	ckpt_path = "/data/ckpt.pt"
	if os.path.exists(ckpt_path):
		ck = torch.load(ckpt_path, map_location=device)
		mymodel.load_state_dict(ck["model"])
		opt.load_state_dict(ck["opt"])
		sched.load_state_dict(ck["sched"])
		start = ck["epoch"] + 1
		lossi = ck["lossi"]
		vlossi = ck["vlossi"]
		accui = ck["accui"]
		print(f"resuming from epoch {start}")

	for epoch in range(epochs):
		for ctx, target in train_dl:
			x, y = mix(ctx, target)
			x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

			with torch.autocast("cuda", dtype=torch.bfloat16):
				logits = cmymodel(x)
				loss = criterion(logits, y)

			opt.zero_grad()
			loss.backward()
			opt.step()
			sched.step()


		lossi.append(loss.item())
		
		with torch.no_grad():
			cmymodel.eval()
			correct = total = 0
			for ctx, target in val_dl:
				x, y = ctx.to(device), target.to(device)
				logits = cmymodel(x)
				preds = logits.argmax(dim=1)
				correct += (preds == y).sum().item()
				total += y.numel()
				accui.append(correct/total)

				vloss = criterion(logits, y).item()
				vlossi.append(vloss)
			cmymodel.train()

			rec = {"epoch": epoch, "train_loss": loss.item(),
               "val_loss": vloss, "acc": correct / total}
			print(rec)
			with open("/data/log.jsonl", "a") as f:
				f.write(json.dumps(rec) + "\n")
			torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch}, ckpt_path)
			vol.commit()
		

	with open("/data/vit_imagenet.jsonl", "a") as f:
		f.write(json.dumps({
			"train": lossi, "val": vlossi,
			"acc": accui
		}))
	
	torch.save(mymodel.state_dict(), "/data/vit_imagenet.pt")
	
	vol.commit()
	return loss.item()

@app.local_entrypoint()
def main():
	print("final loss:", train.remote())