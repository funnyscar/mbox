# THIS will take a lot of compute from scratch as you train a ViT and a LLM to predict the same embedding

import modal
import torch
import torch.nn as nn

app = modal.App("imagenet")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "datasets", "torchvision")
)

vol = modal.Volume.from_name("imagenet", create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")
CACHE = "/data/hf"
CKPT = "/data/dit_cifar10.pt"

class CLIP(nn.Module):
	def __init__(self, d=256, VIT_MODEL="google/vit-base-patch16-224", LLM_MODEL="distilbert-base-uncased"):
		super().__init__()
		self.img, self.txt = AutoModel.from_pretrained(VIT_MODEL), AutoModel.from_pretrained(LLM_MODEL)

		self.img_proj = nn.Linear(self.img.config.hidden_size, d)
		self.txt_proj = nn.Linear(self.txt.config.hidden_size, d)
		self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

	def forward(self, pix, ids, mask):
		# grab cls_token of image
		i = self.img(pixel_values=pix).last_hidden_state[:, 0]
		# grab cls_token of tokens
		t = self.txt(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]

		# mat mul into out embedding dim d=256
		i = F.normalize(self.img_proj(i), dim=-1)
		t = F.normalize(self.txt_proj(t), dim=-1)

		# mat mull each embedding dim (B, B)
		logits = self.logit_scale.exp().clamp(max=100) * i @ t.T          # (B, B)
		labels = torch.arange(len(i), device=i.device)                     # matches on diagonal
		return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

	
@app.function(image=image, volumes={"/data": vol}, secrets=[hf], gpu="A100", cpu=8, timeout=24 * 60 * 60)
def train(epochs: int = 5, bs: int = 128):
	import math, random
	import torch, torch.nn as nn, torch.nn.functional as F
	from datasets import load_dataset
	from torch.utils.data import DataLoader
	from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

	IMG, TXT = "google/vit-base-patch16-224", "distilbert-base-uncased"
	proc, tok = AutoImageProcessor.from_pretrained(IMG), AutoTokenizer.from_pretrained(TXT)

	def collate(batch):
		pix = proc([b["image"].convert("RGB") for b in batch], return_tensors="pt").pixel_values
		txt = tok([random.choice(b["caption"]) for b in batch], padding=True,
			truncation=True, max_length=64, return_tensors="pt")
		return pix, txt.input_ids, txt.attention_mask

	ds = load_dataset("lmms-lab/flickr30k", split="test", cache_dir=CACHE)
	dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=8,
		drop_last=True, collate_fn=collate, persistent_workers=True)

	model = CLIP(
		VIT_MODEL=IMG,
		LLM_MODEL=TXT
	).cuda()
	heads = [*model.img_proj.parameters(), *model.txt_proj.parameters(), model.logit_scale]
	opt = torch.optim.AdamW([
		{"params": [*model.img.parameters(), *model.txt.parameters()], "lr": 1e-5},
		{"params": heads, "lr": 1e-3},
	], weight_decay=0.01)

	for epoch in range(epochs):
		for pix, ids, mask in dl:
			with torch.autocast("cuda", dtype=torch.bfloat16):
				loss = model(pix.cuda(), ids.cuda(), mask.cuda())
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()
		print(f"epoch {epoch} loss {loss.item():.3f}")
		torch.save(model.state_dict(), "/data/clip.pt")
		vol.commit()


@app.local_entrypoint()
def main():
	train.remote()