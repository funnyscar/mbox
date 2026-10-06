import modal
import torch
import torch.nn as nn

app = modal.App("cifar10")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "datasets", "torchvision")
)

# Persistent storage for the dataset and checkpoints
vol = modal.Volume.from_name("imagenet", create_if_missing=True)

class AttentionHead(nn.Module):
	def __init__(self, C, d):
		super().__init__()
		self.d = d
		self.Q, self.K, self.V = nn.Linear(C, d), nn.Linear(C, d), nn.Linear(C, d)

		# UNLIKE llm attention, we don't want causal masks
		# every patch embedding should see every other
		# self.register_buffer("mask", torch.tril(torch.ones(self.T, self.T, dtype=torch.bool))) # move with the weights

	def forward(self, x):
		q, k, v = self.Q(x), self.K(x), self.V(x)
		scores = q @ k.mT / self.d**0.5
		# scores = scores.masked_fill(~self.mask, float('-inf'))
		att = torch.softmax(scores, dim=-1)
		return att @ v

class Block(nn.Module):
	def __init__(self, C, d, nheads):
		super().__init__()

		self.C, self.d = C, d

		self.heads = nn.ModuleList([AttentionHead(self.C, self.d) for _ in range(nheads)])

		self.W0 = nn.Linear(self.C, self.C)

		self.mlp = nn.Sequential(
			nn.Linear(self.C, self.C*4),
			nn.GELU(), # better for VITs
			nn.Linear(self.C*4, self.C),
			nn.Dropout(0.1)
		)
		self.ln1 = nn.LayerNorm(self.C)
		self.ln2 = nn.LayerNorm(self.C)

	def forward(self, x):
		h = self.ln1(x)
		all_heads = torch.cat([attn(h) for attn in self.heads], dim=-1)
		x = x + self.W0(all_heads)
		x = x + self.mlp(self.ln2(x))
		return x


class ViT(nn.Module):
	def __init__(self, channels=3, H=32, W=32, patch_per_side=4,
		embedding_size=192, nheads=4, nblocks=6, num_classes=10):
		super().__init__()

		self.P, self.C = patch_per_side, embedding_size

		self.channels, self.H, self.W = channels, H, W

		P = self.P
		EMB = self.C

		assert EMB % nheads == 0
		self.d = EMB // nheads

		self.cls_token = nn.Parameter(torch.zeros(1, 1, EMB))
		self.emb = nn.Linear(self.channels * (H//P)  * (W//P), EMB)
		self.pos = nn.Embedding(P*P+1, EMB)

		nn.init.trunc_normal_(self.cls_token, std=0.02)
		nn.init.trunc_normal_(self.pos.weight, std=0.02)

		self.blocks = nn.ModuleList([Block(EMB, self.d, nheads) for _ in range(nblocks)])

		self.ln_f = nn.LayerNorm(EMB)
		self.head = nn.Linear(EMB, num_classes)
	
	def forward(self, images):
		B, channels, H, W = images.shape
		assert channels == self.channels and H == self.H and W == self.W

		P = self.P

		# (B, ch, P, H/P, P, W//P)
		patches = images.reshape(B, channels, P, H // P, P, W // P)
		# (B, P, P, ch, H//P, W//P)
		patches = patches.permute(0, 2, 4, 1, 3, 5)
		# (B, P*P, ch * H//P * W//P)
		patches = patches.reshape(B, P * P, -1)

		# (B, P*P, EMB)
		proj = self.emb(patches)
		# (B, P*P+1, EMB)
		emb = torch.cat([self.cls_token.expand(B, -1, -1), proj], dim=1)
		# (P*P+1,) -> (P*P+1, EMB)
		pos_emb = self.pos(torch.arange(P*P + 1, device=images.device))

		x = emb + pos_emb

		for block in self.blocks:
			x = block(x)

		x = self.ln_f(x)
		# (B, P*P+1, 10) so cls + each patch produces a 10 logit prediction.
		# classify only using cls dim
		logits = self.head(x[:,0]) 
		return logits


@app.function(image=image, gpu="T4", volumes={"/data": vol}, timeout=60 * 60)
def train():
	import torch
	from datasets import load_dataset
	from torchvision.transforms import v2
	import torchvision.transforms as transforms

	train_tf = T.Compose([
		T.RandomCrop(32, padding=4),
		T.RandomHorizontalFlip(),
		T.RandAugment(num_ops=2, magnitude=9),
		T.ToTensor(),
	])

	train_ds = load_dataset("uoft-cs/cifar10", split="train", streaming=True)
	train_ds = train_ds.shuffle(buffer_size=10_000, seed=0)

	def train_collate(batch):
		x = torch.stack([train_tf(b["img"]) for b in batch]) 
		y = torch.tensor([b["label"] for b in batch])
		return x, y

	def collate(batch):
		x = torch.stack([T.ToTensor()(b["img"]) for b in batch]) 
		y = torch.tensor([b["label"] for b in batch])
		return x, y

	train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, collate_fn=train_collate)

	val_ds = load_dataset("uoft-cs/cifar10", split="test", streaming=True)
	val_dl = torch.utils.data.DataLoader(val_ds, batch_size=256, collate_fn=collate)

	# randomly choose a augmentation
	# mixup - blending images λ·cat_img + (1−λ)·dog_img
	# cutmix - take a random rectangle from dog_img and paste it onto cat_img
	mix = v2.RandomChoice([v2.MixUp(num_classes=10), v2.CutMix(num_classes=10)])

	device = "cuda"

	mymodel = ViT().to(device)

	epochs = 200
	steps_per_epoch = 50_000 // 128
	total = epochs * steps_per_epoch
	warmup = 5 * steps_per_epoch

	criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
	opt = torch.optim.AdamW(mymodel.parameters(), lr=1e-3, weight_decay=0.05) 

	sched = torch.optim.lr_scheduler.SequentialLR(opt, [
    	torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup),
    	torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - warmup),
	], milestones=[warmup])

	lossi = []
	vlossi = []
	accui = []

	for epoch in range(epochs):
		for ctx, target in train_dl:
			x, y = mix(ctx, target)
			x, y = x.to(device), y.to(device)
			logits = mymodel(x)
			loss = criterion(logits, y)

			opt.zero_grad()
			loss.backward()
			opt.step()
			sched.step()

			lossi.append(loss.item())

		
		with torch.no_grad():
			mymodel.eval()
			correct = total = 0
			for ctx, target in val_dl:
				x, y = ctx.to(device), target.to(device)
				logits = mymodel(x)
				preds = logits.argmax(dim=1)
				correct += (preds == y).sum().item()
				total += y.numel()
				accui.append(correct/total)

				loss = criterion(logits, y)
				vlossi.append(loss)
			mymodel.train()
		

	with open("vit_cifar10.jsonl", "ab") as f:
		f.write(json.dumps({
			"train": lossi, "val": vlossi,
			"acc": accui
		}))
	
	torch.save(mymodel.state_dict(), "/data/vit_cifar10.pt")
	
	vol.commit()
	return loss.item()

@app.local_entrypoint()
def main():
	print("final loss:", train.remote())