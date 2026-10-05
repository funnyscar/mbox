import torch
import torch.nn as nn

class Block(nn.Module):
	def __init__(self, C, nheads):
		super().__init__()

		self.heads = nn.MultiheadAttention(
			embed_dim=C,
			num_heads=nheads,
			bias=True,
			batch_first=True
		)

		self.mlp = nn.Sequential(
			nn.Linear(C, C*4),
			nn.GELU(), # better for VITs
			nn.Linear(C*4, C),
		)
		self.ln1 = nn.LayerNorm(C)
		self.ln2 = nn.LayerNorm(C)

	def forward(self, x):
		h = self.ln1(x)
		all_heads, _ = self.heads(h, h, h, need_weights=False)
		x = x + all_heads
		x = x + self.mlp(self.ln2(x))
		return x


class ViT(nn.Module):
	def __init__(self, channels=3, H=224, W=224, patch_per_side=14,
		embedding_size=192, nheads=3, nblocks=12, num_classes=1000):
		super().__init__()

		self.P, self.C = patch_per_side, embedding_size

		self.channels, self.H, self.W = channels, H, W

		P = self.P
		EMB = self.C

		assert EMB % nheads == 0

		self.cls_token = nn.Parameter(torch.zeros(1, 1, EMB))
		self.emb = nn.Linear(self.channels * (H//P)  * (W//P), EMB)
		self.pos = nn.Embedding(P*P+1, EMB)

		nn.init.trunc_normal_(self.cls_token, std=0.02)
		nn.init.trunc_normal_(self.pos.weight, std=0.02)

		self.blocks = nn.ModuleList([Block(EMB, nheads) for _ in range(nblocks)])

		self.ln_f = nn.LayerNorm(EMB)
		self.head = nn.Linear(EMB, num_classes)
	
	def forward(self, images):
		B, channels, H, W = images.shape
		assert channels == self.channels and H == self.H and W == self.W

		P = self.P

		patches = images.reshape(B, channels, P, H // P, P, W // P)
		patches = patches.permute(0, 2, 4, 1, 3, 5)
		patches = patches.reshape(B, P * P, -1)

		proj = self.emb(patches)
		emb = torch.cat([self.cls_token.expand(B, -1, -1), proj], dim=1)

		x = emb + self.pos.weight

		for block in self.blocks:
			x = block(x)

		x = self.ln_f(x)
		logits = self.head(x[:,0]) 
		return logits
