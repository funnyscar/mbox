import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHead(nn.Module):
	def __init__(self, C, nheads):
		super().__init__()
		self.nh = nheads
		self.QKV = nn.Linear(C, 3*C, bias=False)
		self.W0 = nn.Sequential(
			nn.Linear(C, C, bias=False),
			nn.Dropout(0.1)
		)

	def forward(self, x):
		B, T, C = x.shape
		q, k, v = self.QKV(x).split(C, dim=-1)
		q, k, v = (t.view(B, T, self.nh, C//self.nh).transpose(1, 2) for t in (q, k, v))   
		attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
		attn = attn.transpose(1, 2).contiguous().view(B, T, C)
		return self.W0(attn)


class Block(nn.Module):
	def __init__(self, C, nheads, nblocks):
		super().__init__()
		self.multiheads = MultiHead(C, nheads)
		self.mlp = nn.Sequential(
			nn.Linear(C, C*4, bias=False),
			nn.GELU(),
			nn.Linear(C*4, C, bias=False),
			nn.Dropout(0.1)
		)

		std = 0.02 / (2 * nblocks) ** 0.5
		nn.init.normal_(self.mlp[2].weight, mean=0.0, std=std)
		nn.init.normal_(self.multiheads.W0[0].weight, mean=0.0, std=std)

		self.ln1 = nn.LayerNorm(C)
		self.ln2 = nn.LayerNorm(C)

	def forward(self, x):
		h = self.ln1(x)
		x = x + self.multiheads(h)
		x = x + self.mlp(self.ln2(x))
		return x


class MainGPT(nn.Module):
	def __init__(self, context_size=128, vocab_size=50304, embedding_size=128, nheads=4, nblocks=4):
		super().__init__()
		self.nheads = nheads
		self.nblocks = nblocks

		self.Tmax, self.C = context_size, embedding_size

		assert self.C % nheads == 0, "embedding_size must be divisible by nheads"

		self.embeddings = nn.Embedding(vocab_size, self.C) 
		self.pos = nn.Embedding(self.Tmax, self.C) 

		std = 0.02
		nn.init.normal_(self.embeddings.weight, mean=0.0, std=std)
		nn.init.normal_(self.pos.weight, mean=0.0, std=std)

		self.blocks = nn.ModuleList([Block(self.C, nheads, nblocks) for _ in range(nblocks)])

		self.ln_f = nn.LayerNorm(self.C)
		self.head = nn.Linear(self.C, vocab_size, bias=False)

		self.head.weight = self.embeddings.weight
	
	def forward(self, tokens):
		T = tokens.shape[1]
		assert T <= self.Tmax, f"Sequence length {T} exceeds model context size {self.Tmax}"
		x = self.embeddings(tokens) + self.pos(torch.arange(T, device=tokens.device))

		for block in self.blocks:
			x = block(x)

		logits = self.head(self.ln_f(x)) 
		return logits
