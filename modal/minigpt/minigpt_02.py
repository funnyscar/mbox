# minigpt-00 but we made it a lot bigger

import torch
import torch.nn as nn

class AttentionHead(nn.Module):
	def __init__(self, context_size, C, d):
		super().__init__()
		self.Tmax, self.C, self.d = context_size, C, d
		self.Q, self.K, self.V = nn.Linear(self.C, self.d), nn.Linear(self.C, self.d), nn.Linear(self.C, self.d)
		self.register_buffer("mask", torch.tril(torch.ones(self.Tmax, self.Tmax, dtype=torch.bool))) # move with the weights

	def forward(self, x):
		q, k, v = self.Q(x), self.K(x), self.V(x)
		scores = q @ k.mT / self.d**0.5
		T = x.shape[1] # allow for shorter sequences not hardcoded to context_size
		scores = scores.masked_fill(~self.mask[:T, :T], float('-inf'))
		a = torch.softmax(scores, dim=-1)
		return a @ v

class Block(nn.Module):
	def __init__(self, context_size, C, d, nheads):
		super().__init__()

		self.C, self.d = C, d

		self.heads = nn.ModuleList([AttentionHead(context_size, self.C, self.d) for _ in range(nheads)])

		self.W0 = nn.Linear(self.C, self.C)

		self.mlp = nn.Sequential(
			nn.Linear(self.C, self.C*4),
			nn.ReLU(),
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


class MiniGPT(nn.Module):
	def __init__(self, context_size=256, vocab_size=50257, embedding_size=768, nheads=12, nblocks=8):
		super().__init__()

		self.T, self.C = context_size, embedding_size
		self.d = self.C // nheads

		self.embeddings = nn.Embedding(vocab_size, self.C)
		self.pos = nn.Embedding(self.T, self.C) 

		self.blocks = nn.ModuleList([Block(self.T, self.C, self.d, nheads) for _ in range(nblocks)])

		self.ln_f = nn.LayerNorm(self.C)
		self.head = nn.Linear(self.C, vocab_size)
	
	def forward(self, tokens):
		T = tokens.shape[1]
		x = self.embeddings(tokens) + self.pos(torch.arange(T, device=tokens.device))

		for block in self.blocks:
			x = block(x)

		logits = self.head(self.ln_f(x)) 
		return logits
