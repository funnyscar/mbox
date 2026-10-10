import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHead(nn.Module):
	def __init__(self, C, nheads):
		super().__init__()
		self.nh = nheads
		self.QKV = nn.Linear(C, 3*C, bias=False)
		self.W0 = nn.Linear(C, C, bias=False)

		self.register_buffer("k_cache", None, persistent=False)
		self.register_buffer("v_cache", None, persistent=False)

	def forward(self, x, pos=None):
		B, T, C = x.shape
		q, k, v = self.QKV(x).split(C, dim=-1)
		# k: (B, T, C)
		q, k, v = (t.view(B, T, self.nh, C//self.nh).transpose(1, 2) for t in (q, k, v))   
		# k: (B, nh, T, d)

		if pos is None: # normal training unaffected
			attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
		else: # caching during inference 

			# absolute pos in cache (pos = toks alr in cache)
			# (T) not Tmax
			idx = pos + torch.arange(T, device=x.device)

			# rather than recomputing 
			# (B, nheads, Tmax, d), we'll write into dim (Tmax)
			self.k_cache[:, :, idx, :] = k.to(self.k_cache.dtype)
			self.v_cache[:, :, idx, :] = v.to(self.v_cache.dtype)

			# grab tensor shape preassigned
			Tmax = self.k_cache.shape[2]

			# (1, Tmax) <= (T, 1) ... projected
			# (T, Tmax) same causal rule masking all the tokens behind and predicting the next one
			# remember idx is (0, 1, 2, 3, 4...) + off tensor size ()
			mask = torch.arange(Tmax, device=x.device)[None, :] <= idx[:, None]

			# use cache + custom mask instead of is_causal
			attn = F.scaled_dot_product_attention(q, self.k_cache, self.v_cache, attn_mask=mask)

		attn = attn.transpose(1, 2).contiguous().view(B, T, C)
		return self.W0(attn)


class Block(nn.Module):
	def __init__(self, C, nheads, nblocks):
		super().__init__()
		self.multiheads = MultiHead(C, nheads)
		self.mlp = nn.Sequential(
			nn.Linear(C, C*4, bias=False),
			nn.GELU(approximate="tanh"),
			nn.Linear(C*4, C, bias=False)
		)

		std = 0.02 / (2 * nblocks) ** 0.5
		nn.init.normal_(self.mlp[2].weight, mean=0.0, std=std)
		nn.init.normal_(self.multiheads.W0.weight, mean=0.0, std=std)

		self.ln1 = nn.LayerNorm(C, bias=False)
		self.ln2 = nn.LayerNorm(C, bias=False)

	def forward(self, x, pos=None):
		h = self.ln1(x)
		x = x + self.multiheads(h, pos)
		x = x + self.mlp(self.ln2(x))
		return x


class MegaGPT(nn.Module):
	def __init__(self, context_size=128, vocab_size=50304, embedding_size=128, nheads=4, nblocks=4):
		super().__init__()

		assert vocab_size % 64 == 0

		self.Tmax, self.C = context_size, embedding_size
		self.nheads, self.nblocks = nheads, nblocks

		assert self.C % nheads == 0, "embedding_size must be divisible by nheads"

		self.embeddings = nn.Embedding(vocab_size, self.C) 
		self.pos = nn.Embedding(self.Tmax, self.C) 

		std = 0.02
		nn.init.normal_(self.embeddings.weight, mean=0.0, std=std)
		nn.init.normal_(self.pos.weight, mean=0.0, std=std)

		self.blocks = nn.ModuleList([Block(self.C, nheads, nblocks) for _ in range(nblocks)])

		self.ln_f = nn.LayerNorm(self.C, bias=False)
		self.head = nn.Linear(self.C, vocab_size, bias=False)

		self.head.weight = self.embeddings.weight
	
	def setup_cache(self, batch_size=1, dtype=torch.bfloat16):
		dev = self.embeddings.weight.device
		d = self.C //self.nheads
		# kv cache = shape (B, nheads, max context, d)
		shape = (batch_size, self.nheads, self.Tmax, d)

		for b in self.blocks:
			mh = b.multiheads
			# not in state_dict, only for cuda graph later on
			mh.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=dev), persistent=False)
			mh.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=dev), persistent=False)
	
	def forward(self, tokens, pos=None):
		T = tokens.shape[1]
		assert T <= self.Tmax, f"Sequence length {T} exceeds model context size {self.Tmax}"

		offs = torch.arange(T, device=tokens.device)
		if pos is not None:
			offs = offs + pos

		x = self.embeddings(tokens) + self.pos(offs)

		for block in self.blocks:
			x = block(x, pos)

		logits = self.head(self.ln_f(x)) 
		return logits
