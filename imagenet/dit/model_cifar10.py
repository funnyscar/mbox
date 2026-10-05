# Addressing problems:
# - [x] t embedding not continuous and not good embedding
# - [x] y_emb + t_emb (conditional) can be used better than just superposition
# - [x] ada layers
# - [] CFG to get sharper results
# - [x] modulate correctly
# - autoencoder to compress image down (only when image is large)

import math
import torch
import torch.nn as nn

def modulate(x, shift, scale):
	# obviously scale and shift x along D channels 
    # x: (B, P*P toks, embedding);
    # x: (B, N toks, C);
	# shift, scale: (B, C)

	# (B, 1, C)
	scale = scale.unsqueeze(1)
	scale = 1 + scale # baseline is identity. not 0 scale

	# (B, 1, C)
	shift = shift.unsqueeze(1)
	
	return x * scale + shift

class TEmbed(nn.Module):
	def __init__(self, hidden_size, freq_dim=256, max_period=10000):
		super().__init__()
		self.freq_dim = freq_dim
		self.max_period = max_period

		self.mlp = nn.Sequential(
			nn.Linear(freq_dim, hidden_size),
			nn.SiLU(),
			nn.Linear(hidden_size, hidden_size)
		)
	
	def sinusoidal(self, t):
		# half
		half = self.freq_dim // 2

		# f_i = e^{-log(P)*i / half} = P^{-i/half}
		# f_0 = 1 ---> f_{half-1} ~ P^{-1}
		ls = torch.arange(half, device=t.device, dtype=torch.float32)
		freqs = torch.exp(-math.log(self.max_period) * ls / half)
		
		# theta_i = t*freq_i
		# (B, half)
		theta = t.float()[:, None] * freqs[None]

		# (B, 2*half)
		# cos(t*f_0)...cos(t*f_half-1)...sin(t*f_0)...sin(t*f_half-1)
		return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)

	def forward(self, t):
		t = self.sinusoidal(t)
		t = self.mlp(t)
		return t

class FinalLayer(nn.Module):
	def __init__(self, C, out_dim):
		super().__init__()
		self.ln = nn.LayerNorm(C, elementwise_affine=False, eps=1e-6)
		self.adaLN = nn.Sequential( # same tiny mlp to find scale,shift
			nn.SiLU(),
			nn.Linear(C, 2 * C)
		)
		self.linear = nn.Linear(C, out_dim)

		# init zero for all weights AKA identity
		for m in (self.adaLN[-1], self.linear):
			nn.init.zeros_(m.weight)
			nn.init.zeros_(m.bias)
	
	def forward(self, x, c):
		scale, shift = self.adaLN(c).chunk(2, dim=-1)
		x = self.ln(x)
		x = modulate(x, shift, scale)
		return self.linear(x)


class Block(nn.Module):
	def __init__(self, C, nheads):
		super().__init__()

		self.heads = nn.MultiheadAttention(C, nheads, batch_first=True)
		self.mlp = nn.Sequential(
			nn.Linear(C, C*4),
			nn.GELU(approximate="tanh"), # better for VITs
			nn.Linear(C*4, C),
		)

		self.ln1 = nn.LayerNorm(C, elementwise_affine=False, eps=1e-6)
		self.ln2 = nn.LayerNorm(C, elementwise_affine=False, eps=1e-6)

		self.adaLN = nn.Sequential(
			nn.SiLU(),
			nn.Linear(C, 6 * C)
		)
		nn.init.zeros_(self.adaLN[-1].weight)
		nn.init.zeros_(self.adaLN[-1].bias)

	def forward(self, x, c):
		# c_emb is  label + timestep info embedded
		# its contribution is to find scale, shift, gains throughout the blocks
		# like at this timestep, label. what parts matter right now?
		# mixing board, which parts matter
		s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)

		# layer norm to normalize last dim with mean=0 and var=1
		h = self.ln1(x)
		# now scale last dim with mean=s1, var=1+sc1
		h = modulate(h, s1, sc1) 

		out, _ = self.heads(h, h, h, need_weights=False)

		# res connection + gain * last dim
		x = x + g1.unsqueeze(1) * out

		h = self.ln2(x) # mean=0, var=1 on last dim
		h = modulate(h, s2, sc2) # mean=s2, var=1+sc2

		# res, gain on mlp embed of h
		x = x + g2.unsqueeze(1) * self.mlp(h)

		return x


class DiT(nn.Module):
	def __init__(self, channels=3, H=32, W=32, patch_per_side=16,
		embedding_size=192, nheads=4, nblocks=6, num_classes=10):
		super().__init__()

		self.P, self.C = patch_per_side, embedding_size

		self.channels, self.H, self.W = channels, H, W

		P = self.P
		EMB = self.C

		assert EMB % nheads == 0
		self.d = EMB // nheads


		self.emb = nn.Linear(self.channels * (H//P)  * (W//P), EMB)
		self.pos = nn.Embedding(P*P, EMB)
		nn.init.trunc_normal_(self.pos.weight, std=0.02)

		self.y_emb = nn.Embedding(num_classes+1, EMB)
		self.t_emb = TEmbed(EMB)
		nn.init.trunc_normal_(self.y_emb.weight, std=0.02)

		self.blocks = nn.ModuleList([Block(EMB, nheads) for _ in range(nblocks)])

		self.final = FinalLayer(EMB, channels*(H//P)*(W//P))
	
	def forward(self, x_t, t, y):
		B, channels, H, W = x_t.shape
		assert channels == self.channels and H == self.H and W == self.W

		# LOOK at ViT
		P = self.P
		patches = x_t.reshape(B, channels, P, H // P, P, W // P)
		patches = patches.permute(0, 2, 4, 1, 3, 5)
		patches = patches.reshape(B, P * P, -1)

		# (B, P*P, ch*(H/P)*(W/P))
		x = self.emb(patches) + self.pos(torch.arange(P*P, device=x_t.device))

		c = self.y_emb(y) + self.t_emb(t)

		for block in self.blocks:
			x = block(x, c)

		noise = self.final(x, c)

		# unpermute
		noise = noise.reshape(B, P, P, channels, H // P, W // P)
		noise = noise.permute(0, 3, 1, 4, 2, 5)
		noise = noise.reshape(B, channels, H, W)
		return noise
