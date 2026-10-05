# Image >>> latent vector <<<< original Image
import torch
import torch.nn as nn


import torch
import torch.nn as nn


class ResBlock(nn.Module):
	def __init__(self, cin, cout):
		super().__init__()
		self.net = nn.Sequential(
			nn.GroupNorm(32, cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1),
			nn.GroupNorm(32, cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1),
		)
		self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

	def forward(self, x):
		return self.skip(x) + self.net(x)


class VAE(nn.Module):
	# (B, 3, 32, 32) -> latent (B, z_ch, 16, 16) -> (B, 3, 32, 32)
	def __init__(self, ch=64, z_ch=4):
		super().__init__()

		self.enc = nn.Sequential(
			nn.Conv2d(3, ch, 3, padding=1),                      # (B, 64, 32, 32)
			ResBlock(ch, ch),
			nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1),       # (B, 128, 16, 16)
			ResBlock(2 * ch, 2 * ch),
			nn.GroupNorm(32, 2 * ch), nn.SiLU(),
		)
		self.mu = nn.Conv2d(2 * ch, z_ch, 1)                     # (B, 4, 16, 16)
		self.logvar = nn.Conv2d(2 * ch, z_ch, 1)

		self.dec = nn.Sequential(
			nn.Conv2d(z_ch, 2 * ch, 3, padding=1),               # (B, 128, 16, 16)
			ResBlock(2 * ch, 2 * ch),
			nn.Upsample(scale_factor=2, mode="nearest"),         # (B, 128, 32, 32)
			nn.Conv2d(2 * ch, ch, 3, padding=1),                 # (B, 64, 32, 32)
			ResBlock(ch, ch),
			nn.GroupNorm(32, ch), nn.SiLU(),
			nn.Conv2d(ch, 3, 3, padding=1),                      # (B, 3, 32, 32)
		)

	def encode(self, x):
		h = self.enc(x)
		return self.mu(h), self.logvar(h).clamp(-30, 20)

	def decode(self, z):
		return self.dec(z)

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterization
		return self.decode(z), mu, logvar