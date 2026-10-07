# ViT for spectrograms
# audio -> mellog spectrogram -> pretend as image -> ViT -> Classify sounds

import modal

app = modal.App("audionet")

image = (
	modal.Image
		.debian_slim()
		.apt_install("ffmpeg")
		.pip_install("torch==2.8.0", "torchaudio==2.8.0", "torchcodec==0.7.0", "datasets", "transformers")
		.env({"HF_HOME": "/data/hf"})
)

vol = modal.Volume.from_name("audionet", create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")
CKPT = "/data/ast.pt"


@app.function(image=image, volumes={"/data": vol}, secrets=[hf], cpu=8, timeout=24 * 60 * 60)
def download(config: str = "balanced"):
	from datasets import load_dataset
	ds = load_dataset("agkphysics/AudioSet", config, num_proc=8)
	print(ds)
	vol.commit()
	
@app.function(image=image, volumes={"/data": vol}, secrets=[hf], gpu="A100", cpu=8, timeout=24 * 60 * 60)
def train(epochs: int = 200, bs: int = 128):
	import torch, torch.nn as nn, torch.nn.functional as F, torchaudio
	from torch.utils.data import DataLoader
	from datasets import load_dataset, Audio
	from transformers.models.vit.modeling_vit import ViTModel

	SR, N, T = 16000, 160000, 1024
	bs, epochs, lr = 32, 10, 3e-5
	dev = "cuda"

	ds = load_dataset("agkphysics/AudioSet", "balanced", split="train").cast_column("audio", Audio(sampling_rate=SR))
	classes = sorted({l for ls in ds["labels"] for l in ls})
	idx = {c: i for i, c in enumerate(classes)}

	mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, win_length=400, hop_length=160, n_mels=128, f_max=8000)
	to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

	def load_wav(a):
		w = torch.as_tensor(a["array"], dtype=torch.float32)
		if w.ndim > 1: w = w.mean(0)
		return F.pad(w[:N], (0, max(0, N - w.numel())))

	def collate(batch):
		spec = to_db(mel(torch.stack([load_wav(b["audio"]) for b in batch])) + 1e-10)
		spec = F.pad(spec, (0, T - spec.shape[-1]), value=spec.min())
		spec = (spec - spec.mean((1, 2), keepdim=True)) / (spec.std((1, 2), keepdim=True) + 1e-5)
		y = torch.zeros(len(batch), len(classes))
		for i, b in enumerate(batch): y[i, [idx[l] for l in b["labels"]]] = 1
		return spec.unsqueeze(1).repeat(1, 3, 1, 1), y

	class AST(nn.Module):
		def __init__(s, n):
			super().__init__()
			s.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", add_pooling_layer=False)
			s.head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, n))

		def forward(s, pix):
			return s.head(s.vit(pixel_values=pix, interpolate_pos_encoding=True).last_hidden_state[:, 0])

	dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=8, drop_last=True,
		collate_fn=collate, persistent_workers=True)

	model = AST(len(classes)).to(dev)
	opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
	sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.05)

	for ep in range(epochs):
		for i, (pix, y) in enumerate(dl):
			with torch.autocast("cuda", dtype=torch.bfloat16):
				loss = F.binary_cross_entropy_with_logits(model(pix.to(dev)), y.to(dev))
			opt.zero_grad(); loss.backward(); opt.step(); sched.step()
			if i % 50 == 0: print(f"ep {ep} step {i} loss {loss.item():.4f}")
		torch.save(model.state_dict(), f"/data/ast_ep{ep}.pt")


@app.local_entrypoint()
def main():
	train.remote()



