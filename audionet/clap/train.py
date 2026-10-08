# CLIP
# Audio AST ---\ emb
# TXT LLM   ---/

import modal
import torch.nn as nn

app = modal.App("audionet")

image = (
	modal.Image
		.debian_slim()
		.apt_install("ffmpeg")
		.pip_install("torch", "torchaudio", "torchcodec", "datasets", "transformers")
		.env({"HF_HOME": "/data/hf"})
)

vol = modal.Volume.from_name("audionet", create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")

SR, N, T = 16000, 160000, 1024
bs, epochs, lr = 64, 10, 2e-5
TXT = "bert-base-uncased"
dev = "cuda"



@app.function(image=image, volumes={"/data": vol}, secrets=[hf], gpu="A100", cpu=8, timeout=24 * 60 * 60)
def train(epochs: int = 200, bs: int = 128):
	import math, random, torch, torch.nn as nn, torch.nn.functional as F, torchaudio
	from torch.utils.data import DataLoader
	from datasets import load_dataset, Audio
	from transformers import AutoModel, AutoTokenizer
	from transformers import AutoModel, AutoTokenizer
	from transformers.models.vit.modeling_vit import ViTModel

	tok = AutoTokenizer.from_pretrained(TXT)
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
		pix = spec.unsqueeze(1).repeat(1, 3, 1, 1)
		caps = [", ".join(random.sample(b["human_labels"], len(b["human_labels"]))) for b in batch]
		txt = tok(caps, padding=True, truncation=True, max_length=64, return_tensors="pt")
		return pix, txt.input_ids, txt.attention_mask

	ds = load_dataset("agkphysics/AudioSet", "balanced", split="train").cast_column("audio", Audio(sampling_rate=SR))
	dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=8, drop_last=True,
		collate_fn=collate, persistent_workers=True)
	

	class CLAP(nn.Module):
		def __init__(s, d=512):
			super().__init__()
			s.audio = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", add_pooling_layer=False)
			s.text = AutoModel.from_pretrained(TXT, add_pooling_layer=False)
			s.ap, s.tp = nn.Linear(768, d), nn.Linear(768, d)
			s.scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

		def forward(s, pix, ids, mask):
			a = s.audio(pixel_values=pix, interpolate_pos_encoding=True).last_hidden_state[:, 0]
			t = s.text(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
			a, t = F.normalize(s.ap(a), dim=-1), F.normalize(s.tp(t), dim=-1)
			logits = s.scale.exp().clamp(max=100) * a @ t.T
			y = torch.arange(len(a), device=a.device)
			return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2

	model = CLAP().to(dev)
	opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
	sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.05)

	for ep in range(epochs):
		for i, (pix, ids, mask) in enumerate(dl):
			with torch.autocast("cuda", dtype=torch.bfloat16):
				loss = model(pix.to(dev), ids.to(dev), mask.to(dev))
			opt.zero_grad(); loss.backward(); opt.step(); sched.step()
			if i % 50 == 0: print(f"ep {ep} step {i} loss {loss.item():.4f}")
		torch.save(model.state_dict(), f"/data/clap_ep{ep}.pt")


@app.local_entrypoint()
def main():
	train.remote()



