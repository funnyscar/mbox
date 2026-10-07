# Minimal single-GPU training on Modal, with throughput (tok/s) and MFU.
import time, torch, torch.nn as nn
import modal

app = modal.App("chinesegpt")
image = (modal.Image.debian_slim()
	.pip_install("torch", "transformers", "sentencepiece", "datasets")
	.add_local_python_source("model"))
vol = modal.Volume.from_name("chinese", create_if_missing=True)

T, B, STEPS = 512, 32, 4000  # context, batch, steps
PEAK_FLOPS = 70e12  # A100 bf16 dense (H100 SXM: 989e12)
LOG = "/data/metrics-single.jsonl"  # one JSON object per line

def batches(ds, enc):
	eot, n, buf = enc.eos_token_id, B * (T + 1), []
	for row in ds:
		buf += enc.encode(row["text"], add_special_tokens=False) + [eot]
		while len(buf) >= n:
			x = torch.tensor(buf[:n], device="cuda").view(B, T + 1)
			buf = buf[n:]
			yield x[:, :-1], x[:, 1:]


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=4 * 3600,
	secrets=[modal.Secret.from_name("huggingface-secret")])
def train():
	from model import ChineseGPT
	from transformers import AutoTokenizer
	from datasets import load_dataset
	import json

	enc = AutoTokenizer.from_pretrained("01-ai/Yi-6B")
	data = batches(load_dataset("Skywork/SkyPile-150B", split="train", streaming=True), enc)

	vocab = -(-len(enc) // 64) * 64
	model = ChineseGPT(context_size=T, vocab_size=vocab,
        embedding_size=768, nheads=12, nblocks=12
    ).cuda()
	nparams = sum(p.numel() for p in model.parameters())
	flops_per_tok = 6 * nparams + 12 * 12 * 768 * T  # 6N + attention (12 * L * C * T)

	opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
	torch.set_float32_matmul_precision("high")
	cmodel = torch.compile(model)

	for step in range(STEPS):
		x, y = next(data)
		with torch.autocast("cuda", dtype=torch.bfloat16):
			loss = nn.functional.cross_entropy(cmodel(x).mT, y)
		opt.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(cmodel.parameters(), 1.0)
		opt.step()

		if step == 20:
			torch.cuda.synchronize(); t0 = time.time()
		if step > 20 and step % 100 == 0:
			torch.cuda.synchronize()
			tok_s = (step - 20) * B * T / (time.time() - t0)
			mfu = tok_s * flops_per_tok / PEAK_FLOPS
			with open(LOG, "a") as f:
				f.write(json.dumps({"step": step, "loss": loss.item(), "tok_s": tok_s, "mfu": mfu}) + "\n")
			vol.commit()



@app.local_entrypoint()
def main():
	train.remote()