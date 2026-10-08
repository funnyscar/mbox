# Minimal single-GPU training on Modal, with throughput (tok/s) and MFU.
import time, torch, torch.nn as nn
import modal

app = modal.App("chinesegpt")
image = (modal.Image.debian_slim()
	.pip_install("torch", "transformers", "sentencepiece", "datasets")
	.add_local_python_source("model"))
vol = modal.Volume.from_name("chinese", create_if_missing=True)

T, B, STEPS = 512, 32, 1500  # context, batch, steps

def batches(ds, enc):
	eot, n, buf = enc.eos_token_id, B * (T + 1), []
	for row in ds:
		buf += enc.encode(row["text"], add_special_tokens=False) + [eot]
		while len(buf) >= n:
			x = torch.tensor(buf[:n], device="cuda").view(B, T + 1)
			buf = buf[n:]
			yield x[:, :-1], x[:, 1:]


@app.function(image=image, gpu="A100", volumes={"/data": vol}, timeout=60*60*4,
	secrets=[modal.Secret.from_name("huggingface-secret")])
def train(gpu_type="A100"):
	PEAK_FLOPS = 312e12  # 312e12 A100 bf16 dense (H100 SXM: 989e12)
	LOG = f"/data/metric-{gpu_type}.jsonl"  # one JSON object per line

	from model import ChineseGPT
	from transformers import AutoTokenizer
	from datasets import load_dataset
	import json

	enc = AutoTokenizer.from_pretrained("01-ai/Yi-6B")
	data = batches(load_dataset("Skywork/SkyPile-150B", split="train", streaming=True), enc)

	vocab = -(-len(enc) // 64) * 64

	model_sizes = [
		{"embedding_size":128, "nheads":2,  "nblocks":4},	  # 0.8M a
		{"embedding_size":192, "nheads":3,  "nblocks":4},	  # 1.8M b 
		{"embedding_size":192, "nheads":3,  "nblocks":6},    # 2.7M c 
		{"embedding_size":256, "nheads":4,  "nblocks":6},    # 4.7M d 
		{"embedding_size":256, "nheads":4,  "nblocks":8},    # 6.3M e 
		{"embedding_size":320, "nheads":5,  "nblocks":8},    # 9.8M f 
		{"embedding_size":384, "nheads":6,  "nblocks":8},    # 14M g 
		{"embedding_size":448, "nheads":7,  "nblocks":9},    # 22M h
		{"embedding_size":512, "nheads":8,  "nblocks":10},   # 31M i 
		{"embedding_size":576, "nheads":9,  "nblocks":11},   # 44M j
		{"embedding_size":640, "nheads":10, "nblocks":12},   # 59M k
		{"embedding_size":768, "nheads":12, "nblocks":12},   # 85M l 11
		{"embedding_size":832, "nheads":13, "nblocks":13},   # 108M m
		{"embedding_size":896, "nheads":14, "nblocks":14},   # 135M n
		{"embedding_size":960,  "nheads":15, "nblocks":15},   # 166M o
		{"embedding_size":1024, "nheads":16, "nblocks":16},   # 201M p (~24 hrs A100)
		{"embedding_size":1024, "nheads":16, "nblocks":18},   # ~226M q 
		{"embedding_size":1024, "nheads":16, "nblocks":20},   # ~250 r 
		{"embedding_size":1280, "nheads":20, "nblocks":20}   # ~393M s (~24 hrs H100)
	]

	for i, ms in enumerate(model_sizes):
		model = ChineseGPT(context_size=T, vocab_size=vocab, **ms).cuda()
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
					f.write(json.dumps({
						"model": chr(i + ord("a")), 
						"step": step, 
						"loss": loss.item(),
						"tok_s": tok_s,
						"mfu": mfu
					}) + "\n")
				vol.commit()



@app.local_entrypoint()
def main():
	train.remote()