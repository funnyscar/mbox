# need to make changes to take into the larger dataset
# pretty much just sharding on a large dataset
import os
import torch
import torch.nn as nn
import modal
import json

app = modal.App("chinesegpt")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "transformers", "sentencepiece", "datasets")
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("chinese", create_if_missing=True)

context_size = 512
batchsize = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_token_stream(ds, enc, EOT):
	for row in ds:
		yield from enc.encode(row["text"], add_special_tokens=False) + [EOT]

def get_batch_stream(ds, enc, EOT):
	token_generator = get_token_stream(ds, enc, EOT)
	n = batchsize * (context_size + 1) # ensure enough size
	buf = []
	while True:
		while len(buf) < n:
			buf.append(next(token_generator))
		# chunk within size, leave rest for next buf
		chunk, buf = buf[:n], buf[n:]

		# (B, T+1)
		batch = torch.tensor(chunk, dtype=torch.long).view(batchsize, context_size+1)
		batch = batch.to(device, non_blocking=True)
		yield batch[:, :-1], batch[:, 1:]
	


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60 * 4, secrets=[modal.Secret.from_name("huggingface-secret")])
def train(total_steps=10000, model_version="h.01"):
	from model import ChineseGPT
	from transformers import AutoTokenizer
	import torch
	from datasets import load_dataset
	import os
	os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

	out_path = f"/data/chinese-{model_version}.pt"
	loss_path = f"/data/losses-{model_version}.json"

	enc = AutoTokenizer.from_pretrained("01-ai/Yi-6B")
	EOT = enc.eos_token_id

	ds = load_dataset("Skywork/SkyPile-150B", split="train", streaming=True)
	val_ds = ds.take(1e4)
	train_ds = ds.skip(1e4)

	n_vocab = (-len(enc) % 64) + len(enc)

	mymodel = ChineseGPT(
		context_size=context_size,
		vocab_size=n_vocab,
		# change this line for study
		# embedding_size=128, nheads=2,  nblocks=4	  # 0.8M a
		# embedding_size=192, nheads=3,  nblocks=4	  # 1.8M b 
		# embedding_size=192, nheads=3,  nblocks=6    # 2.7M c 
		# embedding_size=256, nheads=4,  nblocks=6    # 4.7M d 
		# embedding_size=256, nheads=4,  nblocks=8    # 6.3M e 
		# embedding_size=320, nheads=5,  nblocks=8    # 9.8M f 
		# embedding_size=384, nheads=6,  nblocks=8    # 14M g 
		embedding_size=448, nheads=7,  nblocks=9    # 22M h
		# embedding_size=512, nheads=8,  nblocks=10   # 31M i 
		# embedding_size=576, nheads=9,  nblocks=11   # 44M j
		# embedding_size=640, nheads=10, nblocks=12   # 59M k
		# embedding_size=768, nheads=12, nblocks=12   # 85M l
		# embedding_size=832, nheads=13, nblocks=13   # 108M m
		# embedding_size=896, nheads=14, nblocks=14   # 135M n
		# embedding_size=960,  nheads=15, nblocks=15   # 166M o
		# embedding_size=1024, nheads=16, nblocks=16   # 201M p (~24 hrs A100)
		# embedding_size=1024, nheads=16, nblocks=18   # ~226M q 
		# embedding_size=1024, nheads=16, nblocks=20   # ~250 r 
		# embedding_size=1280, nheads=20, nblocks=20   # ~393M s (~24 hrs H100)

	).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(mymodel.parameters(), lr=1e-3, fused=True) 

	start = 0
	loss = None
	best_val = float('inf')

	old = out_path
	if os.path.exists(old):
		ck = torch.load(old, map_location=device)
		mymodel.load_state_dict(ck["model"])
		optimizer.load_state_dict(ck["optim"])
		start, best_val = ck["step"] + 1, ck["best_val"]
		train_ds = train_ds.shuffle(seed=start, buffer_size=10_000)
		print(f"Loaded {old}", flush=True)

	torch.set_float32_matmul_precision("high")
	mymodel = torch.compile(mymodel)

	batchstream = get_batch_stream(train_ds, enc, EOT)
	valstream = get_batch_stream(val_ds, enc, EOT)
	val_batches = [next(valstream) for _ in range(10)]

	lossi = []
	vlossi = []

	def eval_loss():
		mymodel.eval()
		with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
			val_loss = 0
			for x, y in val_batches:
				logits = mymodel(x)
				val_loss += criterion(logits.mT, y).item()
			val_loss /= len(val_batches)
		mymodel.train()
		return val_loss

	for step in range(start, total_steps):
		tokens, targets = next(batchstream)

		assert tokens.min() >= 0, "negative token"
		assert tokens.max() < n_vocab, "token id out of bounds"

		with torch.autocast("cuda", dtype=torch.bfloat16):
			logits = mymodel(tokens)
			loss = criterion(logits.mT, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		lossi.append(loss.item())

		if step % 100 == 0:
			val_loss = eval_loss()
			vlossi.append(val_loss)

			if val_loss < best_val:
				best_val = val_loss
				torch.save({
					"model": mymodel._orig_mod.state_dict(),
					"optim": optimizer.state_dict(),
					"step": step,
					"best_val": best_val,
					}, out_path)
				
				vol.commit()
				print("best val: ", val_loss)

			print(step, loss.item())


	with open(loss_path, "w") as f:
		n_emb = sum(p.numel() for m in mymodel.modules() if isinstance(m, nn.Embedding) for p in m.parameters())
		nparams = sum(p.numel() for p in mymodel.parameters()) - n_emb
		json.dump({
			"train": lossi, 
			"val": vlossi,
			"out_path": out_path,
			"nparams":  nparams,
			"vocab_size": n_vocab,
			"context_size": context_size,
			"embedding_size": mymodel._orig_mod.C,
			"nheads": mymodel._orig_mod.nheads,
			"nblocks": mymodel._orig_mod.nblocks
		}, f)
	vol.commit()

	val_loss = eval_loss()
	print("val loss: ", val_loss)
	print("train loss: ", loss.item())

	return loss.item(), val_loss

@app.local_entrypoint()
def main():
	print("final loss:", train.remote())
