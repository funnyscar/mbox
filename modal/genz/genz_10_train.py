# need to make changes to take into the larger dataset
# pretty much just sharding on a large dataset
import os
import torch
import torch.nn as nn
import tiktoken
import modal
import json
import numpy as np

app = modal.App("genz_10")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "tiktoken", "numpy")
		.add_local_python_source("genz_10")
)

vol = modal.Volume.from_name("genz", create_if_missing=True)

context_size = 512
batchsize = 32
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
n_vocab = -(-enc.n_vocab // 64) * 64

N_VAL_FILES = 5          # whole shards held out as a fixed val set
SHARDS_PER_GROUP = 50    # shards sampled from at once
RELOAD_EVERY = 1000      # steps before swapping in a fresh group

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def split_files():
	import glob
	files = sorted(glob.glob("/data/big/*"))
	assert len(files) > N_VAL_FILES, f"only {len(files)} shards found"
	return files[N_VAL_FILES:], files[:N_VAL_FILES]

def load_group(files, k):
	files = np.random.choice(files, size=min(k, len(files)), replace=False)
	shards, starts = [], []
	for f in files:
		d = np.memmap(f, dtype=np.uint16, mode="r")
		s = get_starts(d)
		if len(s):
			shards.append(d)
			starts.append(s)
	return shards, starts

def get_starts(d):
	eot_pos = np.flatnonzero(d == EOT)
	return eot_pos[eot_pos < len(d) - context_size - 1]

def get_batch(shards, starts):
	rows = []
	for s in np.random.randint(len(shards), size=batchsize):   # random shard per sample
		r = starts[s][np.random.randint(len(starts[s]))]        # random doc start in that shard
		rows.append(shards[s][r : r + context_size + 1])
	batch = torch.from_numpy(np.stack(rows).astype(np.int64)).pin_memory().to(device, non_blocking=True)
	return batch[:, :-1], batch[:, 1:]




@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60 * 22)
def train(steps=50000, model_version="l.01"):
	from genz_10 import GenZGPT
	from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

	out_path = f"/data/genz-{model_version}.pt"
	latest_path = f"/data/genz-{model_version}-latest.pt"
	loss_path = f"/data/losses-{model_version}.jsonl"

	train_files, val_files = split_files()
	np.random.seed(0)
	val_shards, val_starts = load_group(val_files, len(val_files))
	val_batches = [get_batch(val_shards, val_starts) for _ in range(20)]
	np.random.seed(None)

	mymodel = GenZGPT(
		context_size=context_size, vocab_size=n_vocab,
		embedding_size=768, nheads=12, nblocks=12   # 85M l
	).to(device)

	criterion = nn.CrossEntropyLoss()

	lr = 6e-4
	decay    = [p for p in mymodel.parameters() if p.requires_grad and p.dim() >= 2] # matrices
	no_decay = [p for p in mymodel.parameters() if p.requires_grad and p.dim() < 2] # biases, ln
	optimizer = torch.optim.AdamW([
		{"params": decay, "weight_decay": 0.1},
    	{"params": no_decay, "weight_decay": 0.0}
	], lr=lr, betas=(0.9, 0.95), fused=True)

	warmup = 500
	scheduler = SequentialLR(
		optimizer,
		schedulers=[
			LinearLR(optimizer, start_factor=1e-3, total_iters=warmup),
			CosineAnnealingLR(optimizer, T_max=steps - warmup, eta_min=3e-5),
		],
		milestones=[warmup],
	)

	start = 0
	loss = None
	best_val = float('inf')

	old = latest_path
	if os.path.exists(old):
		ck = torch.load(old, map_location=device)
		mymodel.load_state_dict(ck["model"])
		optimizer.load_state_dict(ck["optim"])
		scheduler.load_state_dict(ck["sched"]) 
		start, best_val = ck["step"] + 1, ck["best_val"]
		print(f"Loaded {old}", flush=True)
	
	assert start < steps

	torch.set_float32_matmul_precision("high")
	mymodel = torch.compile(mymodel)


	lossi = []
	vlossi = []


	def eval_loss():
		mymodel.eval()
		with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
			val_loss = 0
			for x, y in val_batches:
				logits = mymodel(x)
				val_loss += criterion(logits.reshape(-1, n_vocab), y.reshape(-1)).item()
			val_loss /= len(val_batches)
		mymodel.train()
		return val_loss
	
	shards, starts = load_group(train_files, SHARDS_PER_GROUP)


	for step in range(start, steps):
		if step != start and step % RELOAD_EVERY == 0:
			shards, starts = load_group(train_files, SHARDS_PER_GROUP)

		tokens, targets = get_batch(shards, starts)

		with torch.autocast("cuda", dtype=torch.bfloat16):
			logits = mymodel(tokens)
			loss = criterion(logits.reshape(-1, n_vocab), targets.reshape(-1))

		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
		optimizer.step()
		scheduler.step()


		if step % 2000 == 0:
			print(step, loss.item())
			val_loss = eval_loss()
			vlossi.append(val_loss)
			lossi.append(loss.item())

			prev_best_val = best_val
			best_val = min(val_loss, best_val)

			ckpt = {
				"model": mymodel._orig_mod.state_dict(),
				"optim": optimizer.state_dict(),
				"step": step,
				"best_val": best_val,
				"sched": scheduler.state_dict(),
			}

			torch.save(ckpt, latest_path)

			if val_loss < prev_best_val:
				torch.save(ckpt, out_path)
				print("best val: ", val_loss)
				
			vol.commit()



	val_loss = eval_loss()

	with open(loss_path, "a") as f:
		n_emb = sum(p.numel() for m in mymodel.modules() if isinstance(m, nn.Embedding) for p in m.parameters())
		nparams = sum(p.numel() for p in mymodel.parameters()) - n_emb
		json.dump({
			"train": lossi, "val": vlossi,
			"out_path": out_path,
			"nparams":  nparams,
			"vocab_size": n_vocab,
			"context_size": context_size,
			"embedding_size": mymodel._orig_mod.C,
			"nheads": mymodel._orig_mod.nheads,
			"nblocks": mymodel._orig_mod.nblocks,
			"final_val": val_loss,
			"final_train": loss.item(),
		}, f)
		f.write("\n")
	vol.commit()

	print("val loss: ", val_loss)
	print("train loss: ", loss.item())
	print("n model params: ", sum(p.numel() for p in mymodel.parameters() if p.requires_grad))

	return loss.item(), val_loss



@app.local_entrypoint()
def main():
	print("final", train.remote())


