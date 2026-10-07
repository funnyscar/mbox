# need to make changes to take into the larger dataset
# pretty much just sharding on a large dataset
import os
import torch
import torch.nn as nn
import tiktoken
import urllib.request
import modal
import json

app = modal.App("genz_10")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "tiktoken", "numpy")
		.add_local_python_source("genz_10")
)

vol = modal.Volume.from_name("genz", create_if_missing=True)

context_size = 512
batchsize = 64
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
n_vocab = -(-enc.n_vocab // 64) * 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_shard():
	import numpy as np
	import glob
	import os

	# get random shard
	files = sorted(glob.glob("/data/big/*"))
	sizes = np.array([os.path.getsize(f) for f in files], dtype=float)
	p = sizes / sizes.sum()
	f = np.random.choice(files, p=p)

	data = torch.from_numpy(np.fromfile(f, dtype=np.uint16).astype(np.int64)).to(device)
	n = int(0.9 * len(data))
	train_data = data[:n]
	val_data = data[n:]
	return train_data, val_data

def get_starts(d):
	eot_pos = (d == EOT).nonzero().squeeze(1)
	return eot_pos[eot_pos < len(d) - context_size - 1]

def get_batch(eotp, d):
	rstarts = eotp[torch.randint(len(eotp), (batchsize,), device=d.device)]
	ctx_slice = torch.arange(context_size+1, device=d.device)
	batch = d[rstarts[:, None] + ctx_slice]
	return batch[:, :-1], batch[:, 1:]


@app.function(image=image, gpu="A100", volumes={"/data": vol}, timeout=60 * 60 * 22)
def train(steps=10000, epochs=50, model_version="l"):
	from genz_10 import GenZGPT

	out_path = f"/data/genz-{model_version}.pt"
	latest_path = f"/data/genz-{model_version}-latest.pt"
	loss_path = f"/data/losses-{model_version}.json"

	mymodel = GenZGPT(
		context_size=context_size, vocab_size=n_vocab,
		embedding_size=768, nheads=12, nblocks=12   # 85M l
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
		print(f"Loaded {old}", flush=True)

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
				val_loss += criterion(logits.mT, y).item()
			val_loss /= len(val_batches)
		mymodel.train()
		return val_loss

	for epoch in range(epochs):

		data, val_data = get_shard()
		eot_pos, val_eot_pos = get_starts(data), get_starts(val_data)

		val_batches = [get_batch(val_eot_pos, val_data) for _ in range(20)]

		for step in range(start, steps):

			tokens, targets = get_batch(eot_pos, data)

			with torch.autocast("cuda", dtype=torch.bfloat16):
				logits = mymodel(tokens)
				loss = criterion(logits.mT, targets)

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
			optimizer.step()


			if step % 500 == 0:
				print(step, loss.item())
				val_loss = eval_loss()
				vlossi.append(val_loss)
				lossi.append(loss.item())

				ckpt = {
					"model": mymodel._orig_mod.state_dict(),
					"optim": optimizer.state_dict(),
					"step": step,
					"best_val": best_val,
				}

				torch.save(ckpt, latest_path)

				if val_loss < best_val:
					best_val = val_loss
					torch.save(ckpt, out_path)
					print("best val: ", val_loss)
					
				vol.commit()


	with open(loss_path, "w") as f:
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
			"nblocks": mymodel._orig_mod.nblocks
		}, f)
	vol.commit()

	val_loss = eval_loss()
	print("val loss: ", val_loss)

	print("train loss: ", loss.item())
	print("n model params: ", sum(p.numel() for p in mymodel.parameters() if p.requires_grad))

	return loss.item(), val_loss



@app.local_entrypoint()
def main():
	print("final loss:", train.remote())


