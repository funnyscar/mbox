# This version we are focused on speed and efficiency of this model before
# This is still a very small training set
import os
import torch
import torch.nn as nn
import tiktoken
import urllib.request
import modal
import json

app = modal.App("genz_00")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "tiktoken", "numpy")
		.add_local_python_source("genz_00")
)

# Persistent storage for the dataset and checkpoints
vol = modal.Volume.from_name("genz", create_if_missing=True)

context_size = 128
batchsize = 32
EOT = tiktoken.get_encoding("gpt2").eot_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data():
	import numpy as np
	data = torch.from_numpy(np.fromfile("/data/genz_00_train.bin", dtype=np.uint16).astype(np.int64)).to(device)
	n = int(0.9 * len(data))
	train_data = data[:n]
	val_data = data[n:]
	return train_data, val_data

def get_starts(d):
	eot_pos = (d == EOT).nonzero().squeeze(1)
	eot_pos = torch.cat([torch.tensor([0], device=d.device), eot_pos])
	return eot_pos[eot_pos < len(d) - context_size - 1]

def get_batch(eotp, d):
	rstarts = eotp[torch.randint(len(eotp), (batchsize,), device=d.device)] # (B,)
	ctx_slice = torch.arange(context_size+1, device=d.device) # (T+1,)

	batch = d[rstarts[:, None] + ctx_slice] # (B,1) + (T+1,) = (B, T+1)

	# (B, 0-T), (B, 1 - T+1)
	return batch[:, :-1], batch[:, 1:]


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60)
def train(steps=5000, out_path="/data/genz-0.1.pt"):
	from genz_00 import GenZGPT

	data, val_data = get_data()
	eot_pos, val_eot_pos = get_starts(data), get_starts(val_data)

	mymodel = GenZGPT().to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(mymodel.parameters(), lr=1e-3, fused=True) 

	start = 0
	loss = None
	best_val = float('inf')

	old = "/data/genz-0.1.pt"
	if os.path.exists(old):
		ck = torch.load(old, map_location=device)
		mymodel.load_state_dict(ck["model"])
		optimizer.load_state_dict(ck["optim"])
		start, best_val = ck["step"] + 1, ck["best_val"]
		print(f"Loaded {old}", flush=True)

	# speed up training by using high precision for matmul
	torch.set_float32_matmul_precision("high")
	mymodel = torch.compile(mymodel)


	lossi = []
	vlossi = []

	for step in range(start, steps):

		tokens, targets = get_batch(eot_pos, data)

		# use bfloat16 for training to speed up training and reduce memory usage
		# fp32 still used for sensitive operations like layernorm and softmax
		with torch.autocast("cuda", dtype=torch.bfloat16):
			logits = mymodel(tokens)
			loss = criterion(logits.mT, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		lossi.append(loss.item())
		if step % 100 == 0:
			print(step, loss.item())

		if step % 250 == 0:
			mymodel.eval()
			with torch.no_grad():
				tokens, targets = get_batch(val_eot_pos, val_data)
				with torch.autocast("cuda", dtype=torch.bfloat16):
					logits = mymodel(tokens)
					val_loss = criterion(logits.mT, targets).item()
			mymodel.train()
			vlossi.append(val_loss)

			if val_loss < best_val:
				best_val = val_loss
				torch.save({
					"nparams": sum(p.numel() for p in mymodel.parameters() if p.requires_grad),
					"model": mymodel._orig_mod.state_dict(),
					"optim": optimizer.state_dict(),
					"step": step,
					"best_val": best_val,
					}, out_path)
				
				vol.commit()
				print("best val: ", val_loss)


	with open("/data/losses-0.1.json", "w") as f:
		json.dump({"train": lossi, "val": vlossi}, f)
	vol.commit()

	mymodel.eval()
	with torch.no_grad():
		tokens, targets = get_batch(val_eot_pos, val_data)
		with torch.autocast("cuda", dtype=torch.bfloat16):
			logits = mymodel(tokens)
			val_loss = criterion(logits.mT, targets)
		print("val loss: ", val_loss.item())

	print("train loss: ", loss.item())
	print("n model params: ", sum(p.numel() for p in mymodel.parameters() if p.requires_grad))

	return loss.item(), val_loss.item()



@app.local_entrypoint()
def main():
	print("final loss:", train.remote())


