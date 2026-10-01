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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data():
	import numpy as np
	data = torch.from_numpy(np.fromfile("/data/genz_00_train.bin", dtype=np.uint16).astype(np.int64)).to(device)
	n = int(0.9 * len(data))
	train_data = data[:n]
	val_data = data[n:]
	return train_data, val_data

def get_batch(d):
	ix = torch.randint(len(d) - context_size - 1, (batchsize,))
	x = torch.stack([d[i:i+context_size] for i in ix])
	y = torch.stack([d[i+1:i+context_size+1] for i in ix])  # shifted by 1 token
	return x.to(device), y.to(device)


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60)
def train(steps=5000, out_path="/data/genz-0.0.pt"):
	from genz_00 import GenZGPT

	data, val_data = get_data()

	mymodel = GenZGPT().to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(mymodel.parameters(), lr=1e-3, fused=True) 

	start = 0
	loss = None
	best_val = float('inf')

	old = "/data/genz-0.0.pt"
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

		tokens, targets = get_batch(data)

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
				tokens, targets = get_batch(val_data)
				with torch.autocast("cuda", dtype=torch.bfloat16):
					logits = mymodel(tokens)
					val_loss = criterion(logits.mT, targets).item()
			mymodel.train()
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


	with open("/data/losses-0.0.json", "w") as f:
		json.dump({"train": lossi, "val": vlossi}, f)
	vol.commit()

	mymodel.eval()
	with torch.no_grad():
		tokens, targets = get_batch(val_data)
		with torch.autocast("cuda", dtype=torch.bfloat16):
			logits = mymodel(tokens)
			val_loss = criterion(logits.mT, targets)
		print("val loss: ", val_loss.item())

	print("train loss: ", loss.item())

	return loss.item(), val_loss.item()



@app.local_entrypoint()
def main():
	print("final loss:", train.remote())


