# This is tiny-shakespeare but we are intentionally trying to overfit the text and watch 
# the val/train loss diverge. Also we are trying out separating the files into this format for modal
import os
import torch
import torch.nn as nn
import tiktoken
import urllib.request
import modal
import json

app = modal.App("minigpt")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "tiktoken", "numpy")
		.add_local_python_source("minigpt_02")
)

# Persistent storage for the dataset and checkpoints
vol = modal.Volume.from_name("minigpt", create_if_missing=True)

context_size = 256
batchsize = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data():
	url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	text = urllib.request.urlopen(url).read().decode("utf-8")
	n = int(0.9 * len(text))

	enc = tiktoken.get_encoding('gpt2')

	data = torch.tensor(enc.encode(text[:n]), dtype=torch.long)
	val_data = torch.tensor(enc.encode(text[n:]), dtype=torch.long)
	return data, val_data, enc

def get_batch(d):
	# grabbing random 128 contexts
	ix = torch.randint(len(d) - context_size - 1, (batchsize,))
	x = torch.stack([d[i:i+context_size] for i in ix])
	y = torch.stack([d[i+1:i+context_size+1] for i in ix])  # shifted by 1 token
	return x.to(device), y.to(device)


@app.function(image=image, gpu="A10G", volumes={"/ckpts": vol}, timeout=60 * 60)
def train(epochs=5000, out_path="/ckpts/minigpt-0.2.pt"):
	from minigpt_02 import MiniGPT

	data, val_data, enc = get_data()

	mymodel = MiniGPT(vocab_size=enc.n_vocab).to(device)

	old = "/ckpts/minigpt-0.2.pt"
	if os.path.exists(old):
		mymodel.load_state_dict(torch.load(old, map_location=device))
		print(f"Loaded {old}", flush=True)
	mymodel = torch.compile(mymodel)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(mymodel.parameters()) 


	best_val = float('inf')
	lossi = []
	vlossi = []

	for epoch in range(epochs):

		tokens, targets = get_batch(data)
		logits = mymodel(tokens)
		loss = criterion(logits.mT, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		lossi.append(loss.item())
		print(epoch, loss.item())

		if epoch % 250 == 0:
			mymodel.eval()
			with torch.no_grad():
				tokens, targets = get_batch(val_data)
				logits = mymodel(tokens)
				val_loss = criterion(logits.mT, targets).item()
			mymodel.train()
			vlossi.append(val_loss)
			print(epoch, loss.item(), val_loss)

			if val_loss < best_val:
				best_val = val_loss
				torch.save(mymodel._orig_mod.state_dict(), out_path)
				vol.commit()
				print("best val: ", val_loss)


	with open("/ckpts/losses-0.2.json", "w") as f:
		json.dump({"train": lossi, "val": vlossi}, f)
		vol.commit()

	mymodel.eval()
	with torch.no_grad():
		tokens, targets = get_batch(val_data)
		logits = mymodel(tokens)
		val_loss = criterion(logits.mT, targets)
		print("val loss: ", val_loss.item())

	print("train loss: ", loss.item())

	return loss.item(), val_loss.item()



@app.local_entrypoint()
def main():
	print("final loss:", train.remote())


