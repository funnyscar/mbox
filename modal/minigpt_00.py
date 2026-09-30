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
		.pip_install("torch", "tiktoken")
)

# Persistent storage for the dataset and checkpoints
vol = modal.Volume.from_name("minigpt", create_if_missing=True)

context_size = 128
batchsize = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# do note every time modal loads from this model it loads this txt as well
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(url).read().decode("utf-8")
n = int(0.9 * len(text))

enc = tiktoken.get_encoding('gpt2')

data = torch.tensor(enc.encode(text[:n]), dtype=torch.long)
val_data = torch.tensor(enc.encode(text[n:]), dtype=torch.long)


def get_batch(d):
	# grabbing random 128 contexts
	ix = torch.randint(len(d) - context_size - 1, (batchsize,))
	x = torch.stack([d[i:i+context_size] for i in ix])
	y = torch.stack([d[i+1:i+context_size+1] for i in ix])  # shifted by 1 token
	return x.to(device), y.to(device)


class AttentionHead(nn.Module):
	def __init__(self, C, d):
		super().__init__()
		self.B, self.T, self.C, self.d = batchsize, context_size, C, d
		self.Q, self.K, self.V = nn.Linear(self.C, self.d), nn.Linear(self.C, self.d), nn.Linear(self.C, self.d)
		self.register_buffer("mask", torch.tril(torch.ones(self.T, self.T, dtype=torch.bool))) # move with the weights

	def forward(self, x):
		q, k, v = self.Q(x), self.K(x), self.V(x)
		scores = q @ k.mT / self.d**0.5
		T = x.shape[1]
		scores = scores.masked_fill(~self.mask[:T, :T], float('-inf'))
		att = torch.softmax(scores, dim=-1)
		return att @ v

class Block(nn.Module):
	def __init__(self, C, d, nheads):
		super().__init__()

		self.C, self.d = C, d

		self.heads = nn.ModuleList([AttentionHead(self.C, self.d) for _ in range(nheads)])

		self.W0 = nn.Linear(self.C, self.C)

		self.mlp = nn.Sequential(
			nn.Linear(self.C, self.C*4),
			nn.ReLU(),
			nn.Linear(self.C*4, self.C),
			nn.Dropout(0.1)
		)
		self.ln1 = nn.LayerNorm(self.C)
		self.ln2 = nn.LayerNorm(self.C)

	def forward(self, x):
		h = self.ln1(x)
		all_heads = torch.cat([attn(h) for attn in self.heads], dim=-1)
		x = x + self.W0(all_heads)
		x = x + self.mlp(self.ln2(x))
		return x


class MiniGPT(nn.Module):
	def __init__(self, embedding_size=384, nheads=6, nblocks=6):
		super().__init__()

		self.B, self.T, self.C = batchsize, context_size, embedding_size
		self.d = self.C // nheads

		self.embeddings = nn.Embedding(enc.n_vocab, self.C)
		self.pos = nn.Embedding(self.T, self.C) 

		self.blocks = nn.ModuleList([Block(self.C, self.d, nheads) for _ in range(nblocks)])

		self.ln_f = nn.LayerNorm(self.C)
		self.head = nn.Linear(self.C, enc.n_vocab)
	
	def forward(self, tokens):
		T = tokens.shape[1]
		x = self.embeddings(tokens) + self.pos(torch.arange(T, device=tokens.device))

		for block in self.blocks:
			x = block(x)

		logits = self.head(self.ln_f(x)) 
		return logits


@app.function(image=image, gpu="A10G", volumes={"/ckpts": vol}, timeout=60 * 60)
def train(epochs=5000, out_path="/ckpts/minigpt-0.1.pt"):

	mymodel = MiniGPT().to(device)

	old = "/ckpts/minigpt.pt"
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


	with open("losses-0.1.json", "w") as f:
		json.dump({"train": lossi, "val": vlossi}, f)

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


