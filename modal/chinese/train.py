# need to make changes to take into the larger dataset
# pretty much just sharding on a large dataset
import os
import torch
import torch.nn as nn
import tiktoken
import urllib.request
import modal
import json

app = modal.App("chinesegpt")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "transformers", "numpy")
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("chinese", create_if_missing=True)

context_size = 512
batchsize = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_starts(d):
	eot_pos = (d == EOT).nonzero().squeeze(1)
	return eot_pos[eot_pos < len(d) - context_size - 1]

def get_batch(eotp, d):
	rstarts = eotp[torch.randint(len(eotp), (batchsize,), device=d.device)]
	ctx_slice = torch.arange(context_size+1, device=d.device)
	batch = d[rstarts[:, None] + ctx_slice]
	return batch[:, :-1], batch[:, 1:]


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60)
def train(steps=1000, epochs=50, out_path="/data/chinese-1.0.pt"):
	from model import ChineseGPT
	from transformers import AutoTokenizer

	enc = AutoTokenizer.from_pretrained("01-ai/Yi-6B")
	EOT = enc.eot_token

	ds = load_dataset("CASIA-LM/ChineseWebText2.0", split="test", streaming=True)
	ds = ds.shuffle(seed=42, buffer_size=10_000)

	mymodel = ChineseGPT(
		context_size=context_size,
		vocab_size=enc.n_vocab,
		# change this line for study
		embedding_size=768, nheads=12, nblocks=12
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

	for epoch in epochs:
		ds.set_epoch(epoch)

		ids = [EOT]
		for row in ds.take(256):
			ids.extend(enc.encode(row["text"]) + [EOT])
		
		data = np.array(ids)
		eot_pos = get_starts(ids)

		for step in range(start, steps):

			tokens, targets = get_batch(eot_pos, data)

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
						"model": mymodel._orig_mod.state_dict(),
						"optim": optimizer.state_dict(),
						"step": step,
						"best_val": best_val,
						}, out_path)
					
					vol.commit()
					print("best val: ", val_loss)


	with open("/data/losses-1.0.json", "w") as f:
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
