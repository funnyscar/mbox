# Spit out prompts > get answers from chat
# Feed that data into a temporary dataset to learn from in one batch
# Keep going so and so many iterations.

import os
import torch
import torch.nn as nn
import tiktoken
import modal
import json
import math

app = modal.App("metalearning")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "tiktoken", "numpy", "openrouter")
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("meta", create_if_missing=True)

context_size = 128
batchsize = 32

enc = tiktoken.get_encoding("o200k_base")
EOT = enc.eot_token
EOP = enc.encode_single_token("<|endofprompt|>")
n_vocab = -(-enc.n_vocab // 64) * 64


device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.function(image=image, cpu=4, volumes={"/data": vol}, timeout=60 * 60 * 8, secrets=[modal.Secret.from_name("metalearning")])
def data_worker(worker_i, state_dict):
	import os
	import torch
	from openrouter import OpenRouter
	import numpy as np
	from model import MainGPT
	enc = tiktoken.get_encoding("o200k_base")

	os.makedirs("/data/shards", exist_ok=True)

	@torch.no_grad()
	def generate(model, toks, max_new=64, temp=1.0):
		m = getattr(model, "_orig_mod", model)
		was = m.training
		m.eval()
		x = torch.tensor([toks], device=device)
		out = []
		for _ in range(max_new):
			logits = m(x[:, -context_size:])[:, -1].float() / temp
			logits[:, enc.n_vocab:] = -float("inf")
			nxt = torch.multinomial(logits.softmax(-1), 1)
			t = nxt.item()
			if t in (EOP, EOT):
				break
			out.append(t)
			x = torch.cat([x, nxt], 1)
		m.train(was)
		return enc.decode(out)	
	
	model_path = f"/data/meta-{state_dict['old_model_version']}.pt"
	mymodel = MainGPT(
		context_size=context_size, vocab_size=n_vocab,
		embedding_size=768, nheads=12, nblocks=12
	).to(device)


	if os.path.exists(model_path):
		ck = torch.load(model_path, map_location=device)
		mymodel.load_state_dict(ck["model"])
		print(f"Loaded {state_dict['old_model_version']}", flush=True)

	out_path = f"/data/shards/{state_dict['old_model_version']}-{worker_i}.bin"
	ids = enc.encode_ordinary("Ask me a question.") + [EOP]
	ice_breaker_len = len(ids)
	max_tokens = 250
	SHARD = 100_000
	num_errors = 0
	n_chunks, written = 0, 0
	with OpenRouter(api_key=os.getenv("OPEN_ROUTER_API_KEY")) as client, torch.no_grad(), open(out_path, "ab") as f:
		while written < SHARD: 
			try:
				prompt = generate(mymodel, ids[-context_size:])
				if not prompt.strip():
					continue
				res = client.chat.send(
					model="anthropic/claude-sonnet-4.5",
					temperature=1.1,
					max_tokens=max_tokens,
					messages=[{
						"role": "user",
						"content": prompt,
					}],
				)
				answer = res.choices[0].message.content or ""

				print(prompt + answer + "\n")

				chunk = enc.encode_ordinary(prompt) + [EOP] + enc.encode_ordinary(answer) + [EOT]
				f.write(np.asarray(chunk, dtype=np.uint32).tobytes())
				f.flush()
				written += len(chunk)
				n_chunks += 1

				ids.extend(chunk)
				ids = ids[-context_size:]

				if n_chunks % 20 == 0:
					vol.commit()

			except Exception as e:
				print("Error data_worker: ", e)
				num_errors += 1
				if num_errors > 10:
					raise RuntimeError(f"too many API failures: {e}")

	vol.commit()
	print("fetched data")


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60)
def train(state_dict, total_steps=5000):
	import glob
	import numpy as np
	from collections import Counter
	from model import MainGPT

	mymodel = MainGPT(
		context_size=context_size, vocab_size=n_vocab,
		embedding_size=768, nheads=12, nblocks=12
	).to(device)

	model_version = state_dict["model_version"]
	old_model_version = state_dict["old_model_version"]


	out_path = f"/data/meta-{model_version}.pt"

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(mymodel.parameters(), lr=1e-3, fused=True) 

	start = 0
	loss = None
	best_val = float('inf')

	old = f"/data/meta-{old_model_version}.pt"
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

	full_batch = None

	def get_batch(refresh=False):
		if refresh:
			vol.reload()
			full_batch_size = 10*batchsize
			shards = sorted(glob.glob(f"/data/shards/{old_model_version}-*.bin"))
			randi = np.random.choice(len(shards), full_batch_size)
			counts = Counter(randi.tolist())

			chunks = []
			for shard_i, c in counts.items():
				contents = np.fromfile(shards[shard_i], dtype=np.uint32)
				starts = np.random.randint(0, len(contents) - context_size, c)
				chunks.extend([contents[j:j+context_size+1] for j in starts])

			nonlocal full_batch
			full_batch = torch.from_numpy(np.stack(chunks).astype(np.int64)).view(full_batch_size, context_size+1)
			full_batch = full_batch.to(device)
		ixs = torch.randperm(len(full_batch), device=device)[:batchsize]
		return full_batch[ixs, :-1], full_batch[ixs, 1:]

	def eval_loss():
		mymodel.eval()
		with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
			tokens, targets = get_batch(refresh=True)
			logits = mymodel(tokens)
			val_loss = criterion(logits.mT, targets).item()
		mymodel.train()
		return val_loss


	get_batch(refresh=True)

	for step in range(start, total_steps):

		tokens, targets = get_batch()

		with torch.autocast("cuda", dtype=torch.bfloat16):
			logits = mymodel(tokens)
			loss = criterion(logits.mT, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		lossi.append(loss.item())

		if step % 100 == 0:
			print(step, loss.item())
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


	with open(f"/data/losses-{model_version}.json", "w") as f:
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

	state_dict["old_model_version"] = state_dict["model_version"]
	state_dict["model_version"] += 1

	return loss.item(), val_loss



@app.local_entrypoint()
def main(workers: int = 2):

	# just get data
	with modal.Dict.ephemeral() as state:
		state["model_version"] = 0
		state["old_model_version"] = -1
		data_workers = [data_worker.spawn(i, state) for i in range(workers)]
		for c in data_workers:
			c.get()
	
	# just train
	# with modal.Dict.ephemeral() as state:
	# 	print("final loss:", train.remote(state))

	# indefinite loop
	# with modal.Dict.ephemeral() as state:
	# 	state["model_version"] = 0
	# 	state["old_model_version"] = -1
	# 	data_workers = [data_worker.spawn(i, state) for i in range(workers)]
	# 	print("final loss:", train.remote(state))


