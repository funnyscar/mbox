# 8 GPUs, 8 copies of the model, averaged every opt.step()
# run with 8x the data per second.


# DDP version: one process per GPU inside a single Modal container
import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import modal

app = modal.App("chinesegpt")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "transformers", "sentencepiece", "datasets")
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("chinese", create_if_missing=True)

N_GPUS = 4
context_size = 512
batchsize = 32  # per GPU; effective batch = batchsize * world_size


def get_token_stream(ds, enc, EOT):
	for row in ds:
		yield from enc.encode(row["text"], add_special_tokens=False) + [EOT]


def get_batch_stream(ds, enc, EOT, device):
	token_generator = get_token_stream(ds, enc, EOT)
	n = batchsize * (context_size + 1)
	buf = []
	while True:
		while len(buf) < n:
			buf.append(next(token_generator))
		chunk, buf = buf[:n], buf[n:]
		batch = torch.tensor(chunk, dtype=torch.long).view(batchsize, context_size + 1)
		batch = batch.to(device, non_blocking=True)
		yield batch[:, :-1], batch[:, 1:]


def worker(rank, world_size, total_steps, model_version):
	from model import ChineseGPT
	from transformers import AutoTokenizer
	from datasets import load_dataset
	from datasets.distributed import split_dataset_by_node
	from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

	# --- process group setup ---
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "12355"
	dist.init_process_group("nccl", rank=rank, world_size=world_size)
	torch.cuda.set_device(rank)
	device = f"cuda:{rank}"
	master = rank == 0

	out_path = f"/data/chinese-{model_version}.pt"
	loss_path = f"/data/losses-{model_version}.jsonl"

	enc = AutoTokenizer.from_pretrained("01-ai/Yi-6B")
	EOT = enc.eos_token_id

	ds = load_dataset("Skywork/SkyPile-150B", split="train", streaming=True)
	val_ds = ds.take(10_000)
	train_ds = ds.skip(10_000)

	n_vocab = (-len(enc) % 64) + len(enc)

	mymodel = ChineseGPT(
		context_size=context_size,
		vocab_size=n_vocab,
		# embedding_size=128, nheads=2,  nblocks=4	  # 0.8M a
		# embedding_size=192, nheads=3,  nblocks=4	  # 1.8M b
		# embedding_size=192, nheads=3,  nblocks=6    # 2.7M c
		# embedding_size=256, nheads=4,  nblocks=6    # 4.7M d
		# embedding_size=256, nheads=4,  nblocks=8    # 6.3M e
		# embedding_size=320, nheads=5,  nblocks=8    # 9.8M f
		# embedding_size=384, nheads=6,  nblocks=8    # 14M g
		# embedding_size=448, nheads=7,  nblocks=9    # 22M h
		# embedding_size=512, nheads=8,  nblocks=10   # 31M i
		# embedding_size=576, nheads=9,  nblocks=11   # 44M j
		# embedding_size=640, nheads=10, nblocks=12   # 59M k
		# embedding_size=768, nheads=12, nblocks=12   # 85M l
		embedding_size=832, nheads=13, nblocks=13   # 108M m
		# embedding_size=896, nheads=14, nblocks=14   # 135M n
		# embedding_size=960,  nheads=15, nblocks=15   # 166M o
		# embedding_size=1024, nheads=16, nblocks=16   # 201M p
		# embedding_size=1024, nheads=16, nblocks=18   # ~226M q
		# embedding_size=1024, nheads=16, nblocks=20   # ~250 r
		# embedding_size=1280, nheads=20, nblocks=20   # ~393M s
	).to(device)
    
    # ---- unwrapped model for saving and stats
	raw_model = mymodel

	base_lr, base_batch = 3e-4, 64 # on one gpu
	global_batch = batchsize * world_size
	lr = base_lr * (global_batch / base_batch) ** 0.5
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(mymodel.parameters(), lr=1e-3, fused=True)
	warmup = 500
	scheduler = SequentialLR(
		optimizer,
		schedulers=[
			LinearLR(optimizer, start_factor=1e-3, total_iters=warmup),
			CosineAnnealingLR(optimizer, T_max=total_steps - warmup, eta_min=3e-5),
		],
		milestones=[warmup],
	)

	start = 0
	loss = None
	best_val = float("inf")

	if os.path.exists(out_path):
		ck = torch.load(out_path, map_location=device)
		raw_model.load_state_dict(ck["model"])
		optimizer.load_state_dict(ck["optim"])
		scheduler.load_state_dict(ck["sched"])
		start, best_val = ck["step"] + 1, ck["best_val"]
		train_ds = train_ds.shuffle(seed=start, buffer_size=10_000)
		if master:
			print(f"Loaded {out_path}", flush=True)

	# --- each rank gets its own slice of the data ---
	train_ds = split_dataset_by_node(train_ds, rank=rank, world_size=world_size)
	val_ds = split_dataset_by_node(val_ds, rank=rank, world_size=world_size)

	torch.set_float32_matmul_precision("high")
	mymodel = torch.compile(mymodel)
	mymodel = DDP(mymodel, device_ids=[rank]) # distribute model to device ids
    # ----

	batchstream = get_batch_stream(train_ds, enc, EOT, device)
	valstream = get_batch_stream(val_ds, enc, EOT, device)
	val_batches = [next(valstream) for _ in range(10)]

	lossi = []
	vlossi = []

	def eval_loss():
		mymodel.eval()
		with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
			total = torch.zeros((), device=device)
			for x, y in val_batches:
				total += criterion(mymodel(x).mT, y)
			total /= len(val_batches)
		dist.all_reduce(total, op=dist.ReduceOp.AVG)  # same value on every rank
		mymodel.train()
		return total.item()

	for step in range(start, total_steps):
		tokens, targets = next(batchstream)

		with torch.autocast("cuda", dtype=torch.bfloat16):
			logits = mymodel(tokens)
			loss = criterion(logits.mT, targets)

		optimizer.zero_grad()
		loss.backward()  # gradients are all-reduced across GPUs here
		torch.nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
		optimizer.step()
		scheduler.step()

		if step % 100 == 0:
			train_loss = loss.detach().clone()
			dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
			val_loss = eval_loss()

			if master:
				lossi.append(train_loss.item())
				vlossi.append(val_loss)
				print(step, train_loss.item(), val_loss, flush=True)

				if val_loss < best_val:
					torch.save({
						"model": raw_model.state_dict(),
						"optim": optimizer.state_dict(),
						"sched": scheduler.state_dict(),
						"step": step,
						"best_val": val_loss,
					}, out_path)
					vol.commit()
					print("best val: ", val_loss, flush=True)

			# val_loss is identical on all ranks, so best_val stays in sync
			best_val = min(best_val, val_loss)
			dist.barrier()

	val_loss = eval_loss()
	final_train = loss.detach().clone()

    # --- avgs
	dist.all_reduce(final_train, op=dist.ReduceOp.AVG)

	if master:
		n_emb = sum(p.numel() for m in raw_model.modules() if isinstance(m, nn.Embedding) for p in m.parameters())
		nparams = sum(p.numel() for p in raw_model.parameters()) - n_emb
		with open(loss_path, "a") as f:
			f.write(json.dumps({
				"train": lossi,
				"val": vlossi,
				"final_train": final_train.item(),
				"final_val": val_loss,
				"out_path": out_path,
				"nparams": nparams,
				"vocab_size": n_vocab,
				"context_size": context_size,
				"embedding_size": raw_model.C,
				"nheads": raw_model.nheads,
				"nblocks": raw_model.nblocks,
				"world_size": world_size,
				"global_batch": batchsize * world_size,
			}) + "\n")
		vol.commit()
		print("val loss: ", val_loss)
		print("train loss: ", final_train.item())

	dist.barrier()
	dist.destroy_process_group()


@app.function(image=image, gpu=f"A100:{N_GPUS}", volumes={"/data": vol}, timeout=60 * 60 * 20, secrets=[modal.Secret.from_name("huggingface-secret")])
def train(total_steps=20000, model_version="m.01"):
	import torch.multiprocessing as mp

	world_size = torch.cuda.device_count()
	mp.spawn(worker, args=(world_size, total_steps, model_version), nprocs=world_size, join=True)

	vol.reload()
	with open(f"/data/losses-{model_version}.json") as f:
		res = json.load(f)
	return res["final_train"], res["final_val"]


@app.local_entrypoint()
def main():
	print("final loss:", train.remote())