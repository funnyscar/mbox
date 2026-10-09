import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import modal

app = modal.App("fine")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch", "tiktoken", "datasets")
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("fine", create_if_missing=True)

N_GPUS = 4
context_size = 512
batchsize = 32  # per GPU; effective batch = batchsize * world_size


def get_token_stream(ds, enc, EOT):
	for row in ds:
		yield from enc.encode_ordinary(row["text"]) + [EOT]


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


def worker(rank, world_size, total_steps, model_version, smoke_test=True):
	import tiktoken, time
	from model import MegaGPT
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

	out_path = f"/data/fine-{model_version}.pt"
	best_path = f"/data/fine-{model_version}-best.pt"
	loss_path = f"/data/losses-{model_version}.jsonl"

	enc = tiktoken.get_encoding("gpt2")
	EOT = enc.eot_token
	n_vocab = 50304


	ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
	val_ds = ds.take(10_000)
	train_ds = ds.skip(10_000)
	train_ds = train_ds.shuffle(seed=1914, buffer_size=10_000)

	mymodel = MegaGPT(
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
		# embedding_size=832, nheads=13, nblocks=13   # 108M m
		# embedding_size=896, nheads=14, nblocks=14   # 135M n
		# embedding_size=960,  nheads=15, nblocks=15   # 166M o
		# embedding_size=1024, nheads=16, nblocks=16   # 201M p
		# embedding_size=1024, nheads=16, nblocks=18   # ~226M q
		embedding_size=1024, nheads=16, nblocks=20   # ~250 r
		# embedding_size=1280, nheads=20, nblocks=20   # ~393M s
	).to(device)
    
    # ---- unwrapped model for saving and stats
	raw_model = mymodel
	n_emb = sum(p.numel() for m in raw_model.modules() if isinstance(m, nn.Embedding) for p in m.parameters())
	nparams = sum(p.numel() for p in raw_model.parameters()) - n_emb

	criterion = nn.CrossEntropyLoss()

	base_lr, base_batch = 3e-4, 64 # on one gpu
	global_batch = batchsize * world_size
	lr = base_lr * (global_batch / base_batch) ** 0.5
	decay    = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2] # matrices
	no_decay = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2] # biases, ln
	optimizer = torch.optim.AdamW([
		{"params": decay, "weight_decay": 0.1},
    	{"params": no_decay, "weight_decay": 0.0}
	], lr=lr, betas=(0.9, 0.95), fused=True)

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

	assert start < total_steps, "start >= total_steps"


	for step in range(start, total_steps):
		tokens, targets = next(batchstream)

		with torch.autocast("cuda", dtype=torch.bfloat16):
			logits = mymodel(tokens)
			loss = criterion(logits.reshape(-1, n_vocab), targets.reshape(-1))

		optimizer.zero_grad()
		loss.backward()  # gradients are all-reduced across GPUs here
		torch.nn.utils.clip_grad_norm_(mymodel.parameters(), 1.0)
		optimizer.step()
		scheduler.step()

		if smoke_test and step == start + 20:
			torch.cuda.synchronize(); t0 = time.time()
		if smoke_test and step == start + 120:
			torch.cuda.synchronize()
			tok_s = 100 * batchsize * context_size * world_size / (time.time() - t0)
			flops_tok = 6 * nparams + 6 * n_vocab * raw_model.C + 12 * raw_model.nblocks * context_size * raw_model.C
			if master:
				print("token/s: ", tok_s)
				PEAK_FLOPS = 312e12 # A100
				print("mfu: ", tok_s * flops_tok / (PEAK_FLOPS * world_size))
				print("elapsed 100 steps: ", time.time() - t0)
				print("mem: ", torch.cuda.max_memory_allocated() / 1e9, "GB")
				print("5GB tok:", 5e9 / tok_s / 3600, "hrs")
			dist.barrier()
			dist.destroy_process_group()
			return

		if step % 100 == 0 and not smoke_test:
			train_loss = loss.detach().clone()
			dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
			val_loss = eval_loss()

			if master:
				lossi.append(train_loss.item())
				vlossi.append(val_loss)
				print(step, train_loss.item(), val_loss, flush=True)

				ck = {
					"model": raw_model.state_dict(),
					"optim": optimizer.state_dict(),
					"sched": scheduler.state_dict(),
					"step": step,
					"best_val": min(best_val, val_loss),
				}

				if step > 0 and step % 1000 == 0:
					torch.save(ck, out_path)
					vol.commit()

				if step > 1000 and val_loss < best_val:
					torch.save(ck, best_path)
					print("best val: ", val_loss, flush=True)
					vol.commit()

			# val_loss is identical on all ranks, so best_val stays in sync
			best_val = min(best_val, val_loss)
			dist.barrier()

	val_loss = eval_loss()
	final_train = loss.detach().clone()

    # --- avgs
	dist.all_reduce(final_train, op=dist.ReduceOp.AVG)

	if master:
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
def train(total_steps=76000, model_version="r", smoke_test=True):
	import torch.multiprocessing as mp

	world_size = torch.cuda.device_count()
	mp.spawn(worker, args=(world_size, total_steps, model_version, smoke_test), nprocs=world_size, join=True)
	if smoke_test:
		return None

	vol.reload()
	with open(f"/data/losses-{model_version}.jsonl") as f:
		res = json.loads(f.readlines()[-1])
	return res["final_train"], res["final_val"]


@app.local_entrypoint()
def main(smoke_test: bool = False, steps: int = 76000, version: str = "r"):
	print("final loss:", train.remote(total_steps=steps, model_version=version, smoke_test=smoke_test))