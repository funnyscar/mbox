# when 8 gpus isn't enough
# slightly slower

# FSDP2 on a Modal multi-node cluster.
# N_NODES containers x GPUS_PER_NODE GPUs each; params/grads/optim state sharded across ALL GPUs.
# Each container spawns one process per local GPU; global rank = node_rank * GPUS_PER_NODE + local_rank.
# - fsdp
# - rank math for which gpu
# - no localhost but MASTER_ADDR to find the other's gpus (TCP verrry slow)
# - NVlink
# 


import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
	get_model_state_dict, get_optimizer_state_dict,
	set_model_state_dict, set_optimizer_state_dict,
	StateDictOptions,
)
import modal
import modal.experimental

app = modal.App("chinesegpt")

image = (
	modal.Image
		.debian_slim()
		.pip_install("torch>=2.6", "transformers", "sentencepiece", "datasets")
		.add_local_python_source("model")
)

vol = modal.Volume.from_name("chinese", create_if_missing=True)

N_NODES = 2
GPUS_PER_NODE = 8          # clusters must request every GPU on the host
GPU_TYPE = "H100"
MASTER_PORT = 29500
context_size = 512
batchsize = 32  # per GPU; effective batch = batchsize * world_size

FULL_SD = StateDictOptions(full_state_dict=True, cpu_offload=True)


def get_cluster():
	"""Returns (node_rank, container_ips). Supports new and older Modal APIs."""
	if hasattr(modal, "Cluster"):
		c = modal.Cluster.from_context()
		return c.container_rank(), list(c.container_ips())
	import modal.experimental
	info = modal.experimental.get_cluster_info()
	return info.rank, list(info.container_ips)


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


def worker(local_rank, node_rank, nproc, world_size, master_addr, total_steps, model_version):
	from model import ChineseGPT
	from transformers import AutoTokenizer
	from datasets import load_dataset
	from datasets.distributed import split_dataset_by_node

	# --- process group setup (across all nodes) ---
	rank = node_rank * nproc + local_rank
	os.environ["MASTER_ADDR"] = master_addr
	os.environ["MASTER_PORT"] = str(MASTER_PORT)
	torch.cuda.set_device(local_rank)
	dist.init_process_group("nccl", rank=rank, world_size=world_size)
	device = f"cuda:{local_rank}"
	master = rank == 0

	out_path = f"/data/chinese-{model_version}.pt"
	loss_path = f"/data/losses-{model_version}.json"

	enc = AutoTokenizer.from_pretrained("01-ai/Yi-6B")
	EOT = enc.eos_token_id

	ds = load_dataset("Skywork/SkyPile-150B", split="train", streaming=True)
	val_ds = ds.take(10_000)
	train_ds = ds.skip(10_000)

	n_vocab = (-len(enc) % 64) + len(enc)

	model = ChineseGPT(
		context_size=context_size,
		vocab_size=n_vocab,
		embedding_size=768, nheads=12, nblocks=12   # 85M l
	).to(device)

	# --- FSDP2: shard each transformer block, then the root ---
	mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
	# Adjust the match below if your block class isn't named "*Block".
	for m in model.modules():
		if m is not model and type(m).__name__.endswith("Block"):
			fully_shard(m, mp_policy=mp_policy)
	fully_shard(model, mp_policy=mp_policy)
	raw_model = model

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

	start = 0
	loss = None
	best_val = float("inf")

	if os.path.exists(out_path):
		ck = torch.load(out_path, map_location="cpu")
		set_model_state_dict(model, ck["model"], options=StateDictOptions(full_state_dict=True))
		set_optimizer_state_dict(model, optimizer, ck["optim"], options=StateDictOptions(full_state_dict=True))
		start, best_val = ck["step"] + 1, ck["best_val"]
		train_ds = train_ds.shuffle(seed=start, buffer_size=10_000)
		del ck
		if master:
			print(f"Loaded {out_path}", flush=True)

	def save(step, val_loss):
		# collective across ALL nodes; full state gathered to global rank 0's CPU
		msd = get_model_state_dict(model, options=FULL_SD)
		osd = get_optimizer_state_dict(model, optimizer, options=FULL_SD)
		if master:
			torch.save({"model": msd, "optim": osd, "step": step, "best_val": val_loss}, out_path)
			vol.commit()
			print("best val: ", val_loss, flush=True)

	# --- each global rank gets its own slice of the data ---
	train_ds = split_dataset_by_node(train_ds, rank=rank, world_size=world_size)
	val_ds = split_dataset_by_node(val_ds, rank=rank, world_size=world_size)

	torch.set_float32_matmul_precision("high")
	cmodel = torch.compile(model)

	batchstream = get_batch_stream(train_ds, enc, EOT, device)
	valstream = get_batch_stream(val_ds, enc, EOT, device)
	val_batches = [next(valstream) for _ in range(10)]

	lossi = []
	vlossi = []

	def eval_loss():
		cmodel.eval()
		with torch.no_grad():
			total = torch.zeros((), device=device)
			for x, y in val_batches:
				total += criterion(cmodel(x).float().mT, y)
			total /= len(val_batches)
		dist.all_reduce(total, op=dist.ReduceOp.AVG)
		cmodel.train()
		return total.item()

	for step in range(start, total_steps):
		tokens, targets = next(batchstream)

		logits = cmodel(tokens)
		loss = criterion(logits.float().mT, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % 100 == 0:
			train_loss = loss.detach().clone()
			dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
			val_loss = eval_loss()

			if master:
				lossi.append(train_loss.item())
				vlossi.append(val_loss)
				print(step, train_loss.item(), val_loss, flush=True)

			if val_loss < best_val:
				save(step, val_loss)
			best_val = min(best_val, val_loss)
			dist.barrier()

	val_loss = eval_loss()
	final_train = loss.detach().clone()
	dist.all_reduce(final_train, op=dist.ReduceOp.AVG)

	if master:
		n_emb = sum(p.numel() for m in raw_model.modules() if isinstance(m, nn.Embedding) for p in m.parameters())
		nparams = sum(p.numel() for p in raw_model.parameters()) - n_emb
		with open(loss_path, "w") as f:
			json.dump({
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
				"n_nodes": world_size // nproc,
				"global_batch": batchsize * world_size,
			}, f)
		vol.commit()
		print("val loss: ", val_loss)
		print("train loss: ", final_train.item())

	dist.barrier()
	dist.destroy_process_group()


@app.function(
	image=image,
	gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
	volumes={"/data": vol},
	timeout=60 * 60 * 4,
	secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.experimental.clustered(size=N_NODES, rdma=True)  # newer SDKs: @modal.clustered(...)
def train(total_steps=20000, model_version="l"):
	import torch.multiprocessing as mp

	node_rank, ips = get_cluster()
	nproc = torch.cuda.device_count()
	world_size = nproc * len(ips)
	master_addr = ips[0]

	mp.spawn(
		worker,
		args=(node_rank, nproc, world_size, master_addr, total_steps, model_version),
		nprocs=nproc,
		join=True,
	)

	# only rank 0's container output is returned by Modal
	if node_rank != 0:
		return None
	vol.reload()
	with open(f"/data/losses-{model_version}.json") as f:
		res = json.load(f)
	return res["final_train"], res["final_val"]


@app.local_entrypoint()
def main():
	print("final loss:", train.remote())