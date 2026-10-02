import os
import argparse
import tempfile
import torch
import torch.nn as nn

NGPUS = int(os.environ.get("NGPUS", 4))
context_size = 512
batchsize = 32  # per GPU; global batch = batchsize * NGPUS


def get_token_stream(ds, enc, EOT):
	for row in ds:
		yield from enc.encode(row["text"], add_special_tokens=False) + [EOT]


def get_batch_stream(ds, enc, EOT, device):
	toks = get_token_stream(ds, enc, EOT)
	n = batchsize * (context_size + 1)
	buf = []
	while True:
		while len(buf) < n:
			buf.append(next(toks))
		chunk, buf = buf[:n], buf[n:]
		batch = torch.tensor(chunk, dtype=torch.long).view(batchsize, context_size + 1).to(device)
		yield batch[:, :-1], batch[:, 1:]


def train_loop(config):
	import ray.train
	from ray.train import Checkpoint
	from ray.train.torch import prepare_model, get_device
	from transformers import AutoTokenizer
	from datasets import load_dataset
	from datasets.distributed import split_dataset_by_node
	from model import ChineseGPT

	ctx = ray.train.get_context()
	rank, world = ctx.get_world_rank(), ctx.get_world_size()
	device = get_device()

	enc = AutoTokenizer.from_pretrained("01-ai/Yi-6B")
	enc.model_max_length = int(1e9)
	EOT = enc.eos_token_id

	base = load_dataset("CASIA-LM/ChineseWebText2.0", split="test", streaming=True)
	val_ds = base.take(1000)  # identical on every rank
	train_ds = base.skip(1000)

	raw = ChineseGPT(context_size=context_size, vocab_size=len(enc), **config["model"])

	decay = [p for p in raw.parameters() if p.dim() >= 2]
	no_decay = [p for p in raw.parameters() if p.dim() < 2]
	optimizer = torch.optim.AdamW(
		[{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
		lr=config["lr"], fused=True,
	)
	sched = torch.optim.lr_scheduler.OneCycleLR(
		optimizer, max_lr=config["lr"], total_steps=config["total_steps"], pct_start=0.05
	)

	# resume
	start, best_val = 0, float("inf")
	ckpt = ray.train.get_checkpoint()
	if ckpt:
		with ckpt.as_directory() as d:
			ck = torch.load(os.path.join(d, "ckpt.pt"), map_location="cpu")
		raw.load_state_dict(ck["model"])
		optimizer.load_state_dict(ck["optim"])
		sched.load_state_dict(ck["sched"])
		start, best_val = ck["step"] + 1, ck["best_val"]
		train_ds = train_ds.shuffle(seed=start, buffer_size=10_000)

	# each rank gets a disjoint slice of the stream
	train_ds = split_dataset_by_node(train_ds, rank=rank, world_size=world)

	model = prepare_model(raw)  # moves to device + wraps in DDP
	torch.set_float32_matmul_precision("high")
	model = torch.compile(model)

	criterion = nn.CrossEntropyLoss()
	batches = get_batch_stream(train_ds, enc, EOT, device)
	valstream = get_batch_stream(val_ds, enc, EOT, device)
	val_batches = [next(valstream) for _ in range(20)]

	def eval_loss():
		model.eval()
		tot = 0.0
		with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
			for x, y in val_batches:
				tot += criterion(model(x).mT, y).item()
		model.train()
		return tot / len(val_batches)

	for step in range(start, config["total_steps"]):
		x, y = next(batches)
		with torch.autocast("cuda", dtype=torch.bfloat16):
			loss = criterion(model(x).mT, y)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		sched.step()

		if step % config["eval_every"] == 0 or step == config["total_steps"] - 1:
			val_loss = eval_loss()
			metrics = {
				"step": step,
				"train_loss": loss.item(),
				"val_loss": val_loss,
				"lr": sched.get_last_lr()[0],
				"tokens": (step + 1) * batchsize * context_size * world,
			}
			best_val = min(best_val, val_loss)

			# all ranks must call report; only rank 0 attaches a checkpoint
			with tempfile.TemporaryDirectory() as d:
				checkpoint = None
				if rank == 0:
					torch.save({
						"model": raw.state_dict(),
						"optim": optimizer.state_dict(),
						"sched": sched.state_dict(),
						"step": step,
						"best_val": best_val,
					}, os.path.join(d, "ckpt.pt"))
					checkpoint = Checkpoint.from_directory(d)
				ray.train.report(metrics, checkpoint=checkpoint)



CONFIGS = {
	"a": dict(embedding_size=128,  nheads=2,  nblocks=4),
	"b": dict(embedding_size=192,  nheads=3,  nblocks=4),
	"c": dict(embedding_size=192,  nheads=3,  nblocks=6),
	"d": dict(embedding_size=256,  nheads=4,  nblocks=6),
	"e": dict(embedding_size=256,  nheads=4,  nblocks=8),
	"f": dict(embedding_size=320,  nheads=5,  nblocks=8),
	"g": dict(embedding_size=384,  nheads=6,  nblocks=8),
	"h": dict(embedding_size=448,  nheads=7,  nblocks=9),
	"i": dict(embedding_size=512,  nheads=8,  nblocks=10),
	"j": dict(embedding_size=576,  nheads=9,  nblocks=11),
	"k": dict(embedding_size=640,  nheads=10, nblocks=12),
	"l": dict(embedding_size=768,  nheads=12, nblocks=12),
	"m": dict(embedding_size=832,  nheads=13, nblocks=13),
	"n": dict(embedding_size=896,  nheads=14, nblocks=14),
	"o": dict(embedding_size=960,  nheads=15, nblocks=15),
	"p": dict(embedding_size=1024, nheads=16, nblocks=16),
	"q": dict(embedding_size=1024, nheads=16, nblocks=18),
	"r": dict(embedding_size=1024, nheads=16, nblocks=20),
	"s": dict(embedding_size=1280, nheads=20, nblocks=20),
}


if __name__ == "__main__":
	from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig
	from ray.train.torch import TorchTrainer

	ap = argparse.ArgumentParser()
	ap.add_argument("--model", default="a")
	ap.add_argument("--steps", type=int, default=8000)
	ap.add_argument("--lr", type=float, default=1e-3)
	args = ap.parse_args()

	# cloud bucket on Anyscale (persists across job retries); NFS fallback
	storage = os.environ.get("ANYSCALE_ARTIFACT_STORAGE", "/mnt/cluster_storage")

	trainer = TorchTrainer(
		train_loop,
		train_loop_config={
			"model": CONFIGS[args.model],
			"total_steps": args.steps,
			"lr": args.lr,
			"eval_every": max(args.steps // 50, 1),
		},
		scaling_config=ScalingConfig(num_workers=NGPUS, use_gpu=True),
		run_config=RunConfig(
			name=f"chinese-{args.model}",
			storage_path=storage,  # same name + path => resumes from latest checkpoint
			checkpoint_config=CheckpointConfig(
				num_to_keep=2,
				checkpoint_score_attribute="val_loss",
				checkpoint_score_order="min",
			),
			failure_config=FailureConfig(max_failures=3),  # restart on spot preemption
		),
	)
	result = trainer.fit()
	print(result.metrics)