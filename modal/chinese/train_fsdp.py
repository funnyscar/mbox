# model doesn't fit on one gpu anymore
# ~ 1B params and up

import modal

app = modal.App("fsdp")
image = modal.Image.debian_slim().pip_install("torch")
vol = modal.Volume.from_name("ckpts", create_if_missing=True)
N_GPUS = 8

def worker():
    import os, torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    # 1. Init
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    # 2. Model with repeated blocks (the unit FSDP shards)
    class Model(nn.Module):
        def __init__(self, d=512, n_layers=8):
            super().__init__()
            self.blocks = nn.ModuleList(
                nn.TransformerEncoderLayer(d, 8, 4 * d, batch_first=True) for _ in range(n_layers)
            )
            self.head = nn.Linear(d, 1)
        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return self.head(x)

    model = Model().cuda()

    # 3. Shard each block, then the root. bf16 compute, fp32 grad reduce
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mp_policy=mp)
    fully_shard(model, mp_policy=mp)

    # 4. Optimizer AFTER sharding (it sees the sharded params)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ds = TensorDataset(torch.randn(4096, 128, 512), torch.randn(4096, 128, 1))
    sampler = DistributedSampler(ds)
    loader = DataLoader(ds, batch_size=8, sampler=sampler)

    # 5. Train loop (same as DDP)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for x, y in loader:
            loss = nn.functional.mse_loss(model(x.cuda()).float(), y.cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()

        # 6. Checkpoint: gather full weights to CPU, save on rank 0.
        #    Every rank must call get_model_state_dict (it's a collective).
        sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            print(f"epoch {epoch} loss {loss.item():.4f}")
            torch.save(sd, "/ckpt/model.pt")

    dist.destroy_process_group()

@app.function(gpu=f"H100:{N_GPUS}", image=image, volumes={"/ckpt": vol}, timeout=86400)
def train():
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch
    config = LaunchConfig(
        min_nodes=1, max_nodes=1, nproc_per_node=N_GPUS,
        rdzv_backend="c10d", rdzv_endpoint="localhost:29500",
        run_id="fsdp", max_restarts=0,
    )
    elastic_launch(config, worker)()
    vol.commit()