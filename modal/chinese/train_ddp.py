# 8 GPUs, 8 copies of the model, averaged every opt.step()
# run with 8x the data per second.
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def main():
    # 1. Init process group (torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR)
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    # 2. Data: DistributedSampler gives each rank a different shard
    ds = TensorDataset(torch.randn(10_000, 32), torch.randn(10_000, 1))
    sampler = DistributedSampler(ds, shuffle=True)
    loader = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=2, pin_memory=True)

    # 3. Model: move to GPU, then wrap in DDP
    model = torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).cuda()
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # 4. Train loop
    for epoch in range(10):
        sampler.set_epoch(epoch)  # reshuffle differently each epoch
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()   # DDP all-reduces gradients here
            opt.step()

        # 5. Log/checkpoint on rank 0 only
        if rank == 0:
            print(f"epoch {epoch} loss {loss.item():.4f}")
            torch.save(model.module.state_dict(), "/ckpt/model.pt")  # .module = unwrapped model

    dist.destroy_process_group()

if __name__ == "__main__":
    main()