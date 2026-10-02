import time, torch

N = sum(p.numel() for p in model.parameters())
flops_per_token = 6 * N
peak = 989e12          # per GPU, adjust for your hardware
world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
tokens_per_step = global_batch_size * seq_len

for step, batch in enumerate(loader):
    if step == 10:                       # skip warmup steps
        torch.cuda.synchronize(); t0 = time.perf_counter(); s0 = step

    loss = train_step(batch)             # fwd + bwd + optimizer.step()

    if step > 10 and step % 20 == 0:
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / (step - s0)
        tps = tokens_per_step / dt
        mfu = tps * flops_per_token / (world * peak)
        print(f"step {step} | {dt*1000:.0f} ms/step | {tps:,.0f} tok/s "
              f"| {tps/world:,.0f} tok/s/gpu | MFU {mfu:.1%}")