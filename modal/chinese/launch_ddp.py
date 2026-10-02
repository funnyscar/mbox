# 8 GPUs, 8 copies of the model, averaged every opt.step()
# run with 8x the data per second.


import modal

app = modal.App("chinese-ddp")
image = (modal.Image.debian_slim()
         .pip_install("torch")
         .add_local_file("train.py", "/root/train.py"))

@app.function(gpu="H100:4", image=image, timeout=60*60*24)
def train():
    from torch.distributed.run import parse_args, run
    run(parse_args([
        "--nproc_per_node=4",
        "/root/train.py",
    ]))