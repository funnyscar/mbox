# when 8 gpus isn't enough
# slightly slower

import modal.experimental

N_NODES = 2

@app.function(gpu="H100:8", image=image, timeout=60*60*24)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train_multi():
    from torch.distributed.run import parse_args, run
    info = modal.experimental.get_cluster_info()
    run(parse_args([
        f"--nnodes={N_NODES}",
        f"--node_rank={info.rank}",
        f"--master_addr={info.container_ips[0]}",
        "--master_port=1234",
        "--nproc_per_node=8",
        "/root/train.py",
    ]))