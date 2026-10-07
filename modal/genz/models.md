- genz-0.0.pt - Overfit model (context_size=256, vocab_size=50304, embedding_size=756, nheads=12, nblocks=8)
	- model, optimizer, step, save
- genz-0.1.pt (changed on genz_00.py)
	- fine (context_size=128, vocab_size=50304, embedding_size=128, nheads=4, nblocks=4), 
	- sample starting at [eot]

- genz-1.0.pt
	- training on reddit, tiktok commends
	- data sharding
	- load shard 2 gpu
	- per-batch mixing
- genz-1.0.pt
	- DDP for speedup training time

- genz-1.5.pt
	- fine-tuning on the 1.0 model


- genz-1.6.pt
	- fine tuned qna

- genz-1.7.pt
	- fine tuned translation

- genz-2.0 pt
	- even larger model

### Serving

Deploy the authenticated web endpoint from this directory:

```sh
modal deploy serve.py
```

Send a generation request to the deployed endpoint. Set `API_TOKEN` to the
value configured in the `models-gateway` Modal secret:

```sh
curl -X POST "$ENDPOINT_URL" \
  -H "Authorization: $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"whats good","model_version":"gen_10","max_tokens":100}'
```

Set `model_version` to `gen_00` to use `/data/genz-0.1.pt` or `gen_10` to use
`/data/genz-l.pt`. The service snapshots CPU-loaded model weights, then moves
both models to the A10G and warms up inference after restore. It keeps the
models resident so requests do not pay model-switching or checkpoint-loading
costs. `scaledown_window=300` keeps an idle container available for five
minutes; longer warm retention can reduce cold starts further at additional
GPU cost.

Run inference without deploying the endpoint:

```sh
modal run serve.py --model-version gen_00 --prompt "whats good"
```
