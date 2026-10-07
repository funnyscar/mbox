
# shared
context_size=512, vocab_size=enc.n_vocab

- a: embedding_size=128, nheads=2,  nblocks=4    # 0.8M non-emb
- b: embedding_size=192, nheads=3,  nblocks=4    # 1.8M
- c: embedding_size=192, nheads=3,  nblocks=6    # 2.7M
- d: embedding_size=256, nheads=4,  nblocks=6    # 4.7M
- e: embedding_size=256, nheads=4,  nblocks=8    # 6.3M
- f: embedding_size=320, nheads=5,  nblocks=8    # 9.8M
- g: embedding_size=384, nheads=6,  nblocks=8    # 14M
- h: embedding_size=448, nheads=7,  nblocks=9    # 22M
- i: embedding_size=512, nheads=8,  nblocks=10   # 31M
- j: embedding_size=576, nheads=9,  nblocks=11   # 44M
- k: embedding_size=640, nheads=10, nblocks=12   # 59M
- l: embedding_size=768, nheads=12, nblocks=12   # 85M
- m: embedding_size=832, nheads=13, nblocks=13   # 108M
- n: embedding_size=896, nheads=14, nblocks=14   # 135M
- o: embedding_size=960,  nheads=15, nblocks=15   # 166M non-emb
- p: embedding_size=1024, nheads=16, nblocks=16   # 201M (~24 hrs A100)
- q: embedding_size=1024, nheads=16, nblocks=18   # ~226M
- r: embedding_size=1024, nheads=16, nblocks=20   # ~250
- s: embedding_size=1280, nheads=20, nblocks=20   # ~393M (~24 hrs H100)

### Serving

The serving API reads `chinese-{letter}.pt` from the `chinese` Modal volume.
Deploy it from this directory:

```sh
modal deploy serve.py
```

Send a request to the deployed endpoint with a version from `a` through `s`:

```sh
curl -X POST "$ENDPOINT_URL" \
  -H "Authorization: $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好","model_version":"l","max_tokens":100}'
```

For inference from the terminal without deploying the endpoint, run:

```sh
modal run serve.py --model-version l --prompt "你好"
```



everything else held same. even order of data

graphs: 
- loss v. steps for various non-embedding params.
- compute time


### studying throughput

run script for throughput

| GPU | $/h | Peak BF16 TFLOPS (dense) |
|---|---|---|
| Nvidia B300 | $7.10 | ~2,250 |
| Nvidia B200 | $6.25 | ~2,250 |
| Nvidia H200 SXM | $4.54 | 989 |
| Nvidia H100 SXM5 | $3.95 | 989 |
| Nvidia RTX PRO 6000 | $3.03 | ~504 |
| Nvidia A100, 80 GB | $2.50 | 312 |
| Nvidia A100, 40 GB | $2.10 | 312 |
| Nvidia L40S | $1.95 | 362 |
| Nvidia A10 | $1.10 | 125 |
| Nvidia L4 | $0.80 | 121 |
| Nvidia T4 | $0.59 | 65 (FP16, no BF16) |


chinchilla rule:
```
N_nonemb ≈ 12 · L · d²        (per block: 4d² attention + 8d² MLP)
tokens   = 20 · N_nonemb
```

```
FLOPs ≈ 6 · (N_nonemb + V·d) · tokens      (6 = 2 fwd + 4 bwd per param per token; V·d = output layer)
hours = FLOPs / (effective FLOP/s · 3600)
```


### studying single GPU max

| GPU | Memory | Max params (states only) | Practical max |
|---|---|---|---|
| Nvidia B300 | 288 GB | ~18B | ~11–12B |
| Nvidia B200 | 192 GB | ~12B | ~7–8B |
| Nvidia H200 SXM | 141 GB | ~8.8B | ~5–6B |
| Nvidia H100 SXM5 | 80 GB | ~5.0B | ~3B |
| Nvidia RTX PRO 6000 | 96 GB | ~6.0B | ~4B |
| Nvidia A100, 80 GB | 80 GB | ~5.0B | ~3B |
| Nvidia A100, 40 GB | 40 GB | ~2.5B | ~1.5B |
| Nvidia L40S | 48 GB | ~3.0B | ~2B |
| Nvidia A10 | 24 GB | ~1.5B | ~1B |
| Nvidia L4 | 24 GB | ~1.5B | ~1B |
| Nvidia T4 | 16 GB | ~1.0B | ~0.6B |


Get's you Gemma 4, GPT-3 range

Best model on B300:
- ~$35k
- 11.5B params
- ~230B tokens
- 205 days

On custered 8xB300:
- ~$35k (GPU/hr)
- 11.5B params
- ~230B tokens
- 26-34 days
