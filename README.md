# mbox
> project-based learning. one's output = what they know, what they can build

it's so easy to abstract away everything (pytorch, agents, etc). 
so build **minimalistic toy models** here.
as a **mirror of the mental model in one's head**.
goal is to suck out the abstraction to apply somewhere else.
so make sure its something quite foundational.

toy problems. as bare bones as possible. such that it fits on an a4 page. and can re-understand in < 30 sec.
anything beyond is fluff or just unabstracting too many things at once. 

so some mental-model building ground rules:
+ socrates method with chat
+ chat as reference lookup
+ speed up the mental model building cycle. build abstractions <-- chat/library --> refining
+ *for just reimplementing papers (no abstractions) go to [dl repo](https://github.com/funnyscar/dl), organized messily lolz sry*

projects:

✅ [mmore](minimal/mmore) - autoregressive: bengio2003, oord2016, vaswani2017-ish <br>
✅ [mapreduce](mapreduce) - wordcount dean2004 <br>
✅ [autograd](minimal/autograd) - pytorch reimplementation (remove abstractions), fused kernels<br>
✅ [sne / tsne](minimal/tsne) - hinton 2002, 2008, maybe imagenet future?<br>
✅ [flax,jax,xla](minimal/jax) - flax,jax,optax,orbax exploring the deepmind vertical research stack<br>
✅ [umaps](minimal/tsne) - rough umaps implemtation<br>
🟡 [vectorsearch](minimal/tsne) - HNSW, Malkov&Yashunin2016 <br>
🟡 [serve efficiency](inference) <br>
- kvcache,
- continuous batching
- vllm
- CUDA graphs, spec decoding, prefix caching, paged attention <br>
- quantization
🟡 [modal train/data-sharding](modal) - distributed computing model training<br>
🟡 [kernels](metal) cuDNN,cuBLASS, writing cuda kernels <br>
🟡 [cuda](metal) nvcc, ptx, sass, amd rocm, hip <br>
🟡 [cnfs](minimal/cnfs) - cnf, wasserstein <br>
🟡 [flashattention](minimal/tsne) - flashattention <br>
🟡 [words,gpt-*](minimal/words) - larger transformers models vaswani2018<br>
🟡 [imagenet](minimal/imagenet) - resnet, vit, deepdream, <br>
🟡 [spark, flink, databricks]() - matei2016 (i took his cs162 class lolz)<br>
🟡 [multimodality]() - gemini <br>
🟡 [interpretability]()
🟡 [ray]() <br>
🟡 [kubeflow]() <br>
🟡 [jepa]() -  <br>
🟡 minichat - <br>
⬜ [jaxgpt]() -  <br>
⬜ [post-training]() - sft, rlhf <br>
⬜ [pixelcnn]() - oord2016 <br>
⬜ [moe]() - shazeer2017<br>
⬜ [pysr]() - cranmer2023<br>
⬜ [gnome]() - <br>
⬜ [miniautopilot]() - <br>
⬜ fold - <br>
⬜ weather - <br>
⬜ decoding<br>
⬜ tail latency<br>
⬜ zookeeper, etcd<br>
⬜ parallelism, all-reduce<br>
⬜ sparse or linear attention, KV-cache compression <br>
⬜ nerfs, lightfields, worldmodels - <br>
⬜ video gen
⬜ triton - <br>
⬜ einops - <br>
⬜ FSDP / DeepSpeed (ZeRO) - <br>
⬜ megatron lm - <br>
⬜ post-trained - sft, rlhf, safety, alignment, tooling, mcp<br>
⬜ [alpha ___]() alphaevolve, alphadev, MuZero, ....  <br>
⬜ [agentic]() co-scientist <br>
* gambling engine
	* ⬜ 

* spack
* openmp, mpi, cuda, hip, fortran
* suitesparse, petc, trilios, sundials, eigen, Blas, Lapack, mfem
* sparse matrices
* LU, QR, SVD, eigenvalue problems, and iterative solvers (CG, GMRES)
* Floating point: precision, rounding, stability, conditioning.

* meshes
* fpga, verilog, eecs151
* navier stokes erodos
* FreeRTOS / Zephyr:
* lean4 swarms
* collierbox, firmware
* g++, clang, gcc, msvc, custom
* gRPC, protobufs
* OS, groundup
* computer design + fab
* python interpreters
* browsers
* agentic orchestration designs
* konnect, cad
* flops, ops speed
* openEMS, Meep, Optiland, rayoptics, Elmer, FEniCS, OpenFOAM, FreeCAD FEM, CalculiX, Elmer, Code_Aster, Molflow
* qiskit
* astrodynamics
* insane engines
* engines
* rocket eng


* [ ] bigtable
* [ ] presto, trino
* [ ] nccl
* [ ] spanner
* [ ] dynamo
* [ ] kafka
* [ ] cassandra
* [ ] s3, hdfs, gfs, lustre
* [ ] parquet,arrow
* [ ] raft
* [ ] CDNs

* [ ] Konnect
* [ ] hydrocode sims
* [ ] qlib


* networking, tcp stack, ipconfig ssh ports, rg, old servers
* abmb, tracing work of 1920 physicists

* cobol
* [ ] SAP
* [ ] DMAIC mini-project
* [ ] Queueing simulation in Python

* [ ] MIT OCW 2.008 – Design and Manufacturing II
* [ ] Measurement and tolerances (calipers, micrometers, GD&T basics)



> hanging up the cape on vibe coding, cuz i like learn nothing. these are fundamentals and why not...like a weekend project.





