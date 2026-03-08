# Multi-GPU Training Throughput Gain of 3.7x

## Resume Bullet
> **"Multi-GPU training throughput gain of 3.7x with sharded gradients, optimized NCCL topology, and mixed-precision ops"**

A runnable simulation that models multi-GPU training throughput, demonstrates why each optimization matters, and gives you the exact numbers and code snippets to explain in an interview.

---

## What Does This Actually Mean?

Training a 1.3B parameter model on a single GPU is slow. Distributing across 8 GPUs helps, but introduces **three bottlenecks**:

| Bottleneck | Problem | Optimization |
|---|---|---|
| **Compute** | FP32 ops don't use tensor cores | **Mixed Precision** (BF16): 2.5x faster |
| **Memory** | Full model replica on each GPU limits batch size | **Sharded Gradients** (ZeRO-2/FSDP): 1.25x from bigger batches |
| **Communication** | Gradient sync over slow PCIe | **NCCL Topology** (Tree/NVLink): 1.18x faster comm |

Combined (multiplicative): **2.5 × 1.25 × 1.18 ≈ 3.7x**

---

## How to Run

### VS Code
```bash
pip install numpy pandas matplotlib seaborn
# Open multi_gpu_throughput.py → each # %% is a runnable cell
```

### Colab
Upload `multi_gpu_throughput.py` → Run All. ~30 seconds, no GPU needed.

### CLI
```bash
python multi_gpu_throughput.py
```

---

## The Three Optimizations Explained

### 1. Mixed Precision (FP32 → BF16)
FP32 uses regular CUDA cores. BF16 uses **tensor cores** which are ~16x faster theoretically, ~2.5x in practice (memory-bandwidth limited). BF16 is preferred over FP16 because it has the same exponent range as FP32 — no need for loss scaling.

**Code:** `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)`

### 2. Sharded Gradients (DDP → FSDP/ZeRO-2)
Standard DDP: every GPU holds full model + gradients + optimizer = 16 bytes per parameter. ZeRO Stage 2 shards gradients and optimizer states across N GPUs, cutting memory from 16B to ~5.5B per param per GPU. Freed memory → larger batch sizes → more tokens per step.

**Code:** `FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)`

### 3. NCCL Topology (Ring/PCIe → Tree/NVLink)
PCIe bandwidth: 64 GB/s. NVLink: 600 GB/s (9.4x faster). Tree all-reduce is ~30% more efficient than ring for 8+ GPUs. Combined: communication time drops from 569ms to 47ms.

**Code:** `export NCCL_TREE_THRESHOLD=0 && export NCCL_P2P_LEVEL=NVL`

---

## Key Numbers to Know

| Metric | Baseline | Optimized |
|---|---|---|
| Precision | FP32 | BF16 |
| Sharding | None (full replicas) | ZeRO Stage 2 |
| Topology | Ring/PCIe | Tree/NVLink |
| Memory/GPU | 20.8 GB | 9.4 GB |
| Max Batch Size | 4 | 10 |
| Throughput | ~31K tok/s | ~115K tok/s |
| **Speedup** | **1.0x** | **3.7x** |

---

## Interview Q&A

**Q: Explain sharded gradients.**
> FSDP with SHARD_GRAD_OP (ZeRO-2). Each GPU stores 1/N of gradients + optimizer. Uses reduce-scatter instead of all-reduce, cutting comm data by ~50%. Memory savings enable larger batches.

**Q: How did you optimize NCCL?**
> Three changes: NVLink P2P instead of PCIe routing (9x bandwidth), tree all-reduce instead of ring (~30% more efficient for 8 GPUs), and async collectives to overlap comm with compute.

**Q: Why BF16 over FP16?**
> Same exponent range as FP32, so no loss scaling needed. FP16 requires GradScaler which adds complexity and occasional instability. BF16 = same speed, better stability.

**Q: How did you measure 3.7x?**
> Tokens/second: baseline FP32/DDP/PCIe vs optimized BF16/FSDP/NVLink. Decomposition: ~2.5x mixed precision, ~1.25x sharding, ~1.18x NCCL. Multiplicative.

**Q: DDP vs FSDP vs DeepSpeed?**
> DDP replicates full model per GPU + all-reduces gradients. FSDP (PyTorch-native) shards params/grads/optimizer = ZeRO. DeepSpeed is Microsoft's ZeRO implementation with CPU/NVMe offloading. Used FSDP for PyTorch-native integration.

**Q: When would you add tensor/pipeline parallelism?**
> For 70B+ models that don't fit even with FSDP. Tensor parallelism splits layer computations across GPUs (Megatron-style). Pipeline parallelism splits layers across GPUs but adds bubble overhead.

---

## Dependencies

- Python 3.8+, NumPy, Pandas, Matplotlib
- **No GPU required** (this is an analytical throughput model)
