# GradCache Training (train_contrastors.py)

GradCache training for Qwen3-VL-Embedding, adapted from [nomic-ai/contrastors](https://github.com/nomic-ai/contrastors). Verified correct via gradient equivalence tests (single-GPU + multi-GPU).

## Scope Note

This document describes the original GradCache path used by `train_contrastors.py` in `standard` mode.

For `--mode query-side-tune`:

- default backward path is `--query-side-backward direct`
- `--query-side-backward gradcache` still exists, but is currently experimental
- multi-GPU real-data smoke tests were stable with `direct` and unstable with `gradcache`

## Why GradCache?

Contrastive learning benefits from large batch sizes (more negatives = better). But GPU memory limits batch size. GradCache breaks this constraint:

```
Standard:  batch=4, chunk=4 → memory of 4, negatives = 4
GradCache: batch=4, chunk=2 → memory of 2, negatives = 4
Multi-GPU: batch=4, chunk=2, 5 GPUs → memory of 2, negatives = 20
```

## How It Works

GradCache splits each batch into small chunks and processes them in 3 steps:

1. **Forward all chunks WITHOUT grad** → cache embeddings + RNG states (constant memory)
2. **Compute InfoNCE loss on ALL cached embeddings** → get embedding gradients via backward on detached tensors
3. **Replay forward WITH grad** using saved RNG states, apply surrogate loss (`dot(emb, cached_grad)`) → real parameter gradients

By the chain rule, `d(loss)/d(θ) = d(loss)/d(emb) · d(emb)/d(θ)`. Step 2 computes the first factor, step 3 computes the second. The result is mathematically identical to a full-batch backward pass.

### DDP Integration

Multi-GPU adds two distributed primitives:

- **`gather_with_grad`**: All-gathers document embeddings across ranks (step 2). Backward does `reduce_scatter` to distribute gradients back.
- **Manual `all_reduce(AVG)`**: All surrogate backward calls run under `no_sync()` to avoid DDP reducer deadlocks (query chunks skip the visual encoder while doc chunks use it → different "used" parameter sets confuse `find_unused_parameters`). Gradients are manually synced after all chunks.

The loss is scaled by `world_size` before backward, so after `all_reduce(AVG)`:
```
final_grad = (1/W) × Σ_r [W × d(CE_r)/d(θ)] = Σ_r d(CE_r)/d(θ) = d(total_CE)/d(θ)
```

### Key Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `RandContext` | contrastors/rand_state.py | Save/restore GPU RNG state for dropout replay |
| `gather_with_grad` | contrastors/distributed.py | All-gather with gradient flow (backward = reduce_scatter) |
| `clip_loss` | contrastors/loss.py | InfoNCE with learnable logit scale + hard negative support |
| `grad_cache_loss` | contrastors/loss.py | Full GradCache pipeline (3-step) |
| `LogitScale` | contrastors/OpenCLIP | Learnable `log_scale` parameter, clamped post-step |
| `BiQwen3` | colpali-engine | Qwen3-VL wrapped as bi-encoder (last-token pool + L2 norm) |

### LogitScale

Learnable temperature in log-space, initialized to `ln(1/0.07) ≈ 2.66`:

```python
forward:  similarity * exp(log_scale)     # no clamp in forward (avoids gradient dead zone)
after optimizer.step():  log_scale.clamp_(0, ln(100))  # contrastors pattern
```

## Quick Start

```bash
PYTHON=.venv-sglang/bin/python

# Single GPU
CUDA_VISIBLE_DEVICES=3 $PYTHON training/train_contrastors.py \
    --max-steps 500 --batch-size 4 --grad-cache-chunk 2

# Multi-GPU (5 GPUs, cross-GPU negatives + GradCache)
CUDA_VISIBLE_DEVICES=3,4,5,6,7 .venv-sglang/bin/torchrun --nproc_per_node=5 \
    training/train_contrastors.py --max-steps 500 --batch-size 8 --grad-cache-chunk 2

# With hard negatives (requires train_hn.jsonl from mine_hard_negatives.py)
CUDA_VISIBLE_DEVICES=3,4,5,6,7 .venv-sglang/bin/torchrun --nproc_per_node=5 \
    training/train_contrastors.py --train-jsonl training/data/train_hn.jsonl \
    --num-hard-negatives 2 --batch-size 4 --grad-cache-chunk 2

# Resume from checkpoint
$PYTHON training/train_contrastors.py \
    --resume training/output_contrastors/checkpoint-200
```

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--batch-size` | 4 | Per-GPU batch size |
| `--grad-cache-chunk` | 2 | Chunk size for GradCache (memory = this × per-sample cost) |
| `--lr` | 2e-5 | Peak learning rate (cosine schedule) |
| `--warmup-steps` | 50 | Linear warmup |
| `--max-steps` | 500 | Total training steps |
| `--temperature` | 0.07 | Initial temperature (learnable logit scale = 1/temp) |
| `--num-hard-negatives` | 0 | Hard negs per query (docs interleaved: [pos, neg1, neg2, ...]) |
| `--lora-r` | 32 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--max-num-visual-tokens` | 256 | Image resolution (~200K pixels) |
| `--max-grad-norm` | 1.0 | Gradient clipping (model + logit_scale) |

## Comparison with train_colpali.py

| Feature | train_colpali.py | train_contrastors.py |
|---------|-----------------|---------------------|
| Training infra | HF Trainer (ContrastiveTrainer) | Custom loop |
| GradCache | No | Yes |
| Cross-GPU negatives | `all_gather` | `gather_with_grad` |
| Temperature | Fixed | Learnable (LogitScale) |
| Hard negatives | No | Yes (`--num-hard-negatives`) |
| DDP strategy | Standard DDP | `no_sync` + manual `all_reduce` |
| Checkpoint/resume | HF Trainer | Manual |

## Verified Correct

Gradient equivalence tests confirm GradCache produces identical gradients to a full-memory reference (15 tests, all passing):

**Single-GPU** (`tests/test_grad_equivalence.py`):
- GradCache chain-rule decomposition: cosine ≥ 0.9999 for chunk_size = 1, 2, batch_size
- RandContext dropout replay: cosine ≥ 0.9999 for all chunk sizes
- `clip_loss` label arithmetic: basic, hard negatives, divisibility assertion
- `_clear_rope_deltas`: prevents image→text rope state leakage

**Multi-GPU** (`tests/test_grad_multi_gpu.py`, 2×GPU):
- GradCache DDP vs reference: cosine ≥ 0.9998 (with and without dropout)
- `gather_with_grad` backward: reduce_scatter gives correct gradient = W
- `loss*W + all_reduce(AVG)` = gradient of total loss: exact match
- Gradients identical across ranks after sync: max diff = 0

```bash
# Run tests
CUDA_VISIBLE_DEVICES=2 python training/tests/test_grad_equivalence.py
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 training/tests/test_grad_multi_gpu.py
```

## Code Attribution

Core GradCache implementation adapted from [nomic-ai/contrastors](https://github.com/nomic-ai/contrastors):
- `contrastors/loss.py` — `grad_cache_loss`, `clip_loss`, `get_chunked_embeddings`
- `contrastors/rand_state.py` — `RandContext`
- `contrastors/distributed.py` — `gather_with_grad`
- `contrastors/models/biencoder/modeling_biencoder.py` — `LogitScale`
- `contrastors/trainers/text_text.py` — post-step `clamp_()` pattern
