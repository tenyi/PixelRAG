# Swift Training Pipeline (`train_swift.py`)

Alternative to `train_contrastors.py` using [ms-swift](https://github.com/modelscope/ms-swift)'s built-in embedding training. Simpler (single `sft_main()` call) but lacks GradCache and learnable temperature.

## Quick Start

```bash
# 0. Install
uv sync

# 1. Convert data from contrastors format to swift format
uv run python convert_data_for_swift.py \
    --input training/data/train_hn.jsonl --output data/train_hn_swift.jsonl
uv run python convert_data_for_swift.py \
    --input training/data/eval.jsonl --output data/eval_swift.jsonl

# 2. Train (2 GPUs, best config)
CUDA_VISIBLE_DEVICES=1,2 uv run python train_swift.py \
    --train-jsonl data/train_hn_swift.jsonl \
    --eval-jsonl data/eval_swift.jsonl \
    --num-hard-negatives 5 \
    --batch-size 4 \
    --lr 1e-5 \
    --max-steps 50 \
    --warmup-steps 20 \
    --eval-steps 25 \
    --save-steps 50 \
    --nproc-per-node 2

# 3. Remap adapter keys for BiQwen3 eval compatibility
#    (see "Adapter Key Remap" section below)

# 4. Eval with BiQwen3
cd /home/user/Vis-RAG/agent && \
CUDA_VISIBLE_DEVICES=4 uv run python scripts/eval_lora_checkpoint.py \
    /path/to/checkpoint-biqwen3 --tiles-dir tiles-hard-mini-v6
```

## Data Conversion

Swift expects a different JSONL format than `train_contrastors.py`. Use `convert_data_for_swift.py` to convert:

**Input** (contrastors format):
```json
{"query": "...", "chunk_path": "/path/to/pos.png", "neg_chunk_paths": ["/path/to/neg1.png", ...]}
```

**Output** (swift format):
```json
{
  "messages": [
    {"role": "system", "content": "Retrieve images or text relevant to the user's query."},
    {"role": "user", "content": "<query>"}
  ],
  "positive_messages": [[
    {"role": "system", "content": "Represent the user's input."},
    {"role": "user", "content": "<image>"}
  ]],
  "positive_images": [["/path/to/pos.png"]],
  "negative_messages": [
    [{"role": "system", "content": "Represent the user's input."}, {"role": "user", "content": "<image>"}]
  ],
  "negative_images": [["/path/to/neg1.png"]]
}
```

Instructions (`QUERY_INSTRUCTION`, `DOC_INSTRUCTION`) match `train_contrastors.py` exactly.

**Important**: If `--num-hard-negatives > 0`, **eval data must also have negatives** or swift will crash. Convert from `eval_hn.jsonl` (not `eval.jsonl`) in that case, or mine negatives for eval data too.

## Key Args

| Arg | Default | Notes |
|-----|---------|-------|
| `--batch-size` | 4 | Per-GPU. **No GradCache** — memory scales with batch size |
| `--lr` | 2e-5 | Cosine schedule (or `--scheduler constant`) |
| `--warmup-steps` | 50 | Linear warmup |
| `--temperature` | 0.07 | **Fixed** (not learnable, set via `INFONCE_TEMPERATURE` env var) |
| `--num-hard-negatives` | 0 | Must match data format |
| `--lora-r` / `--lora-alpha` | 32 / 32 | Same as contrastors |
| `--max-num-visual-tokens` | 4096 | Converted to max_pixels internally (×784) |
| `--max-steps` | 500 | Total training steps |
| `--eval-steps` | 100 | Validation loss frequency |
| `--save-steps` | 100 | Checkpoint frequency |
| `--deepspeed` | None | `zero2` or `zero3` for multi-GPU memory savings |
| `--freeze-vit` | True | Freeze vision encoder |
| `--nproc-per-node` | 1 | Number of GPUs |
| `--resume` | None | Path to checkpoint directory to resume from |

## Adapter Key Remap

Swift wraps the model in `Qwen2_5_VLForConditionalGeneration`, which adds an extra `model.` layer in adapter key names:

- **Swift saves**: `base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight`
- **BiQwen3 expects**: `base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight`

Before running BiQwen3 eval, remap keys:

```python
import shutil, os
from safetensors.torch import load_file, save_file

ckpt = 'training/output_swift/vX-XXX/checkpoint-50'
out = ckpt + '-biqwen3'
shutil.copytree(ckpt, out, dirs_exist_ok=True)
state = load_file(f'{out}/adapter_model.safetensors')
remapped = {k.replace('base_model.model.model.', 'base_model.model.'): v for k, v in state.items()}
save_file(remapped, f'{out}/adapter_model.safetensors')
```

Without this remap, the adapter loads silently but has zero effect — all eval metrics will match the base model.

## Known Differences vs `train_contrastors.py`

| Aspect | contrastors | swift | Impact |
|--------|-------------|-------|--------|
| **GradCache** | Yes (decouple batch from memory) | No | batch_size=4 max vs 16 with GradCache |
| **Temperature** | Learnable `LogitScale(1/0.07)` | Fixed 0.07 | Temperature can't adapt during training |
| **Retrieval eval** | R@1/5/10, MRR during training | Loss-only eval | Run retrieval eval separately after training |
| **Cross-GPU gather** | `gather_with_grad` (full gradients) | `gather_object` + detach (local grads only) | Doc gradient cosine=0.979 on 2 GPUs |
| **Adapter keys** | `base_model.model.language_model.*` | `base_model.model.model.language_model.*` | Must remap keys for BiQwen3 eval (see above) |

### Why these differences exist

- **No GradCache**: ms-swift uses HF Trainer directly, which doesn't support GradCache's 3-step (no-grad forward → cache → surrogate backward) technique. This means GPU memory scales with batch size.
- **Fixed temperature**: swift sets `INFONCE_TEMPERATURE` via env var at startup. There's no `LogitScale` module — the temperature is a constant divisor in the loss function.
- **No retrieval eval**: swift's `EmbeddingTrainer` only computes eval loss (margin, mean_pos, mean_neg). Full retrieval eval (building a corpus index, computing R@K) must be done post-training.
- **Detached cross-GPU gather**: swift uses `dist.gather_object` which detaches tensors. Contrastors uses a custom `gather_with_grad` that preserves gradients through all ranks. On 2 GPUs, doc gradient cosine similarity is 0.979 — close but not identical.

## Equivalence Tests

9 tests verify numerical equivalence between the two pipelines:

```bash
# Single-GPU tests (8 tests)
CUDA_VISIBLE_DEVICES=0 uv run python tests/test_swift_equivalence.py

# Multi-GPU gather gradient test (requires 2 GPUs)
CUDA_VISIBLE_DEVICES=1,2 uv run torchrun --nproc_per_node=2 \
    tests/test_swift_equivalence.py --multi-gpu
```

| Test | What it verifies | Result |
|------|-----------------|--------|
| Tokenization | Same input_ids for identical query/doc | Exact match |
| Embeddings | Same vectors from same model weights | cosine > 0.999 |
| Loss computation | InfoNCE loss formula identical | diff = 0 |
| LoRA targets | Same 224 trainable parameters | Exact match |
| Hard negative labels | Label construction matches | diff = 0 |
| Data collation | `convert_data_for_swift.py` output matches | Bitwise match |
| Training step | Single step loss convergence | diff = 0.007 |
| Single-GPU gather | Gather produces same result | diff = 0 |
| Multi-GPU gather | Loss and gradient comparison | Loss diff = 0, doc grad cosine = 0.979 |

## Training Results (1000-step run, 2026-04-03)

Config: 2× GPU, batch_size=4, lr=1e-5, 5 hard negatives, cosine schedule, warmup=20 steps. Training time: ~4 hours.

### Eval Loss

| Step | eval_loss | mean_pos_sim | mean_neg_sim |
|------|-----------|-------------|-------------|
| 100 | 0.2507 | 0.607 | 0.064 |
| 200 | 0.1945 | 0.663 | 0.050 |
| 300 | 0.1827 | 0.672 | 0.037 |
| 500 | **0.1725** | 0.683 | 0.024 |
| 700 | 0.1705 | 0.687 | 0.017 |
| 1000 | 0.1699 | 0.685 | 0.015 |

Best eval_loss at step 700–1000 (plateau), but retrieval metrics peak at step 500.

### Retrieval Eval (verify_embeddings.py, 500 pairs from eval.jsonl)

| Metric | Base (no fine-tune) | checkpoint-500 | checkpoint-1000 |
|--------|-------------------|----------------|-----------------|
| **R@1** | 60.4% | **65.8%** (+5.4%) | 65.4% (+5.0%) |
| **R@5** | — | **79.8%** | 77.0% |
| **R@10** | — | **80.0%** | 78.4% |
| **MRR** | 0.683 | **0.725** | 0.724 |
| mean_pos_sim | 0.433 | 0.559 | 0.558 |
| margin | 0.301 | 0.534 | 0.541 |

**Best checkpoint: step 500.** Step 1000 shows slight overfitting (R@1 drops 65.8→65.4%). Eval loss keeps decreasing after 500 but retrieval accuracy doesn't — the model is fitting to the loss metric without improving actual retrieval.

## Resume Training

```bash
uv run python train_swift.py --resume training/output_swift/vX-XXX/checkpoint-50
```

## How It Works Internally

`train_swift.py` sets environment variables and calls `sft_main(SftArguments(...))`:

1. **Env vars**: `INFONCE_TEMPERATURE`, `INFONCE_USE_BATCH` (in-batch negatives), `INFONCE_HARD_NEGATIVES`, `NPROC_PER_NODE`
2. **Model**: `Qwen2_5_VLForConditionalGeneration` with `lm_head` monkey-patched to `nn.Identity()` (embedding mode)
3. **Pooling**: last-token hidden state, L2-normalized
4. **Loss**: `InfonceLoss` — cross-entropy on cosine similarity matrix / temperature
5. **LoRA**: applied to `q_proj`, `k_proj`, `v_proj`, `o_proj` (same as contrastors)
6. **Trainer**: `EmbeddingTrainer` (subclass of HF `Trainer`)
