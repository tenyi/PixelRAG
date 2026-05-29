# Training Developer Notes

This file keeps internal training notes that are useful for maintainers but too
noisy for the top-level reproduction README.

## Core Files

| File | Description |
|------|-------------|
| `train_contrastors.py` | Primary training script for the reproduced embedding fine-tune. Uses InfoNCE, hard negatives, GradCache, and optional ViT LoRA. |
| `mine_hard_negatives.py` | Mines near-miss documents with the base model for hard-negative training. |
| `filter_hard_negatives_vqa.py` | Filters mined retrieval candidates with a VLM so false negatives are not used as hard negatives. |
| `run_filter_hard_negatives_chunks.py` | Runs VQA hard-negative filtering in fixed-size chunks, useful for large JSONL files. |
| `verify_embeddings.py` | Compares base vs fine-tuned embeddings with similarity margins and retrieval metrics. |
| `tests/test_grad_equivalence.py` | Single-GPU gradient correctness tests for GradCache math. |
| `tests/test_grad_multi_gpu.py` | Multi-GPU DDP/gather/loss-scaling gradient correctness tests. |

Legacy / secondary scripts:

- `train_colpali.py` — HF Trainer-based path. Simpler, but not the reproduced training run.
- `train_swift.py` — ms-swift alternative.
- `train.py`, `model.py`, `dataset.py`, `evaluate.py` — older local training/eval code.

## Data Formats

Basic JSONL format:

```json
{"query": "What is the population of Tokyo?", "chunk_path": "/path/to/chunk.png"}
```

Hard-negative format:

```json
{"query": "...", "chunk_path": "...", "neg_chunk_paths": ["/path/to/neg1.png", "/path/to/neg2.png"]}
```

Hard-negative-with-retrieval-candidates format, used before VQA filtering:

```json
{"query": "...", "chunk_path": "...", "neg_chunk_paths": ["..."], "retrieve_top20": [{"rank": 1, "path": "...", "score": 0.61}]}
```

Notes:

- `chunk_path` is resolved relative to the JSONL file when paths are relative.
- Training images are screenshot chunks; last chunks can be smaller than the common tile size.
- `neg_chunk_paths` should contain mined hard negatives, not random negatives.

## Training Pipeline

1. Data loading pre-validates images at init so all DDP ranks process the same number of batches.
2. `BiQwen3Processor` handles text tokenization and image preprocessing with visual-token resolution control.
3. `BiQwen3` embeds the text query and image document into single L2-normalized vectors.
4. InfoNCE is computed over the similarity matrix.
5. With hard negatives, docs are interleaved as `[pos, neg1, neg2, pos, neg1, neg2, ...]`.
6. Multi-GPU training uses `gather_with_grad` so document embeddings from other ranks contribute gradients.
7. GradCache keeps activation memory tied to `--grad-cache-chunk` rather than the full effective batch.

## VQA Filtering For Hard Negatives

`filter_hard_negatives_vqa.py` removes false negatives from mined retrieval candidates:

1. Read `retrieve_top20`.
2. Skip the positive `chunk_path`.
3. Check up to the first `K` non-positive candidates (`--candidate-k`, default `10`).
4. Ask the VLM to answer the query from each candidate image.
5. Judge that answer on the same image.
6. If verdict is `CORRECT`, treat the candidate as a false negative and skip it.
7. If verdict is `WRONG` or `CANNOT_ANSWER`, keep it as a hard negative.
8. Stop after collecting `--num-hard-negatives` hard negatives.
9. Skip the example if not enough hard negatives are found within the first `K` candidates.

Example:

```bash
OPENAI_API_KEY=... python filter_hard_negatives_vqa.py \
    --input /tmp/sample_100_hn.jsonl \
    --output /tmp/sample_100_hn_v2.jsonl \
    --reviews-output /tmp/sample_100_hn_v2.reviews.jsonl \
    --summary-output /tmp/sample_100_hn_v2.summary.json \
    --candidate-k 10 \
    --num-hard-negatives 2 \
    --concurrency 8
```

For large files:

```bash
OPENAI_API_KEY=... python run_filter_hard_negatives_chunks.py \
    --input training/data/lite-query-v2-full-filtered-hn.jsonl \
    --output-dir training/data/lite-query-v2-full-filtered-hn-v2-chunks \
    --chunk-size 10000 \
    --candidate-k 10 \
    --num-hard-negatives 2 \
    --concurrency 8 \
    --skip-existing
```

Each chunk folder contains:

- `filtered_hn.jsonl`
- `candidate_reviews.jsonl`
- `summary.json`

`summary.json` is updated incrementally and tracks missing-path ratios for positive paths,
reviewed candidate paths, and all checked paths combined.

## Dataset Packaging

For regenerating or uploading a Hugging Face dataset:

```bash
# Prepare HF dataset folder: convert absolute paths to relative paths and hardlink images.
python prepare_hf_dataset.py \
    --split-dir training/data/lite-query-v2-full-filtered-hn-v2-chunks/split \
    --image-root /opt/dlami/nvme/kiwix_tiles \
    --output-dir hf_dataset_export/screenshot-training

# Package images into tar shards.
python package_hf_image_shards.py \
    --source-dir hf_dataset_export/screenshot-training \
    --output-dir hf_dataset_export_sharded/screenshot-training

# Upload to Hugging Face.
python upload_hf_dataset.py \
    --local-dir hf_dataset_export_sharded/screenshot-training
```

Upload requires a Hugging Face token with write permission.

## Test Commands

```bash
# Single-GPU: GradCache math + RandContext + clip_loss + rope_deltas
CUDA_VISIBLE_DEVICES=0 uv run python tests/test_grad_equivalence.py

# Multi-GPU: DDP + gather + loss scaling + gradient sync
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 tests/test_grad_multi_gpu.py
```

## Archived / Experimental Mode

`train_contrastors.py` still has a `query-side-tune` mode that trains only the
query tower while keeping the doc/image tower frozen, so datastore embeddings do
not change. This is not part of the reproduction path in `README.md`.

Important caveats:

- Query-side retrieval eval depends on an external search API that accepts pre-computed `embedding` queries.
- `--query-side-backward direct` was the stable path in smoke tests.
- `--query-side-backward gradcache` was experimental and may hang in multi-GPU real-data runs.
