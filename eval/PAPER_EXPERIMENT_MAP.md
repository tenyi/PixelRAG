# Paper Experiment Map
Maps paper results → source experiments in `~/pixelrag-src/Vis-RAG/agent/experiments/`.

## Shared Config (all paper experiments unless noted)
- **think**: enabled (no `--no-think` flag)
- **max_tokens**: 16384
- **retrieval_top_k**: 5
- **reader_top_k**: 3
- **query_instruction (pixel)**: "Retrieve images or text relevant to the user's query."
- **query_instruction (text)**: "Retrieve text relevant to the user's query."
- **Readers**: Qwen3-VL-4B-Instruct (VL-4B) and Qwen3.5-4B (Q3.5)

## Table 1: Text-centric Wikipedia QA

### SimpleQA → `simpleqa_paper_top3_v1`
- Script: `experiments/simpleqa_paper_top3_v1/run.sh`
- Grader: GPT-4o judge (`scripts/evaluate.py simpleqa`)
- Ports: base=30888, LoRA=30893, DoRA=30895, Traf=30889, NeuML=30896
- n=1000
- summary.tsv has graded_count, not accuracy (accuracy was in evaluate.py stdout)
- Outputs: `$EXP_DIR/outputs/sqa_*.jsonl` (cleaned/deleted)

### NQ → `nq_paper_top3_v1`
- Script: `experiments/nq_paper_top3_v1/run.sh`
- Grader: exact match
- n=1000
- summary.tsv has EM and F1

### NQ-Tables → `nqt_paper_top3_v1`
- Script: `experiments/nqt_paper_top3_v1/run.sh`
- Grader: exact match
- n=1068
- summary.tsv has EM and F1

### TriviaQA → `triviaqa_paper_top3_v1`
- Script: `experiments/triviaqa_paper_top3_v1/run.sh`
- Grader: exact match
- n=1000

## Table 1: Multimodal QA

### MMSearch → `mmsearch_paper_top3_v1`
- Script: `experiments/mmsearch_paper_top3_v1/run.sh`
- n=300
- summary.tsv has scores

### EVQA → `evqa_paper_top3_v1`
- Script: `experiments/evqa_paper_top3_v1/run.sh`
- Grader: GPT-4.1 judge
- n=1000 per subset (landmarks, inaturalist)
- NOTE: Q3.5 cells originally ran with `--no-think`, later backfilled in `q35_think_backfill_v1`

### LiveVQA → `livevqa_v3_qa_v1`
- Script: `experiments/livevqa_v3_qa_v1/run.sh` (if exists)
- Also backfilled in `q35_think_backfill_v1`

## Figure 2: Token Efficiency (SimpleQA)

### No-think version → `token_efficiency_q35_nothink_v1`
- Script: `experiments/token_efficiency_q35_nothink_v1/run.sh`
- max_tokens=200, --no-think
- summary.tsv has actual accuracy numbers:
  - base top1=0.575, top2=0.677, top3=0.722
  - LoRA top1=0.629, top2=0.719, top3=0.750
- These are NO-THINK numbers; paper Figure 2 likely uses think numbers

### Bug-fixed text version → `token_efficiency_v2`
- Fixed text retrieval bug (retrieval_top_k used instead of reader_top_k)
- Adds top-2 cells

## Table 3: Modality Ablation → `ablation_modality_v1`
- Script: `experiments/ablation_modality_v1/run.sh`

## Think vs No-Think

### `q35_nothink_full_v1`
- Full benchmark sweep with Q3.5 no-think (max_tokens=200)
- Intended as comparison to VL-4B paper runs

### `q35_think_backfill_v1`
- Re-runs Q3.5 cells with think enabled (max_tokens=16384)
- Matches VL-4B paper config exactly
- Backfills EVQA, NeuML text, LiveVQA

### `q35_matrix_completion_v1`
- Fills missing cells in think/no-think × retriever × k matrix
- Expected values noted in README:
  - no-think base top3: ~72.2%
  - think LoRA top3: ~77.9%
  - think Traf top3: ~70.2%

## Reference Numbers from Experiment Summaries

### NQ (EM, from nq_paper_top3_v1/summary.tsv)
q35: base=0.338, lora=0.328, dora=0.334, traf=0.280
vl4b: base=0.317, lora=0.311, dora=0.311, traf=0.294

### NQ-Tables (EM, from nqt_paper_top3_v1/summary.tsv)
q35: base=0.258, lora=0.275, dora=0.274, traf=0.227 (n=497!)
vl4b: base=0.241, lora=0.266, dora=0.271, traf=0.219

### MMSearch (score, from mmsearch_paper_top3_v1/summary.tsv)
q35: base=0.287, lora=0.277, dora=0.283, traf=0.253, naive=0.147
vl4b: base=0.240, lora=0.247, dora=0.240, traf=0.203, naive=0.130

### TriviaQA (EM, from triviaqa_paper_top3_v1/summary.tsv)
q35: base=0.718, lora=0.718, dora=0.710, traf=0.714 (n=248!)
vl4b: base=0.696, lora=0.713, dora=0.702, traf=0.731

### SimpleQA no-think (accuracy, from token_efficiency_q35_nothink_v1/summary.tsv)
base: top1=0.575, top2=0.677, top3=0.722
LoRA: top1=0.629, top2=0.719, top3=0.750

### SimpleQA think (expected, from q35_matrix_completion_v1/README.md)
base top3: ~72.2%  (no-think ~72.2% — think doesn't help base much)
LoRA top3: ~77.9%  (no-think 75.0% — think adds ~3%)
Traf top3: ~70.2%  (no-think ~68.5% est — think adds ~2%)

## Key Findings for Reproduction

1. **All paper Q3.5 numbers use think mode** (max_tokens=16384), not no-think
2. Our no-think runs are ~3-6% lower than paper think numbers (SimpleQA LoRA/Traf)
3. Base pixel is insensitive to think (72.2% think vs 72.2% no-think)
4. NQ/NQ-Tables use exact match grading, less sensitive to think/no-think
5. SimpleQA uses LLM judge (GPT-4o in paper, GPT-4.1 in ours)
6. The LoRA index needs the merged LoRA encoder model for query encoding
   - Adapter: `/opt/dlami/nvme/adapters/lora_vit_ckpt200/lora_vit/ckpt200`
   - Merged model: created at runtime via `PeftModel.from_pretrained()` + `merge_and_unload()`
   - See `embedding/embed_tiles.py:558-582`
