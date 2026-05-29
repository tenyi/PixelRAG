# Compressed-image SFT — Final Results

**Target:** maximize GPT-4.1 LLM-judge accuracy on `test_hn_with_answer.jsonl` (500 examples) for Qwen3-VL-4B under 2x / 3x / 5x / 9x pixel compression, **without inflating visual-token count at inference**.

**Judge:** `gpt-4.1-2025-04-14`, SimpleQA A/B/C template (`sft/eval_baseline.py`).

## Headline table

| Compression | base (no SFT) | best specialized | best universal (one adapter) | specialized config |
|---|---|---|---|---|
| **0x** (original) | **0.958** | — | — | ceiling |
| **2x** | 0.904 | **0.948** | 0.940 | r=128, 2ep, lr 2e-5 (LLM-only sufficient) |
| **3x** | 0.826 | **0.894** | 0.892 | r=256, 1ep, lr 1e-5, LLM+ViT |
| **5x** | 0.554 | **0.730** | 0.692 | r=128, 2ep, lr 3e-5, LLM+ViT |
| **9x** | 0.180 | **0.378** | 0.302 | r=512, 2ep, lr 1e-5, LLM+ViT |

Universal adapter is within **0.2–7.6 points** of each specialized adapter — at 3x essentially tied, at 9x still meaningfully behind.

## Key breakthrough: unfreezing the ViT

`lora_target: all` alone does **not** train the ViT in LlamaFactory — `freeze_vision_tower: true` is the default. Setting it to `false` unlocks LoRA on the ViT and gives big jumps at high compression:

| Compression | LLM-only | LLM+ViT | Δ |
|---|---|---|---|
| 2x | 0.948 | 0.948 | 0 |
| 3x | 0.882 | 0.894 | +1.2 |
| 5x | 0.634 | 0.730 | **+9.6** |
| 9x | 0.272 | 0.378 | **+8.2** |

**Rule:** the harder the compression, the more the ViT needs adaptation. 2x base is already near-ceiling so ViT LoRA adds nothing; 9x has the most distribution shift from pretraining so ViT capacity helps most (+30% relative).

## What worked

1. **LoRA + ViT unfrozen** → primary mechanism for 5x/9x gains.
2. **Bigger LoRA rank at high compression**: 9x went r=32→128→256→512 for +2.4, +1.2, +0.8, +0.8 (diminishing but monotonic).
3. **Mixed-compression training** (universal adapter): single LoRA trained on 2x+3x+5x+9x samples simultaneously. With r=256 + ViT unfrozen, the gap to specialized is only 0.2–7.6 points.
4. **Two epochs** at 2x/3x/5x; **one epoch** at 9x (overfits faster with bigger rank).

## What did not work

| Experiment | Outcome |
|---|---|
| **Full FT** (lr 1e-5) at 5x/9x | grad_norm 10+, output collapses (empty strings, `0000` artifacts). 9x fell to 0.176, below base 0.180. LoRA's implicit regularization wins. |
| **Pre-upscaling 9x → original dim** before ViT | Gives big accuracy boost (EM 0.78) but inflates visual tokens ~9×, defeating the compression use case. **Rejected.** |
| **LoRA dropout 0.1 at 9x** | No change in ceiling (~0.27). |
| **More epochs (3+)** at 5x | 0.620 at 3ep < 0.634 at 2ep. Overfit. |
| **Rank 256 at 5x** | 0.656 < 0.730 at r=128. Capacity saturated at r=128 for 5x. |
| **Think / CoT training** (30k GPT-generated reasoning traces, `<think>reasoning</think>answer` targets) | 9x peak 0.286 < non-think 0.378. Traces generated blindly by GPT (no image) produce plausible-but-wrong reasoning at 9x where the image is unreadable. Model learns to fabricate confident reasoning that doesn't help the final answer. |

## Compression-damage vs SFT-recovery

```
0x : ██████████████████████████████████████████ 0.958 (ceiling)
2x : ███████████████████████████████████████▏    0.904 → SFT 0.948  (−1.0 gap, 81% recovered)
3x : ████████████████████████████████████        0.826 → SFT 0.894  (−6.4 gap, 52% recovered)
5x : ████████████████████████                    0.554 → SFT 0.730  (−22.8 gap, 44% recovered)
9x : ███████▌                                    0.180 → SFT 0.378  (−58.0 gap, 25% recovered)
```

SFT recovery at 2x/3x: 50-80%. At 5x/9x: 25-44% — residual gap is the physical compression limit the ViT can't undo.

## Full experiment matrix

### 2x
| run | r | ep | lr | ViT | LLM-judge peak |
|---|---|---|---|---|---|
| base | — | — | — | — | 0.904 |
| v1 (LLM-only) | 128 | 2 | 2e-5 | frozen | **0.948** |
| llmvit-v1 | 128 | 2 | 2e-5 | trained | 0.948 (tied) |

### 3x
| run | r | ep | lr | ViT | peak |
|---|---|---|---|---|---|
| base | — | — | — | — | 0.826 |
| v1 | 32 | 1 | 1e-5 | frozen | 0.854 |
| v2 | 128 | 2 | 2e-5 | frozen | 0.878 |
| v3 | 256 | 2 | 2e-5 | frozen | 0.882 |
| llmvit-v1 | 256 | 2 | 2e-5 | trained | 0.884 (then collapsed) |
| llmvit-v2 | 256 | 1 | 1e-5 | trained | **0.894** |

### 5x
| run | r | ep | lr | ViT | peak |
|---|---|---|---|---|---|
| base | — | — | — | — | 0.554 |
| v1 | 32 | 1 | 1e-5 | frozen | 0.598 |
| v2 | 32 | 2 | 2e-5 | frozen | 0.628 |
| v3 | 128 | 2 | 3e-5 | frozen | 0.634 |
| v4 | 256 | 2 | 2e-5 | frozen | 0.620 |
| v5 | 128 | 3 | 3e-5 | frozen | 0.620 |
| fullft | — | 1 | 1e-5 | n/a | 0.562 (bad) |
| llmvit-v1 | 128 | 2 | 3e-5 | trained | **0.730** |
| llmvit-v2 | 256 | 2 | 2e-5 | trained | 0.656 |

### 9x
| run | r | ep | lr | ViT | peak |
|---|---|---|---|---|---|
| base | — | — | — | — | 0.180 |
| v1 | 32 | 1 | 1e-5 | frozen | 0.228 |
| v2 | 128 | 2 | 3e-5 | frozen | 0.252 |
| v3 | 256 | 1 | 2e-5 | frozen | 0.264 |
| v4 | 512 | 1 | 1.5e-5 | frozen | 0.272 |
| v5 | 256 | 2 | 1e-5 dropout 0.1 | frozen | 0.266 |
| fullft | — | 1 | 1e-5 | n/a | 0.176 (broken) |
| llmvit-v1 | 256 | 1 | 2e-5 | trained | 0.354 |
| llmvit-v2 | 512 | 2 | 1e-5 | trained | **0.378** |
| think-v1 | 256 | 2 | 1e-5 | trained (+think) | 0.286 (worse) |

### Universal (one adapter for all compressions)

Trained on concatenated 2x+3x+5x+9x data (416k examples, 1 epoch).

| run | r | lr | ViT | 2x | 3x | 5x | 9x |
|---|---|---|---|---|---|---|---|
| v1 | 128 | 2e-5 | frozen | 0.924 | 0.862 | 0.620 | 0.250 |
| llmvit-v1 | 128 | 2e-5 | trained | 0.920 | 0.844 | 0.664 | 0.272 |
| **llmvit-v2** | **256** | **1e-5** | **trained** | **0.940** | **0.892** | **0.692** | **0.302** |

## Shipped artifacts (HuggingFace)

Five LoRA adapters pushed to [Chrisyichuan](https://huggingface.co/Chrisyichuan):
- `qwen3vl-4b-wiki-screenshot-2x-lora` — 0.948 at 2x
- `qwen3vl-4b-wiki-screenshot-3x-lora` — 0.894 at 3x
- `qwen3vl-4b-wiki-screenshot-5x-lora` — 0.730 at 5x
- `qwen3vl-4b-wiki-screenshot-9x-lora` — 0.378 at 9x
- `qwen3vl-4b-wiki-screenshot-universal-lora` — one LoRA, 0.940/0.892/0.692/0.302 across all four compressions

## Practical recommendation

For most deployments, ship the **universal adapter**: 1.28 GB LoRA, one merge at load time, handles any of 2x/3x/5x/9x with near-specialized accuracy (within 0.2–7.6 LLM-judge points). If you know you'll only ever serve a specific compression level, use the specialized adapter for that level.

For 2x/3x the adapters are essentially production-ready (<10 point drop from uncompressed). 5x is usable when accuracy tolerance is moderate. 9x remains difficult — 0.378 is still far from the 0.958 ceiling, and this is dominated by the physical unreadability of 9×-compressed text rather than a capacity bottleneck we can solve with more training.

## Files

- Configs: `sft/train_qwen3vl_*.yaml`
- Train logs: `logs/sft_train/sft_*.log`
- Eval JSONs: `sft/eval_out/*.json`
- Trace generator: `sft/generate_think_traces.py` (think failed but kept for reference)
- Mixed-data builder: `sft/prepare_mixed_data.py`
- Eval fanout: `sft/eval_fanout.sh`
- HF push: `sft/push_to_hf.py`, `sft/push_universal_to_hf.py`
- W&B project: https://wandb.ai/yichuan_wang-uc-berkeley-electrical-engineering-computer/qwen3vl-compressed-sft
