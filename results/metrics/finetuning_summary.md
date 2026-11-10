# Fine-tuning results (Greek, whisper-small + LoRA)

All runs: base `openai/whisper-small`, FLEURS Greek train (3,215 recordings, 10.0 h, nothing from dev/test/eval/rated sets), reference_raw targets,
effective batch 16, seed 42, Colab T4. Evaluated on the 333 Greek evaluation sentences, corpus-level WER/CER in %.
Sources: `models/*/training_config.json`, `training_log.csv`, `results/metrics/*_table.md`, `results/analysis_q3*/gains.csv`.

## Training setup

| Run | LoRA targets | Trainable params | Epochs | LR | SpecAugment | Train time (min) |
|:--|:--|--:|--:|--:|:--|--:|
| small-lora-el (main) | q_proj, v_proj | 3.5M | 3 | 5e-4 | no | 24.5 |
| small-lora-el-all (capacity) | q, k, v, out_proj, fc1, fc2 | 13.0M | 3 | 5e-4 | no | 32.0 |
| small-lora-el-reg (regularised) | q_proj, v_proj | 3.5M | 2 | 2e-4 | yes (5% time, 5% mel) | 16.1 |

## Dev log (final epoch; best epoch in brackets)

| Run | Train loss | Dev loss | Dev WER |
|:--|--:|--:|--:|
| zero-shot small (epoch 0) | – | 0.609 | 34.0 |
| small-lora-el | 0.188 | 0.381 (0.379, ep 2) | 29.3 (28.9, ep 2) |
| small-lora-el-all | 0.044 | 0.356 (0.344, ep 2) | 26.9 (25.9, ep 2) |
| small-lora-el-reg | 0.446 | 0.394 | 31.4 |

## Evaluation set WER / CER (%)

| Model | WER raw | WER standard | WER content | CER raw | CER standard | CER content |
|:--|--:|--:|--:|--:|--:|--:|
| small (zero-shot baseline) | 36.04 | 33.06 | 30.87 | 12.44 | 12.78 | 10.75 |
| small-lora-el | 31.03 | 27.85 | 25.67 | 10.04 | 10.57 | 8.68 |
| small-lora-el-all | **27.67** | **24.42** | **22.25** | **9.11** | **9.66** | **7.74** |
| small-lora-el-reg | 31.97 | 29.30 | 27.04 | 10.65 | 11.18 | 9.16 |
| *medium (zero-shot, reference)* | 23.83 | 20.19 | 18.03 | 8.05 | 8.67 | 6.70 |
| *large-v3 (zero-shot, reference)* | 16.42 | 13.04 | 11.10 | 5.42 | 6.07 | 4.21 |

## Gain over the zero-shot small baseline (WER, absolute points; relative reduction; 95% bootstrap CI of the relative reduction)

| Run | raw | standard | content |
|:--|:--|:--|:--|
| small-lora-el | -5.0 pts; 13.9% [10.8, 16.9] | -5.2 pts; 15.8% [12.6, 18.9] | -5.2 pts; 16.8% [13.7, 20.0] |
| small-lora-el-all | -8.4 pts; 23.2% [20.2, 26.3] | -8.6 pts; 26.1% [22.8, 29.2] | -8.6 pts; 27.9% [24.5, 31.2] |
| small-lora-el-reg | -4.1 pts; 11.3% [8.8, 13.8] | -3.8 pts; 11.4% [8.8, 14.0] | -3.8 pts; 12.4% [9.8, 15.1] |

## Other fine-tuned results

| Run | Utterances at WER 0 | Perfect utts % (standard) | Median utt WER (standard) | Utts >30% WER |
|:--|--:|--:|--:|--:|
| small (baseline) | 4 | 1.2 | 31.6 | 179 |
| small-lora-el | 10 | 3.0 | 26.5 | 132 |
| small-lora-el-all | 16 | 4.8 | 22.2 | 99 |
| small-lora-el-reg | 7 | 2.1 | 27.3 | 142 |

## Takeaways

- Every run improves on zero-shot small; none reaches zero-shot medium (20.2 standard WER).
- More LoRA capacity (all linear layers) is the best change: 24.4 vs 27.8 standard WER for the main run, and best on dev too.
- Regularising (lower LR, SpecAugment, 2 epochs) underfits and is worse than the main run, so overfitting is not the main limit.
- The WER gain is not cosmetic: the relative reduction is at least as large under content-aware as under raw normalisation.
- Human ratings (Q3) exist only for `small-lora-el`; the other two are automatic-metric checks.
