# Composite-control verifier v2 — rejected replacement

2026-11-05. This experiment is a negative result; keep v1 unchanged.

Added eight independently generated composite UI crops per source frame to v1's synthetic recipe. Each new crop can span several rounded controls/panels and partial text, unlike the isolated-control examples in v1. Dedicated random streams preserve every original positive/negative sample, covered by regression tests. No audit pixels enter training. Source remains the glyph-font-diversity synthetic manifest SHA256 `312d1014b3046c7380293e90f4d7b84fc04a980fab14b75dcf0f162a765586bb`.

Training: seed 20261112, 20 epochs, fixed threshold 0.5, checkpoint selected from validation balanced accuracy only (epoch 10). Architecture unchanged. Failed checkpoint retained locally in ignored `apc/runs/card-face-verifier-composite-v2/weights.pt`, SHA256 `c8e4623adbaa4394364f4cb9e42186335cab350186d8086293cea39cfb7c97a5`; it is not a release checkpoint.

Synthetic test: 420/420 card crops retained, 773/774 negatives rejected. On the SAME expanded synthetic test, v1 rejects 747/774 negatives. Full per-kind comparison: `card-face-verifier-composite-v2.patch-comparison.json`. This diagnostic does not select checkpoints or tune thresholds.

Captured reference comparisons contradict the favorable synthetic result:

| Reused reference | v1 exact / targets | Composite v2 exact / targets | v2 false proposals |
| --- | ---: | ---: | ---: |
| Audit-v4 | 27/27 | 11/27 | 0 |
| Viewport-v3 | 34/34 | 11/34 | 0 |
| Older development | 34/34 | 14/34 | 0 |
| Older synthetic retention | 80/80 | 80/80 | 0 |

The verifier rejects many actual captured card faces, not just distractors. Oracle glyph checks remain correct, so this regression is filtering, not recognition. Example first river scores: 7h 0.822, Qh 0.422, Ac 0.095, 9d 0.145, 3d 0.052, 6s 0.276, 4h 0.063. Do not lower the threshold to fit these reused labels and call the result independently validated.

One measurable coverage difference: source cards are 56x78 while standard browser cards are 62x78 and compact cards 44x64. A geometry/appearance shortcut is a hypothesis, not a proven internal explanation. Next ablation adds complete-card aspect/scale variants while preserving original samples and the composite negatives. New results must pass captured true-card retention, not just synthetic validation or false-positive rejection.

Reproduce with `python -m apc.perception.train_card_verifier apc/runs/glyph-font-diversity-v1/dataset_manifest.json NEW_OUTPUT --epochs 20 --negative-scenes 8`. Later generator versions retain this behavior when positive-aspect variants are zero. All saved captured comparisons are reused references, not new audits. No coaching-readiness or default promotion is justified.
