# Positive-aspect verifier v3 — rejected replacement

2026-11-05. Same binary network and fixed threshold 0.5; no default promotion.

This ablation keeps v2's eight composite UI negatives per source frame and adds two complete-card aspect/scale variants per source card. Independent random streams preserve the original v1 examples and v2 composite negatives. Each extra card is resized on independent horizontal/vertical factors in [0.8, 1.25] before standard verifier encoding. No glyphs are deliberately clipped and no captured audit pixels enter training.

Source manifest and splits are unchanged from v1/v2. Seed 20261112, 20 epochs, checkpoint selected by validation balanced accuracy only: epoch 14. Training split 12624 patches; validation 1502; test 1474. Synthetic test retains 700/700 cards and rejects 774/774 non-cards. These are same-generator test metrics, not captured-domain readiness.

Failed checkpoint stays locally in ignored `apc/runs/card-face-verifier-aspect-v3/weights.pt`, SHA256 `7f2e771189430398771666aadfe9521262b0992fe0471d602713c8067b597873`. Full history and captured comparisons are adjacent `card-face-verifier-aspect-v3.*.json` files.

| Reused captured reference | Incumbent v1 exact | Composite v2 exact | Aspect v3 exact |
| --- | ---: | ---: | ---: |
| Audit-v4 | 27/27 | 11/27 | 21/27 |
| Viewport-v3 | 34/34 | 11/34 | 26/34 |
| Earlier development | 34/34 | 14/34 | 30/34 |

V3 produces zero false proposals in these checks, but still discards real cards. Older synthetic retention stays 80/80. Aspect variation partially closes the captured recall gap but does not solve it. Keep v1 unchanged. No threshold tuning on reused captured labels.

Reproduction: `python -m apc.perception.train_card_verifier apc/runs/glyph-font-diversity-v1/dataset_manifest.json NEW_OUTPUT --epochs 20 --negative-scenes 8 --positive-aspect-variants 2`.

Next controlled ablation adds independently rendered positive faces with varying fonts, symbol alignment, border/face colors and sizes, while preserving all earlier examples. A successful reused-reference candidate still requires NEW capture validation and independent geometry review before any readiness claim.
