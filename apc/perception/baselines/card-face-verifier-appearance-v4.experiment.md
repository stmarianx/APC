# Appearance-diverse verifier v4 and geometry-routed candidate

2026-11-05. Research components only, not the completed APC coach. No default promotion.

## Controlled experiments

V2 added composite controls and rejected many captured cards. V3 added positive aspect variants and partly restored recall. V4 adds eight independently rendered positive card faces per source frame, with validated fonts, left/centered symbols, variable face/border colors and sizes. Separate random streams preserve all earlier samples. See the adjacent v2/v3 experiment notes for the rejected alternatives; their failed weights remain local ignored runs, not release checkpoints.

V4 uses the same binary convolutional architecture and fixed threshold 0.5. Seed 20261112, 20 epochs; selected epoch 19 from synthetic validation balanced accuracy, not captured/test scores. Training/validation/test: 14480/1726/1698 patches. Synthetic test retains 924/924 positive faces and rejects 773/774 negatives. Captured audit images are never training samples. Original source sessions stay exclusive; generated styles and fonts are shared across splits and do not establish real-domain generalization.

Reproducible recipe and hashes: `card-face-verifier-appearance-v4.dataset-recipe.json`; full history/font fingerprints: `.training.json`. Checkpoint `apc/checkpoints/card-face-verifier-appearance-v4.pt`, SHA256 `59b3cb7bf2d8c6b262fc59a73a6e00d8327ef72f71e2136f00c5aa6520c65aa3`.

```powershell
.\.venv\Scripts\python -m apc.perception.train_card_verifier apc/runs/glyph-font-diversity-v1/dataset_manifest.json NEW_OUTPUT --epochs 20 --negative-scenes 8 --positive-aspect-variants 2 --generated-positive-faces 8 --positive-font C:/Windows/Fonts/segoeuib.ttf --positive-font C:/Windows/Fonts/segoeui.ttf --positive-font C:/Windows/Fonts/seguisym.ttf --positive-font C:/Windows/Fonts/arialbd.ttf
```

Font files are fingerprinted, not copied. They are needed only to reproduce training, not to run trained weights.

## Reused captured evidence and expert routing

V4 alone still rejects one compact Qh (verifier score about 0.012) despite bounded surface evidence. V1 retains it. V1's three retained distractors lack bounded surface evidence. The explicit routed candidate therefore uses v1 for `adaptive_bounded_surface` proposals and v4 for all ambiguous/no-bounded-surface proposals. Both thresholds remain 0.5. Routing uses pixel-derived geometry only, never identity, role, expected card count or layout. It preserves proposal order and records expert hashes/thresholds in inspection output.

| Reused reference | V1 only | V4 only | Routed v1/v4 |
| --- | --- | --- | --- |
| Audit-v4 identities | 27/27 | 26/27 | 27/27 |
| Audit-v4 false proposals | 3 | 1 | 1 |
| Audit-v4 false ink-based identities | 1 | 0 | 0 |
| Viewport-v3 identities | 34/34 | 33/34 | 34/34 |
| Viewport-v3 false proposals | 0 | 0 | 0 |
| Earlier development identities | 34/34 | 34/34 | 34/34 |
| Earlier synthetic identities | 80/80 | 80/80 | 80/80 |

The remaining routed false proposal is on the static 960-wide empty fixture and yields no usable glyph identity. All confirmed identities remain null. These images informed experiment design; this is reference reuse, NOT a fresh independent audit. The original frozen v4 audit remains unchanged. Appearance-based surface routing remains limited for dark/occluded/artwork-heavy cards and must be tested beyond this application.

Use the routed candidate explicitly:

```powershell
.\.venv\Scripts\python -m apc inspect FRAME.png --neural-cards --adaptive-glyphs --adaptive-surfaces --deduplicate-surfaces --tiled-views --glyph-checkpoint apc/checkpoints/glyph-font-diversity-v1.pt --locator-checkpoint apc/checkpoints/card-locator-hardneg-v1.pt --verifier-checkpoint apc/checkpoints/card-face-verifier-appearance-v4.pt --bounded-verifier-checkpoint apc/checkpoints/card-face-verifier-v1.pt
```

Aggregate evaluators accept the same two verifier flags. Omit the bounded flag to evaluate v4 alone. No external-table actions or coaching activation are added.

## Verification and next steps

53 targeted regression tests pass, including generator reproducibility, preservation of old examples, font/option validation, audit exclusion, routing/order/provenance and application selection. Matched timing samples are saved separately in `card-face-verifier-routed-v1.matched-latency.json`; they exclude capture, strategy, state tracking and model loading. Concurrent evaluation latency numbers are not overhead benchmarks.

Freeze all four neural checkpoints, inference sources and routing configuration for a NEW multi-session capture audit, including different empty/control/transition scenes and independent geometry review. Do not equate improvements on these reused references with visible-table training or coaching readiness. Full-table state extraction, temporal/profile integration, solver/neural strategy and the learning loop remain unfinished.
