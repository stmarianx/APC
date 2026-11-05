# Overlapping-view detector candidate

2026-11-05. Explicit research option only. No training, changed neural weights, readiness credit or default promotion.

## Evidence and design

The wide reference frame's two hero-card target cells scored about 0.033 and 0.051 in the incumbent detector; neither had any above-threshold dense proposal matching its box. These are low-objectness misses, not recoverable by replacing grid suppression alone. In the compact frame, surface refinement merges two original proposals onto the same 9s card. The other two false proposals do not find bounded card surfaces and have scores near 0.505 and 0.506.

The candidate adds four uniform overlapping two-thirds-width/two-thirds-height windows at the corners, alongside the original full-frame view. Each is resized to the same trained input size, processed as one batch, and mapped back to full-image coordinates. Windows do not use labels, roles, identities, card count, or a table layout. Additional tiled-view proposals must recover a complete bounded surface in original-image pixels. Original-view proposals are retained even if surface evidence is absent, avoiding an undocumented global appearance filter. Geometry-only IoU >= 0.9 suppression keeps the highest-scoring duplicate, independent of predicted identities.

An unrestricted five-view diagnostic recovered all 34 identities but produced 118 false proposals. It is rejected. That historical result is retained in `tiled-surfaces-v1.viewport-reference.json`; the implemented added-view policy is `bounded_surfaces_only`.

## Reused-reference results

Same frozen weights as prior experiments: recognizer SHA256 `8e2378365f47412a89d5c2f88d26deb3f716d387649be5a82b7d473703c1261a`, detector SHA256 `8168a0607d03bf0d3e22bb5679c6da82dec8fbae5eeba8434f514d5575fd1273`.

| Mode on viewport reference | Exact | False proposals | Missed |
| --- | ---: | ---: | ---: |
| Legacy | 27/34 | 3 | 2 |
| Adaptive v2 | 32/34 | 3 | 2 |
| Adaptive v2 + deduplication | 32/34 | 2 | 2 |
| Adaptive v2 + deduplication + bounded tiled additions | 34/34 | 2 | 0 |

All seven wide-frame cards now match; compact remains 7/7 with the two original-view false proposals. Older development and reused audit-v2 references both remain 34/34, zero false proposals/misses. Older mixed-surface synthetic retention remains 80/80, zero false proposals/misses. No confirmed identities are emitted and hero/board roles remain unresolved.

Synthetic retention's observed p95 was about 305 ms; a separate cold saved-wide-frame CLI inspection took about 435 ms, excluding process/checkpoint loading. These are diagnostic observations with different thread settings, not a matched benchmark or end-to-end coaching latency gate. Five-view processing costs more than one view.

54 targeted tests passed, including full-frame coordinate mapping, rejection of unsupported tile-only proposals, geometry-only duplicate suppression, argument validation, explicit model selection, desktop inspection compatibility, detector/reader inference, and prior frozen audit checks.

All captured frames were reused during diagnosis/selection; they are not fresh independent validation. Their annotations remain draft, their observations correlated, and none enters training. No cross-platform or full-table readiness claim is supported.

## Run the candidate

```powershell
.\.venv\Scripts\python -m apc inspect FRAME.png --neural-cards --adaptive-glyphs --adaptive-surfaces --deduplicate-surfaces --tiled-views --glyph-checkpoint apc/checkpoints/glyph-font-diversity-v1.pt --locator-checkpoint apc/checkpoints/card-locator-hardneg-v1.pt
```

For aggregate comparisons, use `python -m apc.perception.evaluate_browser_collection COLLECTION --surface-pairs --adaptive-surfaces --deduplicate-surfaces --tiled-views --glyph-checkpoint apc/checkpoints/glyph-font-diversity-v1.pt --locator-checkpoint apc/checkpoints/card-locator-hardneg-v1.pt --output REPORT.json`. For the deduplication-only control omit `--tiled-views`.

## Next

Freeze this complete inference configuration for a new preregistered multi-viewport capture audit; include blank/transition frames and distractors so false proposals cannot hide behind recall gains. Independently review geometry. Measure matched warm/cold latency on captured frames. Then decide whether training changes or a learned proposal verifier are needed. Full-table perception, temporal state, profiles, solver/neural strategy and learning-loop integration remain unfinished.
