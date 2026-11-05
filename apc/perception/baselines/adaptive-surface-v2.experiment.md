# Bounded-surface refinement v2

Date: 2026-11-05. Research-only, explicit opt-in. Neural weights and inference defaults unchanged.

## Diagnosis

The v1 bounded refinement recovered compact cards but rejected two wide-view surfaces. `adaptive-surface-v1.crop-diagnostic.json` records the paired raw/legacy/adaptive proposals and identities before this fix. In the wide frame the affected 2h and Ah proposals were oversized. Full bounded card components occupied less than 70% of each original proposal, so the original-area containment requirement rejected them. The fallback returned raw boxes, which did not yield usable glyph bands. Legacy refinement recovered both cards with IoU above 0.97.

The fix compares intersection against the smaller of proposal area and candidate area. It therefore accepts a bounded card mostly contained within an oversized proposal as well as a bounded card containing an undersized proposal. Existing ROI-boundary, aspect, fill, relative-area and expansion limits remain. The choice uses only pixels/box geometry, never labels, identities, confidence scores or a known table layout. Version: `bounded_smaller_area_overlap_v2`.

## Evidence

Same incumbent detector and font-diversity recognizer hashes as `locator-axis-diversity-v1.experiment.md`; no neural training in this experiment.

| Reused viewport reference | Legacy | Adaptive v1 | Adaptive v2 |
| --- | ---: | ---: | ---: |
| Four 960-wide frames | 20/20 | 20/20 | 20/20 |
| Compact frame | 2/7 | 7/7 | 7/7 |
| Wide frame | 5/7 | 3/7 | 5/7 |
| Total exact identities | 27/34 | 30/34 | 32/34 |
| False proposals | 3 | 3 | 3 |
| Missed cards | 2 | 2 | 2 |

Captured development retention: 34/34, zero false proposals/misses. Earlier audit-v2 captures, reused as a reference: 34/34, zero false proposals/misses. Earlier mixed-surface synthetic retention: 80/80, zero false proposals/misses. No confirmed identities emitted. Paired post-fix crops are recorded in `adaptive-surface-v2.crop-diagnostic.json`; aggregate reports use the same prefix.

These are reused, correlated reference/development images, not fresh independent validation. No captured reference/audit frames enter training. Independent geometry review remains pending. Candidate selection has now used this collection, so a new frozen-model capture session is necessary for a generalization claim.

## Reproduction and next work

Use `python -m apc.perception.diagnose_surface_refinement COLLECTION apc/checkpoints/glyph-font-diversity-v1.pt apc/checkpoints/card-locator-hardneg-v1.pt --output REPORT.json` for paired crops. The diagnostic can access labels only after inference, for matching and scoring. It is not an operational reader or a training-data exporter.

Aggregate evaluation: `python -m apc.perception.evaluate_browser_collection COLLECTION --surface-pairs --adaptive-surfaces --glyph-checkpoint apc/checkpoints/glyph-font-diversity-v1.pt --locator-checkpoint apc/checkpoints/card-locator-hardneg-v1.pt --output REPORT.json`.

Synthetic retention uses `python -m apc.perception.evaluate_card_reader apc/checkpoints/glyph-font-diversity-v1.pt apc/runs/mixed-surface-locator-v1/dataset_manifest.json --corner-symbols --surface-pairs --locator-checkpoint apc/checkpoints/card-locator-hardneg-v1.pt --refine-surfaces --adaptive-surfaces --output REPORT.json`.

Remain opt-in: false detections and missed wide-view hero cards are not solved. Next work is detector recall/distractor robustness with retained original-scale synthetic examples, followed by a fresh preregistered capture audit. Full-table state, player profiles and neural strategy integration/evaluation are still unfinished.
