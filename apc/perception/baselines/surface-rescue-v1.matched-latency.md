# Surface-rescue matched timing and next frozen audit

Same 15 v6 frames, shared text-v5 verifier on both routes, wide search and fast recovery/refinement in BOTH variants. Only surface rescue differs. Two Torch threads, three passes, rotating order, first-frame warmup: 45 measured reads per variant. Unique relative frame keys prevent same-basename captures from being conflated.

| Variant | Median ms | p95 ms | Maximum ms |
| --- | ---: | ---: | ---: |
| No supplemental surface proposals | 542.45 | 943.83 | 1168.26 |
| Supplemental proposals enabled | 603.87 | 838.40 | 3108.67 |

Median cost approximately +61ms/+11%. Lower observed p95 is NOT evidence of a speed improvement: median and maximum are worse, and shared-laptop timing is variable. The 3.1s outlier remains in the report; its cause is not established. Includes warm saved-frame I/O and neural/geometry processing, excludes capture/loading/state/strategy/UI. No deadline or 250ms perception pass. Output fingerprints repeat within variants; cross-variant equality is expected to fail because rescue adds the three missing cards.

Reproduce into a new file:

```powershell
.\.venv\Scripts\python -m apc.perception.benchmark_reader_candidates apc/data/reference/browser-fast-text-audit-v6/collection.json --new-verifier apc/runs/card-face-verifier-text-v5/weights.pt --compare-rescue --output apc/runs/new-rescue-timing.json
```

Audit v7 is preregistered in `browser-surface-rescue-audit-v7.plan.json`, collection NOT started. It freezes the shared text-v5 checkpoint on both expert routes, all inference/helper sources including surface_rescue.py, the scoring runner, capture code and fixtures. Fifteen scheduled captures: three four-street hands at widths 920/600/1440 (height 900), then the same three negative designs at those sizes. These widths differ from v6; some appeared in older references, so this is not a new environment claim.

The plan requires source/model fingerprints, capture-stability evidence, complete schedule and prior-image/session disjointness. Any rejected capture is retained and aborts scoring of that schedule; no model-based retakes. Negative designs are reused, not independently novel. All original audit images remain excluded from training. Next: collect after plan commit, review image-bound geometry, score once without tuning. Full-table/role/strategy/calibration/end-to-end timing gates remain open and no defaults are promoted.
