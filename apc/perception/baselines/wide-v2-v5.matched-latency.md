# Matched warm reader timing — latency gate remains open

Three readers, same nine corrected-label v5 reference images, two Torch CPU threads, three passes, 27 measured reads per reader. One untimed first-frame warmup each. Variant order rotates across frames and repeats. No other training or evaluation job ran concurrently. Model/source edits and file inspections occurred but no deliberate background load was introduced; shared laptop conditions are not controlled lab conditions.

| Reader | Median ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: |
| Incumbent routed v1/v4 | 671.59 | 1148.25 | 1264.86 |
| Wide surface v2 + v4 verifier | 1074.79 | 1411.87 | 1420.21 |
| Wide surface v2 + v5 verifier | 1006.45 | 1394.69 | 1404.86 |

Per-frame card-output fingerprints repeat exactly within every variant. This checks consistency across timing passes, NOT identical predictions between variants or correctness/readiness. The new v5 stack improves reused-reference recall/false detections but is slower than the incumbent, primarily associated with the added wide recovery path; this comparison does not isolate internal stage cost or prove the verifier alone is faster.

Includes warm saved-image I/O, proposals, refinement, verification and glyph recognition. Excludes model initialization, screen capture, temporal state, solver/strategy, profiles and UI. These are not full-coach latency results. Even perception-only p95 is above the project's 250ms target. Do not claim real-time readiness from fast policy-network inference alone.

Reproduce:

```powershell
.\.venv\Scripts\python -m apc.perception.benchmark_reader_candidates apc/data/reference/browser-routed-audit-v5/collection.geometry-v2.json --new-verifier apc/runs/card-face-verifier-text-v5/weights.pt --output apc/runs/new-matched-reader-timing.json
```

The output must be new. Machine-readable report retains collection/checkpoint hashes and all 81 observations.

Frozen-audit evaluator now supports a boolean `wide_surface_recovery` plan field, requiring the verifier, adaptive surfaces and a frozen `bright_surface_candidate.py` hash. Plans can also require capture-stability evidence; legacy collections cannot silently satisfy that requirement. No new capture or fresh audit was performed in this timing run.

Next: profile repeated component searches and test shared image-level preprocessing/caching without changing output fingerprints. Then freeze the chosen implementation and run new varied-scene captures. Accuracy, calibration, full-table integration and end-to-end deadline gates remain open; defaults unchanged.
