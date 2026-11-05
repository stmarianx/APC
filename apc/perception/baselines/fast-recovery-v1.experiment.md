# Exact-output vectorized recovery components

Profiled one wide empty reference frame on the v5 wide reader: 1.691s total under profiler, including 1.231s cumulative in 105 connected-component scans. This is a diagnostic profile, not an uninstrumented latency benchmark. Per-row numpy padding/diff/index extraction accounts for substantial work.

New `fast_components.py` batches row-run boundary extraction across the mask, retaining four-connectivity, union ordering, component bounds and pixel area. Tests compare exact ordered tuples against the incumbent on 204 deterministic random/edge masks, plus identical full-card recovery. No new dependency or threshold change. The optimization is opt-in via `BrightSurfaceReader(..., fast_components=True)` and affects only the added bright-surface recovery path; incumbent neural-locator refinement remains unchanged.

Matched v5 wide-reader comparison: same corrected-label nine-frame reference, two CPU threads, 27 reads/variant, rotating order and one first-frame warmup. No training/evaluation ran concurrently.

| Variant | Median ms | p95 ms | Maximum ms |
| --- | ---: | ---: | ---: |
| Original recovery scanning | 963.77 | 1360.48 | 1416.82 |
| Batched run extraction | 743.56 | 1167.62 | 1188.76 |

Approximately 23% lower median and 14% lower p95 in this small run. Every card-output fingerprint matches BOTH across repeats and between variants. This preserves candidates/status/geometry, not a claim of full-table correctness. Results include saved-frame I/O/proposals/verification/glyphs, excluding loading/capture/state/strategy/UI. Still well above the 250ms perception target; shared laptop timing is not a production gate. No weights/defaults promoted and no coaching readiness.

Reproduce:

```powershell
.\.venv\Scripts\python -m apc.perception.benchmark_reader_candidates apc/data/reference/browser-routed-audit-v5/collection.geometry-v2.json --new-verifier apc/runs/card-face-verifier-text-v5/weights.pt --compare-fast --output apc/runs/new-fast-recovery-timing.json
```

Machine-readable report binds checkpoints, candidate/helper sources and collection and retains all 54 observations. Next: extend equivalence/retention coverage before integrating faster scanning into more refinement stages, then freeze the chosen implementation before fresh varied-scene validation. Audit images remain excluded from training; no source-image cache can substitute stale frames.
