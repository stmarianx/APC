# Opt-in adaptive refinement acceleration

`fast_refinement.py` provides a versioned adaptive-refinement locator path using batched row-run extraction. It reuses loaded neural weights and preserves view preprocessing, decoding, ROI expansions, component criteria/tie order, added-view policy and deduplication. No global monkeypatch, result cache, hidden labels or stale-image reuse. Existing frozen inference files remain unchanged. The explicit adaptive algorithm is duplicated here to preserve prior frozen source contracts; equivalence tests must accompany future edits to either implementation.

Verification: exact adaptive-refinement dictionaries on 100 seeded proposal geometries; whole locator output equality (excluding elapsed time) across tiled/non-tiled and deduplicated/non-deduplicated configurations; underlying component equality on 204 seeded random/edge masks.

Matched reused-browser timing, same nine corrected-label v5 frames, two CPU threads, 27 reads/variant, rotating order:

| Variant | Median ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: |
| Original v5 wide reader | 1003.69 | 1456.08 | 1593.29 |
| Fast recovery only | 769.22 | 1226.14 | 1347.38 |
| Fast recovery + locator refinement | 455.27 | 517.10 | 550.16 |

Full acceleration reduces median ~55% and p95 ~64% versus the original in this run. All card-output fingerprints match exactly across variants and repeats. No concurrent training/evaluation ran during timing. Same saved-frame scope as earlier benchmarks: includes I/O/proposals/refinement/verification/glyphs; excludes loading/capture/state/strategy/UI. Shared laptop measurements and reused frames are not a production deadline gate. Perception p95 still exceeds 250ms.

Retention: compact synthetic sweep remains 96/96 exact with eight non-ink false proposals; older viewport reference remains 34/34 with zero false proposals. Nine-frame browser outputs are identical to the original v5 stack, including unresolved roles and null confirmed identities. No readiness/default promotion or new independent accuracy audit.

CLI evaluation flags: `--wide-search --fast-components --fast-refinement --verifier-checkpoint apc/runs/card-face-verifier-text-v5/weights.pt`. Evaluation records helper source hashes and explicit flags. Frozen audit plans now support `fast_surface_components` and `fast_locator_refinement`, requiring explicit wide recovery and matching helper source hashes, with capture-stability requirements separately enforced.

Reproduce timing into a NEW file:

```powershell
.\.venv\Scripts\python -m apc.perception.benchmark_reader_candidates apc/data/reference/browser-routed-audit-v5/collection.geometry-v2.json --new-verifier apc/runs/card-face-verifier-text-v5/weights.pt --compare-full-fast --output apc/runs/new-full-fast-timing.json
```

Next: freeze the complete candidate and capture a fresh varied-scene audit with stability-bound geometry, including negative controls and complete-hand sequences. Optimize remaining mask reductions only with equivalence and latency evidence; do not substitute policy-network timing for full coaching latency. Training/role/full-table/profile/strategy integration gates remain open.
