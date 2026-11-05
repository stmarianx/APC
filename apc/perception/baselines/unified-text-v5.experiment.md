# Shared text-v5 checkpoint on both surface routes

Controlled post-audit experiment: use the already-trained text-fragment v5 checkpoint for BOTH ambiguous and bounded proposals. All glyph/locator weights, geometry, wide/fast options and fixed verifier thresholds remain unchanged. No new training or audit-image supervision. This tests routing/expert replacement, not a new neural architecture.

| Reused reference | Exact cards | False proposals | False ink identities |
| --- | ---: | ---: | ---: |
| Fresh-audit-v6 images, post-audit reuse | 57/60 | 0 (routed incumbent: 3) | 0 |
| Older audit-v4 | 27/27 | 0 | 0 |
| Older viewport-v3 | 34/34 | 0 | 0 |
| Compact synthetic development sweep | 96/96 | 4 (routed incumbent: 8) | 0 |

No extra card loss was observed on these sets. In particular the older compact-card retention check now passes with v5 on the bounded route, unlike the historical appearance-v4-only experiment. Four synthetic false proposals still survive; all have no glyph candidate. Three v6 localization misses remain unchanged. Zero confirmed identities; no readiness, calibration or default promotion.

Both expert hashes are `41e45f326beeb935f218f33c4ba31d268e30f7d8f2bbaa41ef9ffd6020a7d9fc`. Result files record each hash explicitly. Historical metadata route text `bounded_surface_v1_else_ambiguous_v4` was hardcoded and does NOT describe overrides; checkpoint hashes are authoritative. The research reader now reports neutral route semantics and whether the expert hashes are shared. Counts in the preserved reports precede that metadata-only fix and remain unchanged.

Reproduce with explicit selection:

```powershell
.\.venv\Scripts\python -m apc.perception.bright_surface_candidate apc/data/reference/browser-fast-text-audit-v6/collection.json --wide-search --fast-components --fast-refinement --verifier-checkpoint apc/runs/card-face-verifier-text-v5/weights.pt --bounded-verifier-checkpoint apc/runs/card-face-verifier-text-v5/weights.pt --output apc/runs/new-unified-reference.json
```

Current implementation still instantiates/routes two experts with identical weights, so no batching/single-instance speed benefit is claimed. This is same-image reference development, not an independent fresh audit. Next: prioritize the missing-card localization failures, then freeze the selected route configuration and verify new captures. Continue to exclude all audit images from training and preserve the prior frozen result.
