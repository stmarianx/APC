# Supplemental surface proposals for missing neural peaks

Inspection of the three reused v6 misses found partial Qc proposals (best IoU ~0.483) and no overlapping final proposal for compact 2h/As. Widening an existing box cannot recover a card when the surviving proposal set does not overlap it. This motivated a separate image-only candidate source, not a card-count or position-specific rule.

`surface_rescue.py` scans downscaled full-image bright near-neutral components using the established light-card baseline's broad shape/size/fill limits and vectorized four-connectivity. It supplements, rather than replaces, neural proposals. Existing neural scores retain precedence during geometric deduplication; every added candidate still passes learned text-v5 verification and neural rank/suit recognition. No target labels, known seat positions, expected card counts or hand history enter proposal generation. Light surfaces only; dark/occluded cards remain a limitation.

Opt-in stack: wide v2 + fast components/refinement + shared text-v5 verifier checkpoint, now with `--surface-rescue`. No neural weights, thresholds or application defaults changed.

| Reused reference/development set | Exact | False proposals | False ink identities |
| --- | ---: | ---: | ---: |
| v6 audit images, post-audit reuse | 60/60 (previously 57/60) | 0 | 0 |
| Older audit-v4 | 27/27 | 0 | 0 |
| Older viewport-v3 | 34/34 | 0 | 0 |
| Synthetic compact sweep | 96/96 | 4 | 0 |

All confirmed identities remain null and hero/board roles unresolved. These are reused same-application references/development examples, not an independent new frozen audit. The original sealed 57/60 result is preserved. Zero false proposals here does not establish arbitrary-table readiness. Four non-ink false proposals remain in synthetic controls. Additional processing cost has not yet been measured in a matched timing run.

Unit tests cover distinct adjacent surfaces, dark-scene abstention, no identity outputs, geometric deduplication and neural precedence without mutating input proposals. Reports bind collection, checkpoint and candidate sources; no audit frame enters training.

Reproduce into a new file:

```powershell
.\.venv\Scripts\python -m apc.perception.bright_surface_candidate apc/data/reference/browser-fast-text-audit-v6/collection.json --wide-search --fast-components --fast-refinement --surface-rescue --verifier-checkpoint apc/runs/card-face-verifier-text-v5/weights.pt --bounded-verifier-checkpoint apc/runs/card-face-verifier-text-v5/weights.pt --output apc/runs/new-surface-rescue-reference.json
```

Next: matched timing and harder appearance retention, freeze complete candidate including rescue/helper source fingerprints, then acquire genuinely new varied-scene evidence. Full-table state, temporal/profile integration, solver/neural strategy and evaluated learning still remain unfinished.
