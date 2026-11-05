# Opt-in wider bounded surface recovery v2

Synthetic failure inspection found a narrow proposal on positive-001: about x=225..242 while the full card spans x=223..267. Existing expansion 1.4 stops at about x=266, so the component touches the ROI edge and is correctly rejected as incomplete. The bright-seed v1 experiment changes the seed but not search bounds, explaining why that example remains unrecovered. This is direct evidence for that development example, not proof all failures share one cause.

V2 opt-in `--wide-search` adds expansion 2.8 after the existing .35/.7/1.4 sequence. It retains existing aspect, fill, area and overlap checks, rejects clipped components, leaves already bounded proposals unchanged and still runs routed neural verification. No card identity, known seat location or target count enters recovery. Default reader and narrow-search v1 behavior remain unchanged; no weights were retrained.

| Evaluation scope | Exact cards | Misses | False proposals | False ink identities |
| --- | ---: | ---: | ---: | ---: |
| Segoe compact development sweep, baseline | 88/96 | 8 | 8 | 0 |
| Same sweep, wide v2 | 96/96 | 0 | 8 | 0 |
| New Arial Bold / seed 20261117 sweep, wide v2 | 96/96 | 0 | 8 | 0 |
| Reused viewport-v3 reference, wide v2 | 34/34 | 0 | 0 | 0 |
| Reused corrected-label audit-v5 reference, wide v2 | 27/27 | 0 | 2 | 2 |

The Arial sweep changes font and sampled identities but uses the same placement grid and renderer, so it is a correlated development check, not an independent geometry/cross-platform audit. V5 is post-review reference reuse with draft geometry, not a new frozen audit. All confirmed identities remain null; the two ink-bearing negative-fixture false detections persist. There is no calibrated certainty, role validation, latency benefit or coaching-readiness claim. Wider searches may increase work and must be benchmarked end to end.

Source/checkpoint/collection fingerprints are recorded in each candidate result. Reports: `compact-geometry-sweep-v1.bright-wide-v2-reference.json`, `bright-wide-v2.arial-sweep-reference.json`, `bright-wide-v2.viewport-v3-reference.json`, and `bright-wide-v2.audit-v5-reference.json`.

Reproduction (new output files required):

```powershell
.\.venv\Scripts\python -m apc.perception.bright_surface_candidate apc/runs/compact-geometry-sweep-v1/collection.json --wide-search --output apc/runs/new-wide-result.json
.\.venv\Scripts\python -m apc.synthetic.render_compact_sweep apc/runs/new-font-sweep --font C:/Windows/Fonts/arialbd.ttf --seed 20261117
.\.venv\Scripts\python -m apc.perception.bright_surface_candidate apc/runs/new-font-sweep/collection.json --wide-search --output apc/runs/new-font-wide-result.json
```

Next: harder synthetic glyph-like control negatives, retention on other reference collections, matched timing, then freeze this wrapper plus all incumbent sources/weights before a fresh varied-scene audit. Do not train on captured audit frames or promote based solely on reused results.
