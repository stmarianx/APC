# Compact geometry development sweep and bright-seed experiment

Generator: `apc/synthetic/render_compact_sweep.py`, seed 20261106, explicit Segoe UI Bold font fingerprint recorded in collection.json. It reads no captured images. Forty-eight positive scenes cross viewport widths 560/800, x offsets -12/-4/4/12, card gaps 2/6/12 and y positions 220/540. Each contains two distinct synthetic cards, giving 96 appearances. Twelve negative scenes vary narrow control widths, background and placement. All units are BB; these are geometry development cases, not realistic complete poker hands or independent sessions.

Baseline four-checkpoint routed reader: 88/96 exact, eight misses and eight false proposals; no false ink identities and no confirmed identities. Oracle glyph regions are 96/96 correct. Locator-stage trace: default raw boxes cover 44/96 at IoU >=0.5, refinement recovers 88/96; later added-view policy and deduplication retain all 88. Independent best-IoU coverage is not the one-to-one score, but the final reader evaluation also localizes 88.

Opt-in `bright_surface_candidate.py` tests a specific hypothesis: a mostly-background proposal median may miss an embedded light card, so select a light-pixel seed and retry bounded-component recovery under existing geometry limits. Already bounded proposals remain unchanged; no identity/role labels enter recovery. It is a research wrapper, not a default-reader change. It can recover a controlled unit-test case, but **does not improve this development sweep**: 88/96 exact, eight misses, eight false proposals, zero false ink identities. No promotion; no weights or frozen inference sources changed. This does not establish the cause of the remaining failures or justify broader thresholds.

Reproduce into NEW output destinations:

```powershell
.\.venv\Scripts\python -m apc.synthetic.render_compact_sweep apc/runs/new-compact-sweep --font C:/Windows/Fonts/segoeuib.ttf --seed 20261106
.\.venv\Scripts\python -m apc.perception.bright_surface_candidate apc/runs/new-compact-sweep/collection.json --output apc/runs/new-bright-seed-result.json
```

The retained baseline reader report records the original collection hash and four weights; the candidate report records its source and checkpoint hashes. Generated images remain rebuildable in ignored apc/runs/compact-geometry-sweep-v1. No training occurred; these results do not establish captured-table readiness.

Next: inspect candidate-to-surface overlap and component bounds for the eight synthetic misses, then develop a measured geometry or neural-locator training change. Keep audit frames excluded from training and evaluate retention on other reference sets before a fresh frozen audit. The control set also needs harder glyph-like text negatives to reproduce the earlier ink-bearing captured false positives.
