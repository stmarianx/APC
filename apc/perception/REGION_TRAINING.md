# APC spatial recognition baseline

The spatial target adapter converts verified frame annotations into normalized
XYXY regions and masked attributes. Names/stacks/dealer indicators are seat
attributes, not invented independently localized boxes. Uncertain identity and
numeric annotations are masked; fully occluded regions are excluded. Frozen
audit/reference annotations are rejected. The manifest loader verifies image
hashes and preserves capture-session splits.

The region network consumes proposed boxes. It is **not a detector**, and neither
its accuracy nor its network-only latency establishes full-table readiness.
It predicts region class, visibility, card rank/suit, action type, dealer/hero
indicators and button availability. It does not yet learn numeric or name OCR.

Reproduce the first run from the project root (CPU PyTorch, NumPy, Pillow):

```powershell
.\.venv\Scripts\python.exe -m apc.synthetic.render_table apc/runs/region-dataset-v1 --sessions 9 --seed 20261005 --include-turn-clock --include-name-ocr
.\.venv\Scripts\python.exe -m apc.perception.train_regions apc/runs/region-dataset-v1/dataset_manifest.json apc/runs/region-recognizer-v1 --epochs 10 --seed 42
```

Use fresh output directories when repeating. Training selects a checkpoint using
validation only, then evaluates test once. Weights and evaluation JSON remain in
the ignored run directory. Seed, manifest hash, runtime version, validation
history, supervised counts, and latency scope are recorded in evaluation.json.

## First result (2026-10-05)

36 generated frames, nine sessions, three layouts, two themes. These are
synthetic frames, not controlled-visible readiness evidence. Held-out test:
four frames, 50 non-occluded regions.

| Head | Correct / supervised | Accuracy |
| --- | --- | --- |
| Region class | 30 / 50 | 60% |
| Visibility | 50 / 50 | 100% |
| Card rank | 0 / 20 | 0% |
| Card suit | 6 / 20 | 30% |
| Action | 2 / 10 | 20% |
| Dealer | 4 / 8 | 50% |
| Hero | 4 / 8 | 50% |
| Enabled | 10 / 10 | 100% |

Network-only p95: 16.405 ms, with ground-truth regions supplied. Constant/easy
visibility and enabled labels must not obscure failed rank/action recognition.
This checkpoint is rejected for coaching use. Next work: improve spatial glyph
features and data coverage, evaluate independently proposed boxes, integrate
numeric/name recognition, and collect verified controlled-visible sessions.
