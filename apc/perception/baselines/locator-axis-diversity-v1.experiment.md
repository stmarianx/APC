# Scale and bounded-surface experiments

Date: 2026-11-05. Research only; no default model promotion or readiness credit.

## Frozen inputs

- Recognizer: `apc/checkpoints/glyph-font-diversity-v1.pt`, SHA256 `8e2378365f47412a89d5c2f88d26deb3f716d387649be5a82b7d473703c1261a`.
- Incumbent detector: `apc/checkpoints/card-locator-hardneg-v1.pt`, SHA256 `8168a0607d03bf0d3e22bb5679c6da82dec8fbae5eeba8434f514d5575fd1273`.
- Candidate detector: `apc/runs/locator-axis-diversity-v1/weights.pt`, SHA256 `0a952e843d661df4577017b7156c5ca4e8ea2758aa6b4e7c4e101fd2254b3d01`. Failed candidate retained locally in ignored runs, not distributed as a release checkpoint.
- Synthetic training manifest: `apc/runs/browser-symbol-mixture-v1/dataset_manifest.json`, SHA256 `800ed7563c88fc4cc45393c947f2e8f7370d35758df0c920416d81e6707bc340`.
- Viewport collection: `apc/data/reference/browser-viewport-audit-v3/collection.json`, SHA256 `3b161979441f461632c7d792f5193680092a43409b11698a965c161c6400f16f`.

The saved viewport collection has now informed experiment design. All comparisons below reuse it as a reference, NOT a fresh independent audit. It remains excluded from training. Labels are draft geometry; six frames from one session/three hands are correlated, not independent evidence. No original frozen-audit plan or result was changed.

## Reproduction

Run from repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python -m apc.perception.train_card_locator apc/runs/browser-symbol-mixture-v1/dataset_manifest.json apc/runs/locator-axis-diversity-v1 --epochs 50 --relative-size --spatial-augmentation --hard-negative-ratio 3 --scale-min 0.55 --scale-max 1.8 --independent-axes
```

Use a new output directory if the run already exists. Seed 42, CPU, two Torch threads. Checkpoint selected using original validation frames plus fixed broad-scale validation frames, never test scores. Full history and test results are in `locator-axis-diversity-v1.training.json`.

To reproduce the viewport comparison, use `python -m apc.perception.evaluate_browser_collection apc/data/reference/browser-viewport-audit-v3/collection.json --surface-pairs --glyph-checkpoint apc/checkpoints/glyph-font-diversity-v1.pt --locator-checkpoint DETECTOR`, optionally adding `--adaptive-surfaces` and `--output REPORT.json`. The adaptive mode rejects ROI-edge-clipped components and expands by 0.35, 0.7, then 1.4 proposal widths/heights. It never uses target labels or predicted identities to refine a box. Default inference remains unchanged.

## Results

Viewport reference, 34 visible card appearances:

| Detector | Refinement | Exact identities | False proposals | Missed cards |
| --- | --- | ---: | ---: | ---: |
| Incumbent | Legacy | 27/34 | 3 | 2 |
| Incumbent | Adaptive bounded | 30/34 | 3 | 2 |
| Independent-axis candidate | Legacy | 28/34 | 39 | 6 |
| Independent-axis candidate | Adaptive bounded | 26/34 | 41 | 8 |

The incumbent/adaptive combination changes compact recognition from 2/7 to 7/7, but wide recognition falls from 5/7 to 3/7. Its other four frames stay at 20/20. Oracle glyph recognition remains 34/34 in every comparison; crop/detection errors dominate.

Earlier captured development reference (34 appearances): incumbent/adaptive 34/34 with zero false proposals; candidate/legacy 34/34 but 27 false proposals. Earlier mixed-brightness synthetic test: incumbent/adaptive 80/80 with zero false proposals, zero misses, zero confirmed identities. Its observed p95 was approximately 159 ms, diagnostic only, not a controlled latency benchmark.

Candidate raw-box synthetic tests: original 82/100 localized, zero false proposals; broad independent-axis fixed transforms 75/100, 11 false proposals; matched legacy fixed transforms 91/100, one false proposal. These are localization metrics, not identity or full-table coaching accuracy.

## Decision and next work

Do not promote either change. Broad augmentation increases false detections dramatically on captured frames despite favorable synthetic validation. Adaptive surface expansion is opt-in research because its aggregate gain hides a wide-layout regression. Keep model confidence uncalibrated, card identities unconfirmed, and all coaching-readiness gates closed.

Next: inspect adaptive wide-frame crop changes, add synthetic cases covering clipped/adjacent surfaces and responsive UI distractors, and evaluate a more targeted detector-training distribution with retained original-scale examples. Once a candidate is frozen, collect a NEW preregistered multi-viewport capture session before claiming generalization. Full-table perception, state/profile integration and neural strategy evaluation remain separate unfinished requirements.
