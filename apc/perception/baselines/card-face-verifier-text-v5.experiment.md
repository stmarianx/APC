# Text-fragment verifier v5 — trained research candidate

Added twelve generated narrow text-panel subcrops per synthetic source frame, with an independent RNG stream (seed + 40,000,000). Text includes stack/player/action/pot labels, varied fonts/colors and partial words. No playing cards appear in these negative panels, and no captured audit image is read by the training loader. New examples do not perturb prior samples; a regression compares old tensors/labels after excluding the added kind.

Architecture unchanged: RGB card-face convolutional binary verifier, fixed threshold 0.5. Synthetic source sessions remain split-exclusive. Generated styles/fonts are shared across splits, so these scores are not real-domain generalization evidence. Train/validation/test counts: 17264/2062/2034 patches. Seed 20261112, 20 epochs, selected epoch 14 by validation balanced accuracy only. Synthetic test: 923/924 positives retained, 1108/1110 negatives rejected. No threshold or checkpoint selection used captured results.

Checkpoint: `apc/runs/card-face-verifier-text-v5/weights.pt` (local ignored artifact), SHA-256 `41e45f326beeb935f218f33c4ba31d268e30f7d8f2bbaa41ef9ffd6020a7d9fc`. Full history, recipe and font fingerprints: `card-face-verifier-text-v5.training.json`.

Reference evaluation uses opt-in wide surface recovery v2 and replaces ONLY the ambiguous-surface appearance-v4 verifier with v5. Bounded surfaces still use incumbent v1, and all other weights/options stay unchanged:

| Reused development/reference set | Exact | False proposals | False ink identities |
| --- | ---: | ---: | ---: |
| Corrected-label browser v5 | 27/27 | 0 (previously 2) | 0 (previously 2) |
| Older viewport-v3 | 34/34 | 0 | 0 |
| Compact synthetic sweep | 96/96 | 8 | 0 |

The eight synthetic false proposals persist; replacing only one expert does not eliminate all control surfaces. All confirmed identities remain null. Captured reference labels remain developer draft corrections, not independent geometry approval. These reused cases informed research direction; zero false detections here is not a fresh audit pass. No default promotion or coaching activation; calibration, broader retention, timing and fresh varied-scene evaluation remain open.

Reproduce training into a NEW directory:

```powershell
.\.venv\Scripts\python -m apc.perception.train_card_verifier apc/runs/glyph-font-diversity-v1/dataset_manifest.json NEW_OUTPUT --epochs 20 --negative-scenes 8 --positive-aspect-variants 2 --generated-positive-faces 8 --positive-font C:/Windows/Fonts/segoeuib.ttf --positive-font C:/Windows/Fonts/segoeui.ttf --positive-font C:/Windows/Fonts/seguisym.ttf --positive-font C:/Windows/Fonts/arialbd.ttf --text-fragments 12
.\.venv\Scripts\python -m apc.perception.bright_surface_candidate apc/data/reference/browser-routed-audit-v5/collection.geometry-v2.json --wide-search --verifier-checkpoint NEW_OUTPUT/weights.pt --output apc/runs/new-v5-reference.json
```

The baseline checkpoints remain unchanged. Per-kind comparison CLI now supports `--text-fragments`; old options default to zero extra fragments. No training process remains running after this experiment.
