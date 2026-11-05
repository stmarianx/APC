# Learned card-face proposal verifier v1

2026-11-05. A new small neural component, not a completed APC strategy/coaching model.

## Training and reproducibility

Three convolutional stages and a small binary head, RGB 48x64 aspect-preserving crop input. Checkpoint `apc/checkpoints/card-face-verifier-v1.pt`, 117381 bytes, SHA256 `392ce66439a75015b13ad8f0f00349ebc523c82c0f62dd66034a95462a675a71`.

Trained for 20 epochs on synthetic-only patches; seed 20261112; fixed 0.5 threshold declared before training. Selected epoch 13 by validation balanced accuracy only. No captured audit images supplied training samples or checkpoint selection. Dataset recipe, source hash, generator/preprocessing hashes and library versions: `card-face-verifier-v1.dataset-recipe.json`. Full history: `card-face-verifier-v1.training.json`.

Source: existing glyph-font-diversity synthetic dataset with its verified session splits unchanged (58 train, seven validation, seven test sessions). Positive examples are exact and outward-padded clear-card crops. Negative examples are nonoverlapping annotated UI regions and separately generated controls/background patches. There are 8448 training, 998 validation and 970 test patches. Source image/session isolation is checked by the existing manifest validator. Shared generator styles and repeated simple backgrounds are not independent real-domain evidence.

Synthetic test: 420/420 positive crops retained; 549/550 negative crops rejected. These are binary patch-classification metrics, not full-table identity accuracy or calibrated confidence.

Reproduction from repository root:

```powershell
.\.venv\Scripts\python -m apc.perception.train_card_verifier apc/runs/glyph-font-diversity-v1/dataset_manifest.json apc/runs/card-face-verifier-v1 --epochs 20
```

Use a new destination if it already exists. Recreate the source dataset from its referenced generation recipe if absent; its exact manifest/image hashes must match. Audit collections are explicitly rejected as verifier training inputs.

## End-to-end reference comparisons

The verifier filters neural proposals after geometry refinement/deduplication but before glyph recognition. It uses only crop pixels. It is attached by a separate research reader, leaving the prior fingerprint-locked inference files unchanged. Fixed incumbent detector/glyph weights and bounded five-view settings are retained.

| Reused reference | Before verifier | With verifier |
| --- | --- | --- |
| Audit-v4 nine frames | 27/27 exact, 24 false proposals | 27/27 exact, 3 false proposals |
| Audit-v4 false high-score candidates | 2 | 1 |
| Viewport-v3 six frames | 34/34 exact, 2 false proposals | 34/34 exact, 0 false proposals |
| Earlier captured development | 34/34 exact, 0 false proposals | 34/34 exact, 0 false proposals |
| Earlier mixed-surface synthetic | 80/80 exact, 0 false proposals | 80/80 exact, 0 false proposals |

All three remaining false proposals are on the 960-wide static empty fixture; one yields a high unvalidated identity score. Zero confirmed identities emitted. These comparisons reuse images that informed the problem definition; they are NOT fresh independent audits. The original v4 frozen result remains unchanged. No neural training consumes these saved frames.

## Runtime and controls

A matched sequential warm benchmark alternates variant order across three repeats of nine saved frames (27 measurements per mode), two Torch threads, after one warmup read per model. Baseline median/p95 approximately 400/1129 ms; verifier approximately 434/1088 ms. The noisy tail difference does not establish a speed improvement. Includes image I/O, proposals, filtering and glyph inference; excludes model loading, capture, temporal state and strategy. Earlier concurrent evaluation latency numbers should not be compared as overhead measurements. Detailed samples: `card-face-verifier-v1.matched-latency.json`.

Explicit saved-frame application invocation:

```powershell
.\.venv\Scripts\python -m apc inspect FRAME.png --neural-cards --adaptive-glyphs --adaptive-surfaces --deduplicate-surfaces --tiled-views --glyph-checkpoint apc/checkpoints/glyph-font-diversity-v1.pt --locator-checkpoint apc/checkpoints/card-locator-hardneg-v1.pt --verifier-checkpoint apc/checkpoints/card-face-verifier-v1.pt
```

Aggregate evaluators also accept `--verifier-checkpoint`. Defaults do not select it. No external-table control, coaching activation or automatic actions are introduced.

## Decision and next work

Keep as an explicit research candidate, not a default or calibrated identity gate. 47 targeted regression tests pass. Remaining high-score false recognition prevents readiness. Next: characterize retained false crops and held-out negative styles, improve discrimination without sacrificing true-card recall, independently review captured geometry, and freeze the complete four-model/policy configuration for NEW capture validation. Full-table state, player profiles, solver/neural strategy and self-improvement integration remain unfinished.
