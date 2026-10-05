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

## Spatial-grid follow-up

Replacing global average pooling with a 4x4 spatial grid and a learned projection
preserves glyph layout. A second 10-epoch run on the identical manifest and seed
(`apc/runs/region-recognizer-v2`) achieved class 42/50 (84%), dealer 8/8,
hero 8/8; rank remained 0/20, suit 6/20, action 2/10. Network-only p95 was
17.589 ms. This remains a failed coaching candidate, not a readiness milestone.
Checkpoints now explicitly record architecture version `spatial-grid-v2`.
The four-frame test set is too small for broad accuracy claims. Since these test
results have been inspected during development, a fresh session-isolated audit
set is required for any final promotion decision.

## Native-detail follow-up

Training contains all 13 ranks (4–15 examples per rank); missing classes do not
explain the failure. The renderer uses small default-font glyphs, so whole-frame
downscaling from 1280 to 512 is a plausible information bottleneck. The loader
now caps at 1280 without upscaling. Grouped grid sampling extracts multiple
regions without allocating a separate full-frame copy for each. An interleaved
batch test checks that grouping does not reorder predictions.

Run `region-recognizer-v3` used the same manifest, seed and ten epochs: rank 3/20
(15%), class 42/50, suit 6/20, action 2/10; network p95 15.857 ms. This small
improvement does not establish causality or readiness. More diverse training and
dedicated glyph supervision are still necessary. Source-size preprocessing is
recorded in new checkpoint/report metadata.

## Expanded development corpus

`region-dataset-v2` uses 90 sessions / 360 frames, seed 20261006, with clocks and
names included. Manifest validation passed, with zero controlled-visible frames.
Generation commands match the first run above, substituting the new directory,
`--sessions 90`, and `--seed 20261006`. The `region-recognizer-v4` experiment uses
ten epochs, seed 42, native-detail spatial-grid architecture. Evaluation now
records per-head confusion matrices, observed/total class counts and macro recall.
Checkpoint selection averages validation head macro recall so frequent/easy
labels cannot dominate by count. Final promotion still requires a fresh audit.

## Annotation-free card proposals

`card_proposals.propose_cards` finds bright near-neutral connected surfaces and
filters them by aspect, size and fill. It consumes only frame pixels, not board
counts or learned fixed geometry. This is a limited light-card appearance
baseline: dark decks, overlaps, rotation and occlusion can fail. One-to-one IoU
matching prevents duplicate detections from receiving duplicate credit. Tests
cover a five-card flop, blank-frame rejection, duplicate matching and all
renderer layouts/themes on seven-card river frames. These tests do not prove
cross-platform localization performance.

## Expanded run result

V4 completed ten epochs. Held-out evaluation covered 36 frames / 498 regions:
class 412/498 (82.73%), rank 78/180 (43.33%, macro recall 38.30%), suit 180/180,
action 90/90, hero/dealer 120/120 each. Network-only p95 was 14.837 ms. Enabled
and visibility scores cover only one observed class and cannot establish
disabled/occluded performance. Rank recognition remains inadequate. Hero-card
versus board-card role confusion also persists: identical local crops require
table context to disambiguate. No coaching promotion or readiness gate passed.

## Pixel-to-neural integration failure

`infer_cards.CardReader` now consumes image pixels and proposed boxes, with no
annotation input. First held-out frame test found two regions in 115.964 ms
(includes image IO/proposals/network, excludes checkpoint loading). Its high-score
predictions were Qd/Qs, while the separately inspected annotation was 2c/5s.
This demonstrates that softmax scores are not reliable confidence under proposal
geometry shifts. The reader therefore emits candidates only: confirmed `card`
remains null even above the score threshold. Calibration, localization-shift
robustness and fresh held-out audits are required before identities can feed
coaching state. Hero/board role remains unresolved.

The complete V4 pixel-proposal audit covered 36 test frames / 180 cards. All
boxes matched at IoU >= .5, with zero false/missing proposals, but exact card
candidates were correct only 5/180 (2.78%). Of 138 high-score matched candidates,
five were correct (3.62%). Frame-pipeline p95 was 116.661 ms. This is stronger
evidence of failed recognition robustness than the exact-box evaluation.
The evaluator now includes unmatched high-score proposals in end-to-end
precision so false boxes cannot disappear from confidence metrics.

V5 is a ten-epoch training experiment with seed 42 and relative box jitter .06.
Jitter affects training only; validation/test geometry remains untouched. Reports
record the jitter value, and tests check bounds, batch indices and zero jitter.

Paired V4 validation diagnostic (`--split validation --compare-crops`): 180
matched cards, 73 correct with annotation crops versus ten with proposal crops;
63 exact-correct readings became wrong, and 170 candidates changed. Mean IoU
was .96117 and mean maximum edge shift was one source pixel. This isolates a
strong crop-geometry sensitivity; it does not establish the only failure cause.

V5 completed. Pixel-proposal test: 15/180 correct identities (8.33%), compared
with V4's 5/180; all 180 boxes still matched. No candidates exceeded the .9
joint-score threshold, so precision at that threshold is undefined, not 100%.
Frame-pipeline p95 was 80.008 ms on this run. Jitter improved this development
metric but did not make recognition usable; no readiness gate passed.

## Ink-normalized neural glyph experiment

`card_glyphs` isolates foreground contrast on light card faces, tightly bounds
the printed token and rescales it to a centered 32x32 input. A fixture checks
one-pixel crop invariance. This assumes light cards with simple tokens; artwork,
other fonts and real suit symbols require broader training/evaluation.

`train_card_glyphs` trains a neural rank/suit recognizer, selects by validation
exact-card count and evaluates the held-out split once. V1 used the 360-frame
manifest, seed 42 and 15 epochs. Exact-box test: 180/180 cards. Independent
pixel-proposal test: 180/180 identities, 180 matched boxes, zero false/missing
boxes across 36 held-out synthetic frames. 145 readings exceeded .9 raw joint
score, all correct on this development set; scores remain uncalibrated. Pipeline
p95 was 90.835 ms, excluding checkpoint loading. This is a synthetic-domain
milestone only; confirmed cards remain null and full-table readiness remains
false. Real/controlled-visible and fresh cross-domain audits remain necessary.

Reproduce the glyph run and independent-box audit:

```powershell
.\.venv\Scripts\python.exe -m apc.perception.train_card_glyphs apc/runs/region-dataset-v2/dataset_manifest.json apc/runs/card-glyph-v1 --epochs 15
.\.venv\Scripts\python.exe -m apc.perception.evaluate_card_reader apc/runs/card-glyph-v1/glyph_weights.pt apc/runs/region-dataset-v2/dataset_manifest.json
```

Use a fresh output directory if the run already exists. Runtime checkpoint and
evaluation files are local ignored artifacts. Ink-free proposals explicitly
report `missing_ink`, null candidate and zero identity scores.

Frozen user-frame audit: the glyph reader produced only one proposal and guessed
Td (actual localized board card was 2s); score remained uncertain, with no
confirmed card emitted. Pipeline time was 195.794 ms. This frozen partial frame
was not used for training or readiness counts. It exposes domain-specific light
surface/whole-token assumptions. Optional `rank_box`/`suit_box` annotations now
support separately labeled glyph regions inside card boxes; the validator checks
containment and finite coordinates. Spatial targets preserve these explicit
labels without inventing missing glyph regions. Full test suite before this
schema extension: 262 passed.

The renderer can now emit explicit glyph bounds using `--include-glyph-boxes`;
the trainer consumes them with `--glyph-pair` and refuses missing glyph labels.
It normalizes each labeled region independently using the surrounding card's
background color and concatenates six RGB channels. Pair checkpoints have a
distinct model kind/architecture and cannot be loaded as a whole-card reader.
V1 paired run: nine sessions, 36 frames, seed 20261007, 15 epochs, seed 42;
validation-selected checkpoint achieved 17/20 exact held-out cards, suit 20/20.
These are annotation-supplied glyph regions, not independent localization.
Next: diverse fonts/symbols/corner placements and pixel-based glyph proposals.

The renderer now supports `--card-design corner_symbols`: larger corner rank
glyphs, vector-drawn suit symbols, and a larger suit illustration. The design
is reflected in the theme identifier rather than silently relabeling identical
styles. Default centered-token output is preserved. A corner-symbol paired run
(36 frames / nine sessions, seed 20261008, 15 epochs) achieved 14/20 exact cards
and 20/20 suits using labeled glyph regions. It is not a promotion candidate.
Both designs pass dataset validation and six-channel extraction tests. A sample
render was visually inspected for placement and glyph readability.
