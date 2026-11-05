# Measured alignment: real capture development check

One default-viewport capture of APC's own virtual-chip table; no resize, retake,
game action, inference, training or scoring. The original JPEG and before/after
observations are retained in `acquisition/`. The first save invocation rejected an
unsupported use-policy string before persistence; the same captured bytes were
then saved under `draft_review_no_training`, without taking another screenshot.

The optional adapter records four visible table outer-border extrema as DOM
landmarks before/after the screenshot. `measurement.json` contains developer
visual measurements of those same extrema in the native 1265 × 712 image—not
coordinates generated from a width ratio or from model detections. The explicit
fit places the J-spade and T-heart card outlines at [519,530,581,607] and
[588,530,650,607]. Developer visual inspection of `review/000-frame.png` found
both card outlines on the visible faces and both glyph bands containing their
intended symbols. Bands are broad upper/lower card content regions, not exact
text-span bounds. This is a check of one frame and one owned layout only.

The collection and review remain draft/unapproved, independent annotation review
is pending, and readiness credit is false. Human correspondence truth is not
established by the transform validator. No historical audit was resumed or
modified. The local server was stopped after collection.

Reproduction (new output paths only):

1. `python -m apc.annotator.prepare_aligned_collection <acquisition/capture.json> <measurement.json> <new-collection.json>`
2. `python -m apc.annotator.review_browser_collection <new-collection.json> <new-review-folder>`

The output collection must contain the original acquisition within its directory
tree. The preparer rejects sealed/rejected captures, changed image bytes,
unmatched landmark identities/DOM points, unsupported geometry and overwrites.
Next: repeat at planned compact/wide sizes with every capture retained, then
obtain independent overlay review before a new frozen perception audit.
