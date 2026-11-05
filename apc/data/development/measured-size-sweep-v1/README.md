# Compact/wide measured alignment setup check

All four planned captures are retained, without model inference, scoring,
training, extra screenshots or replacement samples. One hand, one owned table,
one session; these are correlated setup observations, not four independent hands.

| Viewport | First raster | Second raster | Developer disposition |
| --- | --- | --- | --- |
| 600 × 900 | 585 × 329 | 585 × 853 | First rejected by bounds; second manually measured |
| 1440 × 900 | 600 × 875 | 1425 × 866 | First visually clipped despite passing bounds; second manually measured |

Both pairs have identical before/after and cross-pair DOM geometry, including
table-border landmarks. Both pairs disagree in raster dimensions. The first wide
image has exactly the same SHA-256 as `viewport-sync-diagnostic-v1/w1440-first`,
even though this session reports different cards; no cards are visible in that
clipped image. This duplicate is retained and is not presented as novel data.

`result.json` binds and summarizes all original images. No alignment evidence
or collection was fabricated for either first capture: complete corresponding
table-border extrema cannot be measured there. Original manifests are unchanged,
including the insufficient `eligible_for_draft_scoring: true` on the first wide
image. That field is not an annotation approval.

Each second-capture folder contains independent native-image developer pixel
measurements of the four visible table outer-border extrema, a derived **draft**
collection, and the exact-loader review overlay. Developer visual inspection
found both card outlines on the 9-spade and T-spade faces at compact and wide
sizes, with upper/lower bands enclosing ranks/suits. Independent annotation
review remains pending. The reviewed subset was determined by observable
landmark availability for this diagnostic, not model outputs, and must not be
reported as an unbiased accuracy benchmark. Taking the second screenshot is not
established as a universal fix.

No labels are approved, no readiness credit, no weights changed. Temporary browser
sizing was reset and the local table server stopped. Session identity is now an
explicit optional `--session` argument to `prepare_aligned_collection`, used here
as `measured-size-sweep-v1` instead of incorrectly inheriting the earlier demo ID.

Next: obtain independent annotation review; integrate pixel-grounded capture
validation into a newly declared evaluation protocol before collecting a scored
audit. The old v7 remains aborted and untouched. Broader perception fields,
temporal state, neural coaching quality and latency remain separate open gates.
