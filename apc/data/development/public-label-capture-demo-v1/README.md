# Real public-label capture path check

Two planned setup screenshots of the same owned virtual-chip hand are preserved.
No game action, hidden-state query, model inference, scoring, training or extra
capture occurred. Both endpoint snapshots include visible DOM regions and agree
exactly. Raster sizes differ: first 1265 × 712, second 1265 × 864, despite both
reporting a 1280 × 900 DOM viewport. Both pass the old card-bounds check. The first
visibly clips the action controls and remains unimported; no replacement audit
sample or accuracy result is claimed.

The second capture has developer-measured native-image table-border landmarks.
The dedicated `labeling-project` imports it as an **unverified suggestion**, with
a lossless PNG encoding of the original JPEG's decoded RGB pixels. Its single
frame is assigned relative time zero as the standalone import session origin;
this does not reconstruct timing between the two setup captures.

Visible draft labels successfully imported:

- Hero seat 1, 99.5 BB, dealer; Opponent seat 2, 99 BB;
- pot 1.5 BB, call price 0.5 BB, preflop;
- hero 2-diamond / 6-club, no board cards;
- observed enabled action names fold/call/raise/all_in.

The mapped card faces, two seat regions and pot region are shown in
`review/overlay.png`. Developer visual inspection found the boxes over those
visible objects. All four action-button boxes are explicitly skipped because
their DOM positions are below the measured table-border landmark coverage.
Their semantic action names remain review-only draft values, not verified
button-box annotations. This incomplete capture cannot count toward the verified
full-table dataset. Independent review is pending, annotations/verified count are
zero, and readiness credit is false. Names are only the owned table's visible
"Hero"/"Opponent" labels, not persistent opponent identity resolution.

The reusable `apc.annotator.review_capture_suggestion` command renders all supplied
card/seat/pot/button draft boxes from a fingerprint-checked project suggestion
into a new review folder without changing or approving annotation work. This
extends the earlier card-only collection overlay path.

Temporary browser sizing was reset and the local server stopped. Next: add
visible landmark coverage around controls, independently review complete labels,
then expand to additional hands, streets and layouts. No neural recognition,
strategy quality, speed or visible-table readiness claim follows from this demo.

## Extended controls measurement v2

`second/measurement-controls-v2.json` adds four manually measured visible button
border extrema to the original four table anchors. Its DOM positions are derived
only from `public_regions.action_buttons` rectangles already preserved in both
capture endpoint snapshots. The binder requires all original table anchors and
rejects unknown descriptions, changed DOM coordinates and duplicate controls.
The raw capture, original measurement, initial project and initial review are
unchanged. No new screenshot was taken for this extension.

`review-controls-v2/overlay.png` now contains nine mapped regions: two card faces,
two seats, pot, and fold/call/raise/all_in buttons. None are skipped. Developer
visual inspection found button outlines on the corresponding visible controls.
Independent review remains pending; these are still incomplete, unapproved
full-table annotations and not model predictions. This is the **same source
frame**, not an additional dataset example or independent capture session.

Reproduce in a fresh dedicated `apc-full-hand-browser` annotation project using
`import_browser_acquisition` with the v2 measurement, session
`public-label-capture-demo-v1`, timestamp 0, then `review_capture_suggestion` into a
new folder. The local comparison project is ignored under
`apc/runs/public-controls-import-v2`; regression tests reconstruct it in a new
temporary project and compare the fingerprinted review report exactly.
