# Explicit raster alignment for draft browser collections

Resize diagnostics demonstrated that a screenshot can be a stale/cropped raster
even when before/after DOM observations agree. Do not assume full-width scaling
from dimensions alone, and do not select the second capture as a silent retake.

The browser collection loader and its model-independent overlay renderer now
support per-frame `screenshot_mapping: "measured_pixel_alignment_v1"`. Supply a
`pixel_alignment` object with:

- `schema_version: "apc_pixel_alignment_v1"`;
- `image_sha256`, `image_size: [width, height]`, and `viewport`, bound to that frame;
- `measurement_method: "manual_visible_landmarks"`;
- at least four `anchors`, each containing a distinct visible landmark
  `description`, its `dom_xy`, and independently measured screenshot `pixel_xy`.

Choose real corresponding visible corners distributed around the labeled region.
Do not compute pixel points from assumed screenshot width or model predictions:
that would merely restate the assumption being checked. Measure native image
pixels, not display-preview coordinates. Anchors must span at least 25% of each
viewport axis and must not be collinear. The loader fits positive axis-aligned
scale/translation, permits at most two pixels of landmark residual and 2% scale
anisotropy, and rejects boxes outside landmark coverage or saved-image bounds.
These are explicit draft geometry tolerances, not learned confidence thresholds.

Card, rank and suit labels all use this same fitted mapping. The existing overlay
renderer therefore shows exactly what the evaluator would score. Review every
overlay independently before approving labels. Declared manual measurement is
not proof that correspondences are correct; the geometry check cannot verify the
truth of a human match or detect motion between anchors. Unsupported rotations,
perspective transforms and ambiguous/cropped card regions must be rejected.

Legacy collection mappings remain available for historical diagnostics; this
change does not retroactively validate them, alter sealed captures, or award
readiness/training approval. New frozen audits must include the new helper and
updated loader hashes in their source contract; old plans require archived code.
The six-frame resize diagnostic remains unapproved. Real landmark measurements
and overlays must be checked per image, not inferred from these software tests.

## Importing into the labeling project

`python -m apc.annotator.import_browser_acquisition <project> <capture.json> --session <id> --timestamp-ms <measured-relative-time> --pixel-alignment <measurement.json>`

The optional measured path uses the same transform as collection evaluation and
review overlays. It requires matching landmark descriptions/DOM points in the
capture's before/after observations. Image bytes, size, endpoint geometry,
measurement/image binding, and all mapped card boxes are checked before any
project mutation. The suggestion records the original measurement and its hash.
JPEG-to-PNG import preserves decoded RGB pixels without rescaling, so normalized
coordinates remain unchanged. No annotation is approved automatically.

Omitting the option retains the legacy width-scale assumption for historical
workflows; it is explicitly named in new suggestions, not asserted to be a
verified pixel mapping. Prefer measured alignment for new captures. The saved
`measured-alignment-demo-v1` provides a real one-frame developer check; independent
review and cross-size/cross-layout evidence are still outstanding.

## Workbench import

The local annotation workbench now has **Import measured table capture**. Supply
the absolute paths to the original capture manifest and its measurement JSON,
plus the correct session and measured relative timestamp in the shared fields.
This invokes the same validated importer under the application's project lock.
The new endpoint accepts JSON and rejects cross-origin browser submissions.

After import, **Apply suggested fields to draft** supports capture-derived card
boxes and identities as well as model suggestions. It copies capture cards into
a new unsaved draft, marks visibility uncertain and provenance unverified, and
does not invent missing seats, pot or hand state. Explicit application replaces
the draft's hero/board card arrays; other object/state fields are preserved.
Applying is not saving or approving: all remaining fields still need review.

Developer UI verification imported the real measured J-spade/T-heart example into
the local ignored `apc/runs/workbench-measured-import-demo-v1` project and applied
its suggestion. The screen showed one frame, zero verified annotations, card
outlines, and the unsaved-draft confirmation. No annotation was saved. Screenshot
proof is local in that project's `ui-proof.jpg`; this is application integration
evidence, not an additional training sample or independent label review.

## Frozen evaluation contract

A future scored audit using this mapping must declare
`required_screenshot_mapping: "measured_pixel_alignment_v1"` and
`capture_stability_required: true` before collection. It must fingerprint
`pixel_alignment.py`, `evaluate_browser_collection.py`, and `capture_stability.py`
in `inference_source_sha256`, in addition to all inference dependencies. The
evaluator rejects an undeclared/mixed mapping, missing geometry-source lock,
changed helper or missing stability requirement before constructing the reader.
This protects reproducibility; it still does not independently certify manually
supplied pixel correspondences. Existing sealed reports/plans are not rewritten.
