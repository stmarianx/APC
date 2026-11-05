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
The six-frame resize diagnostic remains unapproved. Real landmark measurement
and overlay review are the next steps, not yet completed by this implementation.
