# Captured browser development collection

Six original viewport screenshots from APC's full-hand virtual-chip browser table,
captured through the browser automation screenshot API. This is **one capture
session**, three different hands, 34 card appearances and 19 distinct identities.
The four frames from hand 1 are correlated; no independent sample count or
cross-platform performance is implied. No external poker client was controlled.

`collection.json` records screenshot hashes and rounded DOM geometry. Glyph boxes
are the visible rank/suit span bounds, not ink contours. Each image and its DOM
observation were collected without an intervening table action. Visible card
identities were visually checked, but independent overlay/geometry review remains
pending. The screenshots can differ in pixel size from the DOM viewport; boxes
must be normalized using the recorded 1280x720 DOM dimensions. Several views are
scrolled and omit part of the table. Do not infer missing seats or hidden cards.

This compact draft format is intentionally not a trainable annotation manifest.
There are no measured capture timestamps, assigned data splits, verified full-table
annotations, readiness credit or completed training. Do not relabel individual
hands as independent capture sessions to satisfy export or readiness thresholds.
These development examples are not the frozen browser-reference audit set and may
be used for future training only after review and a documented split protocol.

Reproduce the diagnostic:

```text
python -m apc.perception.evaluate_browser_collection apc/data/development/browser-domain-bridge-v1/collection.json
```

The report compares the current full-frame card pipeline with oracle DOM glyph
crops. Oracle accuracy diagnoses recognition separately; it is not deployment
accuracy. Both modes use the same neural glyph checkpoint. No weights are updated.
