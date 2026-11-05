# Stable visible-frame collection

New browser collections should set `capture_stability_required: true`. Each frame must include `capture_stability` with:

- `schema_version`: `apc_capture_stability_v1`
- `image_sha256` and `image_size`: the saved PNG's SHA-256 and `[width, height]`
- `before` and `after`: independent visible-DOM observations bracketing the screenshot, each containing `viewport`, `scroll` (`[scrollX, scrollY]`) and `cards`
- Each snapshot card: `[role, identity, x, y, width, height]`, in the same order as draft labels. Use only visibly rendered card information; never hidden game state.

Do not navigate, scroll, advance a hand or resize between observations and screenshot. Snapshot metadata must be collected at capture time, not reconstructed later. Both snapshots must match exactly, including scroll and geometry. Card labels must match their captured geometry. Image dimensions and hash bind the evidence to the saved PNG. The loader rejects missing required evidence, movement or mismatches before scoring.

Retain rejected frames and their evidence for diagnosis; never replace samples based on model predictions. A rejected frame must not enter training or accuracy scoring. A fresh audit schedule should define rejection/retry handling before collection starts.

This is a stability check, not an atomic screenshot guarantee: identical endpoints do not exclude transient movement. Review image-space card and glyph rectangles over saved pixels independently before reporting validated accuracy. Provider scaling/cropping must still use an explicit mapping. This evidence provides no coaching-readiness credit.

Legacy collections remain readable for reproducibility but are not retroactively certified. Evaluation reports count frames with validated stability evidence and explicitly leave pixel review false. Audit v5's original labels and result are preserved; its detected wide-preflop mismatch remains unresolved.

## Collector API

`apc.capture.browser_frame.capture_browser_frame(observe, screenshot)` accepts two async callbacks from the permitted browser adapter. `observe` must return the visible-only snapshot described above; `screenshot` must return PNG bytes. The collector calls observe, screenshot, observe exactly once, copies observations immediately, binds image dimensions/hash, validates stability and returns bytes plus eligibility/rejection evidence. It does not control a table, retry, or inspect model predictions.

`save_browser_frame(result, folder)` writes `frame.png` and `capture.json` in a new folder only, retaining both eligible and rejected acquisitions. Existing folders are refused. Capture callback failures propagate without fabricating frames. Persistence errors must also propagate; a folder without `capture.json` is incomplete and must not be imported. Non-JSON-safe malformed observations cause persistence to fail before folder creation. This API is tested with controlled callbacks; an actual browser adapter and end-to-end capture demonstration are still required.
