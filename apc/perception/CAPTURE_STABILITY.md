# Stable visible-frame collection

## Automatic browser persistence bridge

The authorized JS caller can pass the result of `capturePracticeFrame(tab)` to `saveBrowserAcquisition(acquired, NEW_FOLDER, {usePolicy: 'draft_review_no_training'})` from `apc/capture/save_browser_acquisition.mjs`. Parent folder must already exist. It writes original image bytes plus `capture.json` without manual metadata transcription; existing leaf folders are refused. Use `sealed_audit_no_training` for audit acquisitions; that policy is carried into the manifest and refused by the labeling importer. This bridge has controlled callback and cross-runtime integration coverage; a new live-browser persistence demonstration remains pending.

Both PNG and JPEG browser output are detected from bytes. JPEG originals are saved as `frame.jpg`, PNG as `frame.png`; no capture bytes are transcoded by JS. Python revalidates actual image decoding and dimensions. Dataset v1 accepts PNG only, so the importer creates a lossless PNG encoding of decoded JPEG RGB pixels, preserving the original acquisition, source hash/format and derivative method in the suggestion. This is not an additional JPEG compression pass or approval of labels.

Legacy audit/demo files ending in `.png` may contain JPEG bytes from the browser provider. Their original files/hashes remain unchanged for reproducibility. New acquisitions use truthful extensions. Existing collection-loader behavior does not retroactively certify native format or annotation geometry.

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

## Import saved acquisitions into the labeling app

`python -m apc.annotator.import_browser_acquisition PROJECT CAPTURE_JSON --session SESSION --timestamp-ms MEASURED_TIMESTAMP` imports the Python collector's saved manifest directly into a dedicated `apc-full-hand-browser` project. Supply the actual measured timestamp; the importer does not invent timing. It revalidates bytes, PNG dimensions, before/after evidence, labels and screenshot crop bounds before importing a frame. It creates a review suggestion, never an approved annotation or readiness credit. Sealed-audit captures are rejected; original audit collections must stay outside training. Existing review work is preserved. Rejected acquisition files remain unchanged at their source.

The supported mapping for this importer is the capture provider's uniform-width-scale/top-left crop. Other providers require an explicit adapter before use. This removes manual conversion for saved Python acquisitions; the JavaScript adapter still needs an automatic persistence bridge.

Real-browser adapter: `apc/capture/practice_browser_adapter.mjs` exports `observePracticeFrame(tab)` and `capturePracticeFrame(tab)` for an authorized browser tab exposing the documented Playwright/evaluate and screenshot APIs. It reads only APC's own visible card DOM, not private game state; it is labeling infrastructure, not the neural perception model. The single-frame demonstration at `apc/data/reference/capture-stability-demo-v1` verifies actual browser acquisition and loader integration. Automatic cross-runtime persistence/import remains unfinished.

`apc.capture.browser_frame.capture_browser_frame(observe, screenshot)` accepts two async callbacks from the permitted browser adapter. `observe` must return the visible-only snapshot described above; `screenshot` must return PNG bytes. The collector calls observe, screenshot, observe exactly once, copies observations immediately, binds image dimensions/hash, validates stability and returns bytes plus eligibility/rejection evidence. It does not control a table, retry, or inspect model predictions.

`save_browser_frame(result, folder)` writes `frame.png` and `capture.json` in a new folder only, retaining both eligible and rejected acquisitions. Existing folders are refused. Capture callback failures propagate without fabricating frames. Persistence errors must also propagate; a folder without `capture.json` is incomplete and must not be imported. Non-JSON-safe malformed observations cause persistence to fail before folder creation. This API is tested with controlled callbacks; an actual browser adapter and end-to-end capture demonstration are still required.
