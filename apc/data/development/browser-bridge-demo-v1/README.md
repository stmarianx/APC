# Fresh live capture -> automatic persistence -> labeling demonstration

On 2026-11-05 the authorized in-app browser opened APC's own new virtual-chip practice session. No gameplay action was taken. `capturePracticeFrame(tab)` read visible card geometry, captured an image, then read geometry again. `saveBrowserAcquisition` automatically saved the image and acquisition manifest here, with no metadata transcription. Matching snapshots: scroll [0,0], viewport 1280x720, hero 7h/4d. Native JPEG size 1265x712; original SHA-256 is in capture.json. The single-frame acquisition took about 138ms in this browser session; not a latency benchmark.

The supported importer consumed capture.json directly into labeling-project. Its status: one frame, one metadata suggestion, zero approved annotations, zero verified annotations. Timestamp 0 is the origin of this one-frame session, not an invented inter-frame duration. No temporal/action training evidence is claimed.

Dataset v1's PNG is a lossless encoding of decoded source RGB pixels, not a second JPEG compression. A regression test checks decoded pixel equality, original source hash and suggestion labels. Original JPEG is retained unchanged. The suggestion is unverified and requires independent review; no training or readiness credit.

The existing four-network neural reader was separately run on frame.jpg using explicit font-diverse glyph/hard-negative locator/appearance-v4 verifier/bounded-v1 verifier checkpoints, adaptive glyphs/surfaces, deduplication and tiled views. It produced candidates 4d and 7h, with no extra proposals. Both had card=null and unresolved roles, as intended for the research model. Observed inference time approximately 441ms excludes capture, checkpoint loading and coaching. This two-card development smoke test is not a frozen accuracy audit and does not replace the earlier 26/27 corrected-label reference or resolve its failures.

Reproduce import into a new dedicated project with:

```powershell
.\.venv\Scripts\python -m apc.annotator.import_browser_acquisition NEW_PROJECT apc/data/development/browser-bridge-demo-v1/capture.json --session browser-bridge-demo-v1 --timestamp-ms 0
```

Do not import twice over existing review work. Local server session 62986 stopped after capture. No viewport override was used. No external poker platform was accessed. Next: develop compact-card and card-like-control failure cases without training on audit images.
