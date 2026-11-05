# Developer geometry review and corrected-label reference

Reviewed overlays for all nine audit-v5 frames and the one-frame capture-stability demo. Yellow card rectangles, cyan rank regions and magenta suit regions use the scoring loader's exact mapping. No model predictions determined annotation placement. This is developer visual review, not independent annotation approval.

Findings:

- Original h1 preflop/river, h2 preflop/river, h3 river visually align with card surfaces and glyphs. All visible identities agree with draft labels.
- Three negative frames visibly contain no cards; they are resized versions of one static fixture, not independent scenes.
- Original h3 preflop has both y coordinates below the actual cards. Preserved raw collection and original result unchanged.
- `collection.geometry-v2.json` changes only those two y values to `457 * 1280 / 1265`, using visible image card top y=457 and explicit width mapping. Boundary precision is approximately +/-1 image pixel. Corrected overlay aligns visually. Original card sizes, x positions, identities, all image hashes and other labels are unchanged. A regression test enforces that scope.
- Real capture-stability demo overlay aligns with its two cards. Endpoint evidence does not constitute independent pixel review.

Matching surface-pair inference, same four checkpoints and options as the frozen candidate, on corrected draft labels: **26/27 exact**, 26 localized, one compact-preflop miss, two false proposals on the wide negative fixture; both false candidates have high uncalibrated ink scores. Oracle crops: 27/27. Confirmed identities: zero. Result: `browser-routed-audit-v5.geometry-v2-surface-reference.json`.

An earlier diagnostic invocation accidentally omitted `--surface-pairs` and scored 2/27 despite oracle 27/27. It is retained as `browser-routed-audit-v5.geometry-v2-reference.json`, explicitly a corner-extraction control, NOT the candidate's matched result. No weights were modified in either run. The original preregistered result is not replaced by these post-review reference scores.

Reproduce overlays with:

```powershell
.\.venv\Scripts\python -m apc.annotator.review_browser_collection apc/data/reference/browser-routed-audit-v5/collection.geometry-v2.json apc/runs/new-geometry-review
```

The output folder must not already exist. `review.json` binds collection/image hashes and records pixel rectangles; rendering never approves or changes labels. All frames are retained; no captures enter training. Original overlays are in local ignored runs/browser-routed-audit-v5-geometry-review; corrected ones in runs/browser-routed-audit-v5-geometry-v2-review.

Next: investigate compact locator/verifier miss and wide-control false positives using separate development data, add automatic adapter manifest import, and preregister a fresh varied-scene audit. Independent geometry review and full-table integration remain unfinished. No readiness or default promotion.
