# Frozen viewport audit v3

One new capture session, three hands, six planned frames and 34 card appearances
(20 distinct identities). Detector `8168a060…` and recognizer `8e237836…` were frozen
before collection. Hand 1 was captured at 960x800 over four streets, hand 2 at
600x800 on the river, and hand 3 at 1440x900 on the river. All planned captures,
including failures, are retained. No external poker client was operated.

The compact CSS layout changes card size from 62x78 to 44x64 DOM pixels. The
collection format therefore records per-frame viewport, card and glyph geometry.
Card identities were visually checked; independent geometry review remains pending.
Frames within the first hand are correlated; these are not three independent
capture sessions or cross-platform validation.

Coordinate QA found the provider uniformly scales screenshots using the requested
DOM width, and may crop output vertically. The 1440x900 frame is 1425x866 pixels;
stretching coordinates to its output height misplaces glyph crops. This corpus
explicitly declares `uniform_width_scale_top_left_crop`. Its source-coordinate
height is `image_height * DOM_width / image_width`, not necessarily DOM height.
An overlay was visually checked on the wide frame; no detector predictions are
used to construct the transform. Do not apply this provider-specific assumption
to other sources without review. Original images were not replaced or edited.

Corrected frozen-model results:

| Scope | Targets | Exact identities | False proposals | Misses |
| --- | ---: | ---: | ---: | ---: |
| 960x800, four frames | 20 | 20 | 0 | 0 |
| 600x800, compact river | 7 | 2 | 3 | 0 |
| 1440x900, wide river | 7 | 5 | 0 | 2 |

Oracle glyph crops recognize 34/34, isolating proposal/crop weaknesses rather than
incorrectly declaring a new font-recognition failure. There is no coaching readiness
or verified whole-table state. All identities remain unconfirmed in inference.

`sealed_audit_no_training` prohibits training/fine-tuning on these images. Future
work informed by this audit needs a fresh audit before untouched-evaluation claims.

```text
python -m apc.perception.evaluate_frozen_browser_audit --viewport-v3
python -m apc.perception.render_browser_audit_overlay apc/data/reference/browser-viewport-audit-v3/collection.json apc/runs/viewport-qa.png --frame-index 5
```
