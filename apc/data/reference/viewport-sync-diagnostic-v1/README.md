# Resize synchronization setup diagnostic

Six declared captures were retained, in order, without model inference, training,
scoring, retakes, or sample selection. This does not resume aborted audit v7.

| Requested viewport | First image | Second image | Same DOM geometry |
| --- | --- | --- | --- |
| 920 × 900 | 905 × 509 | 905 × 860 | Yes |
| 600 × 900 | 585 × 556 | 585 × 853 | Yes |
| 1440 × 900 | 600 × 875 | 1425 × 866 | Yes |

The first wide image passes the existing coordinate bounds check despite being
visibly cropped horizontally: the 600-pixel raster contains only the left part of
the wide layout, with hero cards absent. Scaling it as though it represented the
full 1440-pixel DOM viewport would create incorrect labels. This is direct evidence
that endpoint equality and bounds checks are necessary but insufficient.

The observations are consistent with resize/render/capture desynchronization;
they do not identify which browser layer causes it. The second captures are not
automatically approved, and a universal "take the second screenshot" fix is not
established. Independent pixel review is still pending. No readiness credit.

Next: establish an explicit raster/DOM calibration contract before a newly
preregistered audit. Preserve native images and reject ambiguous mapping; do not
infer scale from image width alone after viewport changes. Keep setup diagnostics
separate from audit sampling and retain failed captures.

The local practice server was stopped and the temporary viewport override reset.
`result.json` can be regenerated into a **new** output path using
`python -m apc.perception.diagnose_viewport_sync <this-folder> --output <new-path>`.
