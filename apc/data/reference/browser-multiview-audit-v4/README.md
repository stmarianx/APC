# Fresh multiview audit v4

Collected 2026-11-05 from APC's own local virtual-chip table through the browser, after freezing neural weights, inference options and five inference-source fingerprints in `apc/perception/baselines/browser-multiview-audit-v4.plan.json`.

Nine scheduled screenshots, retained without replacement: preflop/river from three new hands at 960x800, 600x800 and 1440x900, then the dedicated `/audit-empty` static negative/distractor fixture at those sizes. The negative scene is NOT an observed gameplay transition or three independent negative scenes. Browser screenshot provider cropping/scaling varies, particularly in the negative frames. Empty frames have no target geometry, so their false-proposal counts do not depend on coordinate mapping. Positive-frame bounds use the previously documented width-scale/top-left mapping.

27 visible card appearances, 17 unique identities, one capture session. Identities were visually checked at capture; card/glyph rectangles are draft DOM measurements. Independent geometry review is pending. No hidden cards or internal engine state supplied labels. Screenshot bytes are original browser captures with SHA256 fingerprints. The viewport override was reset and the local server stopped after collection.

Result: 27/27 exact visible identities and no missed cards, but 24 false proposals: four on gameplay frames and 20 across the empty scenes (7, 3, 10). Seven false proposals yielded ink-based identities; two had high unvalidated scores. Zero confirmed identities emitted. The candidate is NOT coaching-ready and no defaults were changed. See `apc/perception/baselines/browser-multiview-audit-v4.result.json` for all frames and frozen-source/config checks.

Observed one-pass p95 was approximately 919 ms over nine frames, including image I/O, proposals and glyph inference, excluding checkpoint loading and any coaching backend. This is not a matched warm/cold benchmark or a latency-readiness gate.

Use policy: sealed audit, **no training**. This session is new relative to prior development/audit collections, but it is same-application, correlated, narrow-domain evidence, not platform-agnostic validation. Any later candidate chosen using these failures needs another fresh audit. Do not replace failures, tune thresholds on these labels and call the result independent, or count empty frames as hands.

Reproduce with `.\.venv\Scripts\python -m apc.perception.evaluate_frozen_browser_audit --multiview-v4 --output REPORT.json` from a matching source checkout. Fingerprints are strict source-byte checks; line-ending changes can require restoring the matching checkout, not silently revising this frozen plan. No new model training occurred during collection/scoring.

Next: generate separate synthetic card-vs-control/background/blank examples, train/evaluate a proposal verifier or improved detector with retained original-scale examples, assess confidence errors, then freeze a new candidate for fresh captures. Keep these images excluded from training manifests.
