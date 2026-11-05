# Frozen fast-reader audit v6

All 15 preregistered frames were acquired after plan commit e4f5c47: three own-application preflop/flop/turn/river sequences and three distinct static negative fixtures. Original JPEG bytes and automatic acquisition manifests are retained. Every frame has matching before/after geometry/scroll evidence; negatives also bind fixture identity. No replacement or prediction-based selection occurred. No audit image enters training.

Developer reviewed a contact sheet of all card/rank/suit overlays and the three negative scenes; alignment appears consistent and no labels were corrected. This is not independent annotation approval. Review artifacts are locally reproducible using `apc.annotator.review_browser_collection`; annotations remain draft. The collection_status value records the manifest's assembly state before scoring; the separate result records the completed run.

Frozen result: 57/60 visible card appearances localized and recognized, three misses (standard flop, compact flop and compact turn), three false proposals on the disabled-control fixture. No false ink-based identities, no high-score false candidates, no confirmed identities. Oracle glyph crops 60/60. Source/model/schedule/viewports/new-session/image-disjointness checks passed. Result: `apc/perception/baselines/browser-fast-text-audit-v6.result.json`.

Important limitation: the result's image field uses basename frame.jpg for each acquisition, so rows must be identified by schedule order plus hand_group, not basename alone. Full relative image paths and street are authoritative in collection.json. This run's scoring is unaffected because the evaluator processes each path individually; future diagnostic/timing grouping must not use basename as a unique key.

This is a fresh same-application session, not cross-platform validation. Repeated street appearances are correlated. Static negative fixtures are not observed gameplay transitions. No seat/stack/action/profile/strategy or full-coach evaluation is implied. Calibration and independent geometry review remain unfinished. No default promotion; APC is not coaching-ready.

Capture used only authorized browser controls on local virtual chips. Temporary viewport restored; local capture server session 32088 stopped. Raw captures, plan and result remain immutable research evidence.
