# APC progress

Updated: 2026-11-05 03:11 UTC. This is a saved checkpoint snapshot, not an always-running live dashboard.

Current experiment — multi-view card detection:

`[██████████] 100% — optional multi-view candidate verified on reused references; fresh capture audit pending`

Current phase: visible-table perception research. Latest candidate: `apc/checkpoints/glyph-font-diversity-v1.pt`.

- Synthetic glyph test: 220/220 correct.
- Captured development set: 34/34 identities, no false detections.
- Older mixed-brightness synthetic set: 80/80, no false detections.
- Earlier audit images, reused as a reference: 34/34, improved from 32/34. This is not a fresh independent audit.
- Prior font-model regressions: 34 passed. New viewport/mapping regressions: 17 passed. Default model and coaching readiness are unchanged.

Latest viewport audit: 27/34 correct end-to-end, three false proposals and two missed cards. With supplied glyph boxes the recognizer gets 34/34. Compact-layout crop quality and wide-layout detection need work. A provider-specific screenshot-coordinate correction was visually checked; independent label review remains pending.

Latest completed work: identified the remaining wide-view misses as low neural detector scores. Optional overlapping zoom views with bounded-surface evidence recover both cards. Geometry-only suppression removes one duplicate compact proposal. Reused viewport identities now reach 34/34, with two false proposals and zero misses. Old development and reused audit-v2 each remain 34/34; synthetic retention remains 80/80, all with zero false proposals/misses. Unrestricted tiled proposals were rejected due to 118 false proposals. No neural weights or defaults changed. Saved-frame CLI inspection exposes the explicit candidate options; it still emits unconfirmed identities and unresolved roles.

Next work: freeze the complete optional inference configuration for a new capture audit including multi-viewport, blank/transition frames and distractors; independently review geometry and measure matched latency. Two false proposals remain, so recall alone is not readiness. Detailed results and reproduction instructions: `apc/perception/baselines/tiled-surfaces-bounded-v1.experiment.md`. No training job remains running from this checkpoint.

Verification for this checkpoint: 54 targeted regression tests passed, including tiled coordinate mapping, bounded added-view filtering, geometry deduplication, CLI model selection and desktop compatibility. Observed synthetic p95 approximately 305 ms; this is not a full-coach latency gate. Unrelated name-OCR edits were left untouched.

Overall APC: **in development, not coaching-ready**. There is no defensible overall completion percentage yet. Completion still needs independently evaluated full-table perception, temporally consistent state, uncertainty-aware player profiles, solver/GTO and neural-strategy integration, virtual-chip self-play/completed-hand improvement, fast audited BB recommendations, and a verified controlling application. Existing experimental components do not by themselves prove those requirements complete.
