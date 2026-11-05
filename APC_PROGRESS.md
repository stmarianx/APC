# APC progress

Updated: 2026-11-05 03:04 UTC. This is a saved checkpoint snapshot, not an always-running live dashboard.

Current experiment — wide-table crop regression:

`[██████████] 100% — crop fix verified on reused references; remains opt-in; next detector experiment pending`

Current phase: visible-table perception research. Latest candidate: `apc/checkpoints/glyph-font-diversity-v1.pt`.

- Synthetic glyph test: 220/220 correct.
- Captured development set: 34/34 identities, no false detections.
- Older mixed-brightness synthetic set: 80/80, no false detections.
- Earlier audit images, reused as a reference: 34/34, improved from 32/34. This is not a fresh independent audit.
- Prior font-model regressions: 34 passed. New viewport/mapping regressions: 17 passed. Default model and coaching readiness are unchanged.

Latest viewport audit: 27/34 correct end-to-end, three false proposals and two missed cards. With supplied glyph boxes the recognizer gets 34/34. Compact-layout crop quality and wide-layout detection need work. A provider-specific screenshot-coordinate correction was visually checked; independent label review remains pending.

Latest completed work: diagnosed the bounded-surface regression as oversized detector boxes failing an original-area overlap rule. Comparing against the smaller area restores wide-view recognition to 5/7 while retaining compact recognition at 7/7. Reused viewport identities improve from legacy 27/34 and adaptive-v1 30/34 to adaptive-v2 32/34; three false proposals and two missed cards remain. Old captured development stays 34/34 and synthetic retention 80/80 with no false proposals. Broader independent-axis training remains rejected due to false detections. No neural weights or defaults changed. No fresh audit or coaching-readiness claim is made.

Next work: address missed wide-view cards and false proposals, strengthening synthetic responsive-layout/distractor coverage while retaining original-scale examples. A new frozen-model capture audit is required after selecting a candidate. Detailed results and reproduction instructions: `apc/perception/baselines/adaptive-surface-v2.experiment.md`. No training job remains running from this checkpoint.

Verification for this checkpoint: 41 targeted regression tests passed (including oversized/narrow/unbounded surfaces, neighboring distractors, paired-crop diagnostics, detector/reader inference and frozen viewport audit checks). Unrelated name-OCR edits were left untouched.

Overall APC: **in development, not coaching-ready**. There is no defensible overall completion percentage yet. Completion still needs independently evaluated full-table perception, temporally consistent state, uncertainty-aware player profiles, solver/GTO and neural-strategy integration, virtual-chip self-play/completed-hand improvement, fast audited BB recommendations, and a verified controlling application. Existing experimental components do not by themselves prove those requirements complete.
