# APC progress

Updated: 2026-11-05 02:48 UTC. This is a saved checkpoint snapshot, not an always-running live dashboard.

Current experiment — frozen-model viewport audit:

`[██████████] 100% — six captures, coordinate QA, frozen evaluation and audit checks completed`

Current phase: visible-table perception research. Latest candidate: `apc/checkpoints/glyph-font-diversity-v1.pt`.

- Synthetic glyph test: 220/220 correct.
- Captured development set: 34/34 identities, no false detections.
- Older mixed-brightness synthetic set: 80/80, no false detections.
- Earlier audit images, reused as a reference: 34/34, improved from 32/34. This is not a fresh independent audit.
- Prior font-model regressions: 34 passed. New viewport/mapping regressions: 17 passed. Default model and coaching readiness are unchanged.

Latest viewport audit: 27/34 correct end-to-end, three false proposals and two missed cards. With supplied glyph boxes the recognizer gets 34/34. Compact-layout crop quality and wide-layout detection need work. A provider-specific screenshot-coordinate correction was visually checked; independent label review remains pending.

Next: improve scale-aware proposals/cropping with synthetic examples, then test fresh captures. Do not train on the saved audit images.

Overall APC: **in development, not coaching-ready**. There is no defensible overall completion percentage yet. Completion still needs independently evaluated full-table perception, temporally consistent state, uncertainty-aware player profiles, solver/GTO and neural-strategy integration, virtual-chip self-play/completed-hand improvement, fast audited BB recommendations, and a verified controlling application. Existing experimental components do not by themselves prove those requirements complete.
