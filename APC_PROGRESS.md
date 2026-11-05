# APC progress

Updated: 2026-11-05 03:00 UTC. This is a saved checkpoint snapshot, not an always-running live dashboard.

Current experiment — scale-aware detector training:

`[██████████] 100% — experiment complete; neither candidate promoted; next experiment pending`

Current phase: visible-table perception research. Latest candidate: `apc/checkpoints/glyph-font-diversity-v1.pt`.

- Synthetic glyph test: 220/220 correct.
- Captured development set: 34/34 identities, no false detections.
- Older mixed-brightness synthetic set: 80/80, no false detections.
- Earlier audit images, reused as a reference: 34/34, improved from 32/34. This is not a fresh independent audit.
- Prior font-model regressions: 34 passed. New viewport/mapping regressions: 17 passed. Default model and coaching readiness are unchanged.

Latest viewport audit: 27/34 correct end-to-end, three false proposals and two missed cards. With supplied glyph boxes the recognizer gets 34/34. Compact-layout crop quality and wide-layout detection need work. A provider-specific screenshot-coordinate correction was visually checked; independent label review remains pending.

Latest completed work: trained and evaluated independent horizontal/vertical scale augmentation (0.55–1.8), and implemented opt-in bounded card-surface expansion. Broader augmentation produced 39 false proposals on the reused viewport reference and is rejected as a replacement. Bounded expansion with the incumbent detector improved aggregate identities from 27/34 to 30/34, preserving 34/34 old captured and 80/80 old synthetic results, but wide-view recognition regressed from 5/7 to 3/7. It remains research-only. All these comparisons reuse reference images; no fresh audit or coaching-readiness claim is made.

Next work: diagnose wide-frame crop regression and strengthen synthetic responsive-layout/distractor coverage before another detector experiment. A new frozen-model capture audit is required after selecting a candidate. Detailed results and reproduction instructions: `apc/perception/baselines/locator-axis-diversity-v1.experiment.md`. No training job remains running from this checkpoint.

Verification for this checkpoint: 39 targeted regression tests passed (surface refinement, spatial training, detector/reader inference, evaluation and frozen viewport audit checks). Unrelated name-OCR edits were left untouched.

Overall APC: **in development, not coaching-ready**. There is no defensible overall completion percentage yet. Completion still needs independently evaluated full-table perception, temporally consistent state, uncertainty-aware player profiles, solver/GTO and neural-strategy integration, virtual-chip self-play/completed-hand improvement, fast audited BB recommendations, and a verified controlling application. Existing experimental components do not by themselves prove those requirements complete.
