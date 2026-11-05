# APC progress

Updated: 2026-11-05 03:28 UTC. This is a saved checkpoint snapshot, not an always-running live dashboard.

Current experiment — neural card-face proposal verifier:

`[██████████] 100% — neural verifier trained and compared; explicit research candidate, no default promotion`

Current phase: visible-table perception research. Latest candidate: `apc/checkpoints/glyph-font-diversity-v1.pt`.

- Synthetic glyph test: 220/220 correct.
- Captured development set: 34/34 identities, no false detections.
- Older mixed-brightness synthetic set: 80/80, no false detections.
- Earlier audit images, reused as a reference: 34/34, improved from 32/34. This is not a fresh independent audit.
- Prior font-model regressions: 34 passed. New viewport/mapping regressions: 17 passed. Default model and coaching readiness are unchanged.

Latest fresh audit v4: 27/27 visible card appearances correct across three new hands at three sizes, zero misses, but 24 false proposals (four gameplay, 20 on three static empty-scene captures). Seven false proposals produced ink-based identities; two had high unvalidated scores. No confirmed identities emitted. This is one same-application session, not cross-platform readiness. Independent geometry review remains pending.

Latest completed work: trained a new binary convolutional card-face verifier on 8448 synthetic patches, selected epoch 13 by validation only at a fixed 0.5 threshold. Synthetic test retains 420/420 cards and rejects 549/550 negatives. On REUSED audit-v4 references, exact identities remain 27/27 while false proposals fall from 24 to three; one high-score false candidate remains. Reused viewport-v3 improves to 34/34 with zero false proposals; earlier 34/34 captured and 80/80 synthetic checks are preserved. No saved audit images enter training. Checkpoint: `apc/checkpoints/card-face-verifier-v1.pt` (117381 bytes). CLI inspection supports explicit selection; defaults and confirmed-identity gates stay unchanged.

Next work: characterize remaining false crops and held-out negative styles, improve discrimination without losing true cards, independently review geometry, and freeze the full candidate configuration for NEW capture validation. Keep all audit frames out of training. Detailed results and reproduction instructions: `apc/perception/baselines/card-face-verifier-v1.experiment.md`. No training, evaluation or capture server remains running from this checkpoint.

Verification for this checkpoint: 47 targeted regression tests passed, including audit-training exclusion, neural filter/reader integration, dataset determinism, fixed threshold, test separation, CLI selection and prior frozen audits. Matched warm saved-frame timing: verifier median/p95 approximately 434/1088 ms versus baseline 400/1129 ms; no speed-improvement or full-coach latency claim. Unrelated name-OCR edits were left untouched.

Overall APC: **in development, not coaching-ready**. There is no defensible overall completion percentage yet. Completion still needs independently evaluated full-table perception, temporally consistent state, uncertainty-aware player profiles, solver/GTO and neural-strategy integration, virtual-chip self-play/completed-hand improvement, fast audited BB recommendations, and a verified controlling application. Existing experimental components do not by themselves prove those requirements complete.
