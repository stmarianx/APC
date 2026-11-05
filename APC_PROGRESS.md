# APC progress

Updated: 2026-11-05 03:50 UTC. This is a saved checkpoint snapshot, not an always-running live dashboard.

Current experiment — composite negatives and positive-appearance coverage:

`[██████████] 100% — three neural experiments and routed candidate evaluated; fresh routed-stack audit pending`

Current phase: visible-table perception research. Explicit candidate stack: font-diverse glyph model, hard-negative locator, bounded-surface v1 verifier and ambiguous-surface appearance-v4 verifier. Defaults are unchanged.

- Synthetic glyph test: 220/220 correct.
- Captured development set: 34/34 identities, no false detections.
- Older mixed-brightness synthetic set: 80/80, no false detections.
- Earlier audit images, reused as a reference: 34/34, improved from 32/34. This is not a fresh independent audit.
- Prior font-model regressions: 34 passed. New viewport/mapping regressions: 17 passed. Default model and coaching readiness are unchanged.

Latest fresh audit v4: 27/27 visible card appearances correct across three new hands at three sizes, zero misses, but 24 false proposals (four gameplay, 20 on three static empty-scene captures). Seven false proposals produced ink-based identities; two had high unvalidated scores. No confirmed identities emitted. This is one same-application session, not cross-platform readiness. Independent geometry review remains pending.

Latest completed work: trained composite-negative v2, aspect-positive v3 and broader-appearance v4 neural verifiers. V2/v3 reject many captured true cards and are not promoted. V4 uses 14480 synthetic training patches, selects epoch 19 by validation only, and retains 924/924 positive test crops while rejecting 773/774 negatives. V4 alone still loses one captured card. A pixel-evidence route uses incumbent v1 on bounded surfaces and v4 on ambiguous proposals: REUSED audit-v4 recognition remains 27/27 while false proposals fall from three to one, with zero false ink-based identities. Reused viewport-v3 and older development both remain 34/34 with zero false proposals; synthetic retention stays 80/80. New research weights: `apc/checkpoints/card-face-verifier-appearance-v4.pt`. No saved audit images enter training, and confirmed identities remain null.

Next: freeze all four neural checkpoints, inference sources and expert-routing options for a NEW multi-session audit with different negative/control/transition scenes; independently review geometry. The one remaining false proposal and unvalidated role/full-table heads still prevent readiness. Details and reproduction: `apc/perception/baselines/card-face-verifier-appearance-v4.experiment.md`. All training/evaluation/timing processes from this checkpoint have completed; no capture server is running.

Verification for this checkpoint: 53 targeted regression tests passed, covering expanded synthetic generation, sample preservation, audit exclusion, routing/order/provenance, explicit CLI selection and prior frozen audits. Matched warm routed-reader median/p95 approximately 477/1204 ms versus baseline 415/1201 ms, excluding model loading, capture, state and strategy; no speed-improvement or full-coach latency claim. Unrelated name-OCR edits were left untouched.

Overall APC: **in development, not coaching-ready**. There is no defensible overall completion percentage yet. Completion still needs independently evaluated full-table perception, temporally consistent state, uncertainty-aware player profiles, solver/GTO and neural-strategy integration, virtual-chip self-play/completed-hand improvement, fast audited BB recommendations, and a verified controlling application. Existing experimental components do not by themselves prove those requirements complete.
