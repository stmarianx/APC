# APC progress

Updated: 2026-11-05 03:20 UTC. This is a saved checkpoint snapshot, not an always-running live dashboard.

Current experiment — fresh multiview/negative-frame audit:

`[██████████] 100% — fresh audit complete; false-positive/confidence failures recorded; no promotion`

Current phase: visible-table perception research. Latest candidate: `apc/checkpoints/glyph-font-diversity-v1.pt`.

- Synthetic glyph test: 220/220 correct.
- Captured development set: 34/34 identities, no false detections.
- Older mixed-brightness synthetic set: 80/80, no false detections.
- Earlier audit images, reused as a reference: 34/34, improved from 32/34. This is not a fresh independent audit.
- Prior font-model regressions: 34 passed. New viewport/mapping regressions: 17 passed. Default model and coaching readiness are unchanged.

Latest fresh audit v4: 27/27 visible card appearances correct across three new hands at three sizes, zero misses, but 24 false proposals (four gameplay, 20 on three static empty-scene captures). Seven false proposals produced ink-based identities; two had high unvalidated scores. No confirmed identities emitted. This is one same-application session, not cross-platform readiness. Independent geometry review remains pending.

Latest completed work: froze weights/options/inference-source bytes before a planned nine-frame capture session; collected and retained all six gameplay and three explicit static negative screenshots. Added empty-frame scoring and frozen-configuration/source checks. Computer-use capture was limited to APC's own virtual-chip table/fixture; browser sizing restored and temporary server stopped. No neural training, weight changes or default promotion occurred. Earlier candidate retains 34/34 reused viewport identities and 80/80 synthetic retention, but those results do not override the fresh negative-scene failures.

Next work: generate separate synthetic card-vs-controls/background/blank examples and train/evaluate proposal discrimination, including high-score errors. Keep all audit images excluded from training. Independently review geometry and measure matched latency; a new candidate will need another fresh audit. Detailed results and reproduction instructions: `apc/data/reference/browser-multiview-audit-v4/README.md`. No training or capture server remains running from this checkpoint.

Verification for this checkpoint: 40 targeted regression tests passed, covering frozen sources/options, complete capture schedule, empty-frame false positives, browser/table fixture, inference and application compatibility. Fresh observed one-pass p95 approximately 919 ms over nine frames, not a matched benchmark or full-coach latency gate. Unrelated name-OCR edits were left untouched.

Overall APC: **in development, not coaching-ready**. There is no defensible overall completion percentage yet. Completion still needs independently evaluated full-table perception, temporally consistent state, uncertainty-aware player profiles, solver/GTO and neural-strategy integration, virtual-chip self-play/completed-hand improvement, fast audited BB recommendations, and a verified controlling application. Existing experimental components do not by themselves prove those requirements complete.
