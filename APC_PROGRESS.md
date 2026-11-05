# APC progress

Reusable capture orchestration added: `apc/capture/browser_frame.py` calls visible-observation/screenshot callbacks in a fixed before/image/after sequence, retains rejected PNGs and evidence without retries, and saves acquisitions to new folders without overwriting. Controlled callback tests cover scroll races, mutable snapshots, invalid formats, acquisition failure, retained rejection manifests and changed bytes. Actual browser-adapter wiring and end-to-end UI demonstration remain next; no new model accuracy or readiness claim.

Capture-integrity implementation: optional versioned before/after viewport, scroll and card-geometry evidence is now validated by the collection loader, bound to saved image size/hash. New collections can require it; missing evidence or movement then fails before scoring. Seven focused tests passed, plus the preceding 20-test capture/audit regression run. Legacy audit v5 remains provisional and unchanged; no pixel review or accuracy certification is implied. Next implementation step: wire the documented evidence contract into the browser collector, then independently review saved geometry.

Routed audit v5: all nine planned frames are preserved with hashes and draft labels. The frozen scoring run completed; source/model/schedule/disjointness checks passed. Its provisional totals are 24/27 exact, three misses, four false proposals (all four produced high uncalibrated ink candidates), and zero confirmed identities. IMPORTANT: visual review found a vertical capture/label mismatch in h3-preflop-1280.png: recovered DOM y=536.40625 does not match the saved image card top near y=457 (image scale about 0.988). Non-atomic DOM/screenshot capture or scrolling is suspected; the exact cause is not yet verified. Do not interpret its two misses/two false proposals or aggregate recognition as valid accuracy until annotation review. Preserve the raw result unchanged. The wide empty scene independently has two false proposals; the compact preflop has one provisional miss. No audit images may enter training. Next: review every frame's geometry against saved pixels, version any corrections separately, and add an atomic/stability-checked collection workflow before another fresh audit. Temporary viewport override reset; capture server session 50932 stopped; no training job remains active.

Updated: 2026-11-05. This is a saved checkpoint snapshot, not an always-running live dashboard.

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
