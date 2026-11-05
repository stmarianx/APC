# Audit v6 preregistration — collect only after this plan is saved

Status: collection NOT started. No new v6 frame has been scored. Plan: `browser-fast-text-audit-v6.plan.json`. Four checkpoint hashes and inference/capture/scoring/fixture source hashes are locked before collection. Candidate: wide-surface v2, text-fragment ambiguous verifier v5, bounded verifier v1, fast recovery and locator refinement. No weights/defaults are promoted.

Schedule: three new own-application hands, each preflop/flop/turn/river at 800x900, 560x900 and 1280x900 respectively (12 frames), then three distinct static negatives: text at 560x900, lobby at 800x900, disabled controls at 1280x900. This is one same-application session, not cross-platform or multiple independent environments. Street-sequence capture is not itself complete-hand temporal validation.

Negative routes: `/audit-negative/text`, `/audit-negative/lobby`, `/audit-negative/controls`. They contain no playing cards, do not touch game state, and explicitly identify themselves as static fixtures rather than real gameplay transitions. HTTP integration is tested; visual rendering/capture is still pending. Negative adapter observations include fixture identity as well as viewport/scroll so navigation changes fail endpoint equality.

Collection rules:

1. Recheck live process/tab handles. Start a local practice server only if no intended server is live. No external table is in scope for this audit.
2. Use authorized browser UI for viewport/game actions, refreshing visible state after each action. Only local virtual-chip check/call development actions advance streets.
3. Call the adapter and automatic saver with `usePolicy: 'sealed_audit_no_training'`. Keep native JPEG originals, acquisition manifests and both observations; no transcription of labels or extension-based transcoding.
4. Retain every acquisition. If any frame is unstable/rejected, STOP this schedule's scoring. Preserve evidence and preregister a new collection; never replace samples based on predictions. Reset temporary viewport overrides at handoff/end.
5. Build collection.json from saved evidence with required stability, explicit per-frame dimensions/card sizes/glyph regions and native source hashes. Do not import sealed audits into the training labeling project. JPEG rows are now supported by the collection loader.
6. Review overlays against saved pixels. Before/after equality does not prove pixel correspondence. Any annotation corrections must be versioned with provenance, not silently substituted. Independent review remains separate from developer inspection.
7. Score only once the full planned schedule is preserved. Pass all prior development/capture collections for session/image disjointness. Freeze runner verifies source/checkpoint hashes and stability requirement before loading models. No tuning during collection/scoring.

The next audit is card perception only: it cannot prove seat identity, stacks/pot/actions, temporal consistency, player profiling, solver strategy or full-coach deadlines. Existing best matched saved-frame p95 is ~517ms, still above the 250ms target and excluding capture/state/strategy/UI. Report failures as well as successes. Notify readiness only when the full objective's gates are actually met.
