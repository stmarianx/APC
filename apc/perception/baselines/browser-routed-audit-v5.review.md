# Routed audit v5 — provisional, geometry review required

All nine preregistered frames were retained, including failures. Four checkpoint hashes and seven inference-source hashes matched the frozen plan. No training occurred. Scope: one same-application session, three hands, three resized views of one static negative fixture; not cross-platform, multi-session or transition validation.

Raw result: `browser-routed-audit-v5.result.json`. Its 24/27 exact, three misses and four false proposals are **not validated accuracy**. Saved image `h3-preflop-1280.png` visibly places the cards near image y=457 while its recovered DOM labels use y=536.40625 (expected image y approximately 530). The DOM read and screenshot were not atomic; scroll movement is a suspected cause, not established. The raw labels/result remain preserved as evidence. Do not silently replace the screenshot or claim the resulting two misses/two false proposals are model failures.

Other raw diagnostics: compact preflop 1/2, other gameplay frames exact; wide empty fixture has two false proposals with high uncalibrated ink scores; the other two empty views have none. All confirmed identities remain null. Independent geometry review is still needed for all frames. No readiness credit or default-model promotion.

Next steps:

1. Review image-space card/glyph regions against every saved PNG. Record corrected labels in a separate version with provenance; preserve original labels and scoring output.
2. Add capture stability checks (layout/scroll before and after screenshot), recording screenshot dimensions and observed geometry together. Do not select samples based on model predictions.
3. Analyze verifier false positives using separate development examples; keep this audit excluded from training.
4. Freeze a subsequent candidate before fresh independent captures with varied negative and transition scenes.

Collection path: `apc/data/reference/browser-routed-audit-v5/collection.json`. Screenshot hashes are enforced by the collection loader. Capture server stopped and viewport reset after collection.
