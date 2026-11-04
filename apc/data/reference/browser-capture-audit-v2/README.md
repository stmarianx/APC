# Frozen-model browser capture audit v2

Six new screenshots collected after freezing detector `8168a060…` and recognizer
`96a95141…`. The plan was written before collection: four streets from the first
hand, then one river frame from each of two further hands. Every planned capture
was retained before evaluation, including the two identity failures.

This is one new browser capture session with three hands, 34 card appearances and
18 distinct identities. It is not three independent capture sessions. The source
is APC's local virtual-chip practice table, with the same application, font, viewport
and layout as the development collection. No external poker client was operated.

Visible card identities were visually checked against the screenshots. Geometry
comes from visible DOM rectangles; independent overlay review is pending. Normalize
using the recorded DOM viewport, not the screenshot's pixel dimensions. Scrolled
views omit part of the table. No temporal timestamps or whole-table annotations
are claimed, and no readiness credit is added.

`use_policy: sealed_audit_no_training` prohibits using these images for training or
fine-tuning. Keep the model fingerprints and this corpus fixed. Future changes
informed by these results need a fresh audit for an untouched evaluation.

The frozen pipeline localized all 34 card appearances without false proposals,
but recognized 32/34 correctly. It confused hero 7s with 7c in the first preflop
frame and board 2s with 2c in the third-hand river frame. Oracle glyph boxes also
scored 32/34, identifying a recognizer-domain weakness rather than a missed card.
No card identities were confirmed for coaching.

Reproduce from the repository root:

```text
python -m apc.perception.evaluate_frozen_browser_audit
```

The evaluator verifies checkpoint hashes, all image hashes, the complete scheduled
capture order, a different session ID, and disjoint image hashes from development.
These checks are reproducibility guards, not proof of cross-platform generalization,
statistical independence between same-hand frames, or calibrated readiness.
