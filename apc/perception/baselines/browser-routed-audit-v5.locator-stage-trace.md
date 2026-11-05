# Locator geometry trace — reused reference, not a fresh audit

The diagnostic wraps existing decoder, refinement and suppression calls, records intermediate proposal geometry, and returns the incumbent outputs unchanged. No inference sources, checkpoints or thresholds were modified. A regression compares proposal outputs before and after tracing. The alternative box-IoU decoder is recorded as a separate control and never fed into production proposals.

Compact preflop target 4c best IoU: default decode 0.3071; refined all views 0.3071; after added-view policy 0.3071; final deduplication 0.3071. It is therefore not a >=0.5-matching proposal removed by added-view filtering or deduplication. Refinement fails to improve the best candidate. The box-IoU decoder control reaches only 0.3547, still below matching threshold; swapping grid peak suppression alone does not demonstrate a fix.

Adjacent 6c: decode best IoU 0.3585, refined 0.9717 and retained thereafter. This shows that low raw IoU can be rescued by surface refinement, but the mechanism does not recover this 4c. Trace totals for this frame: 59 default-decoded proposals over five views, 69 separate control-decoded proposals, 59 refined, 10 after added-view policy, 10 final. Verification later retains one true card. No reason is yet established for why 4c refinement fails; a joint raw-geometry/surface-evidence issue remains plausible, not proven.

Next: construct a deterministic synthetic compact-card placement/spacing sweep with known pixel geometry and varied controls, then measure decode/refinement recall independently. Use that development evidence for locator loss/generator/refinement experiments; never train on audit images. Freeze any changed candidate before a genuinely new varied-scene audit. No coaching readiness or speed claim.

Reproduce:

```powershell
.\.venv\Scripts\python -m apc.perception.trace_locator_stages apc/data/reference/browser-routed-audit-v5/collection.geometry-v2.json --locator-checkpoint apc/checkpoints/card-locator-hardneg-v1.pt --output apc/runs/new-locator-stage-trace.json
```

The destination parent must exist and output must be new. Results are post-review reference diagnostics only. Labels are used for reporting overlaps, not proposal selection.
