# Stage attribution on reused corrected-label reference

`diagnose_card_pipeline.py` traces the same locator/refinement/deduplication/tiled-view output through the same routed verifiers, then glyph recognition. It does not alter thresholds, models, defaults or inference source files frozen by audit v5. IoU >=0.5 uses the existing one-to-one assignment. Proposal IDs are diagnostic metadata only. Outputs include image, checkpoint and collection fingerprints/provenance, false-proposal boxes and verifier routes/scores. This is reference reuse, not a fresh audit or training input.

Result: 26 correct, one locator-or-refinement miss. On compact preflop the 4c has no >=0.5-overlap proposal even before learned verification. The 6c has IoU approximately 0.972 and is recognized. Thus lowering the verifier threshold cannot restore the absent 4c proposal. The trace does not yet distinguish raw detector decode, surface refinement, tiled-view policy or deduplication as the root cause; the stage name deliberately groups those operations.

Wide negative fixture: two false proposals survive ambiguous-surface appearance-v4 verification, scores approximately 0.785 and 0.574 at fixed 0.5. Both have `adaptive_no_bounded_surface`, source view 0, candidates 7c/8h. These are card-like crops within the lower-left panel/control region, not genuine cards. Scores are uncalibrated; no confirmed cards or player advice are emitted. Raising a threshold based on this reference would not establish generalization and could lose true cards.

Next experiments must use separate synthetic/development data: vary compact adjacent card spacing/placement and introduce narrow text/control subcrops as negatives. Trace detector decode -> refinement -> suppression before changing recall behavior. Keep audit images excluded from training. Preserve the frozen baseline and preregister any new candidate before new varied-scene evaluation. Full-table state, roles, strategy, calibration and latency readiness remain unfinished.

Reproduce with explicit four checkpoint arguments:

```powershell
.\.venv\Scripts\python -m apc.perception.diagnose_card_pipeline apc/data/reference/browser-routed-audit-v5/collection.geometry-v2.json --glyph-checkpoint apc/checkpoints/glyph-font-diversity-v1.pt --locator-checkpoint apc/checkpoints/card-locator-hardneg-v1.pt --verifier-checkpoint apc/checkpoints/card-face-verifier-appearance-v4.pt --bounded-verifier-checkpoint apc/checkpoints/card-face-verifier-v1.pt --output apc/runs/new-pipeline-diagnostic.json
```

Output files are created exclusively, never overwritten. Capture labels enter matching/reporting only, never model inference or routing.
