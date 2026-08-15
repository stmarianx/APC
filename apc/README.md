# APC — AI Poker Coach

APC is the platform-agnostic model layer of this project. It learns poker
concepts from the verified Negreanu course corpus and solver/GTO material,
reads visible virtual-chip poker tables, maintains a temporally valid hand
state, models opponents with uncertainty, and returns fast recommendations in
big blinds.

APC is designed for locally controlled or explicitly permitted virtual-chip
training environments. A PokerStars text-history parser remains available as
one optional source adapter; it is not APC's visual or product boundary.

## System boundary

```text
course evidence + poker math + solver data
                   |
visible frame -> perception -> temporal state -> strategy -> BB recommendation
                        |              |              |
                 confidence gate   player map    provenance audit
                                           |
                                  virtual-chip replay/self-play
                                           |
                                  evaluated checkpoint promotion
```

The initial implementation reuses the existing `coach/` domain engine:

- `visual_capture.py` accepts calibrated, confidence-bearing observations;
- `state_transition.py` rejects impossible temporal changes;
- `live_state.py` normalizes a table state and matches covered solver nodes;
- `profiles.py` and `exploit.py` maintain uncertainty-aware player evidence;
- `solutions.py` and `strategy_selection.py` preserve solver provenance and
  safe fallback behavior.

The synthetic pixel-to-observation provider now covers cards, hero/dealer
seats, pot and call price, integer stacks, and paired action events. It remains
a closed-vocabulary bootstrap; controlled visible-table data and calibrated
models are still required before APC can be trained for arbitrary tables.

## Files

- `APC_MODEL_SPEC.md` — authoritative product, model and training contract.
- `schemas/frame_annotation.schema.json` — one labeled visual frame/sequence item.
- `schemas/dataset_manifest.schema.json` — grouped dataset and split manifest.
- `curriculum/generated/course_curriculum.jsonl` — deterministic course and math training curriculum.
- `curriculum/generated/curriculum_manifest.json` — source hashes, grouped splits, counts and curriculum fingerprint.
- `readiness.json` — current evidence-backed gate status.
- `PORTABILITY_AND_RELEASE.md` — reproducible machine-transfer, artifact and publication policy.
- `tools/artifact_manifest.py` — SHA-256 manifest creation/verification for datasets, checkpoints and run reports moved between machines.
- `perception/baseline.py` — reproducible pixel smoke trainer and held-out evaluator.
- `perception/card_baseline.py` — learned card-slot geometry plus rank/suit pixel heads.
- `perception/table_state_baseline.py` — hero/dealer geometry and closed-vocabulary pot/call-price heads.
- `perception/stack_baseline.py` — segmented renderer-v2 numeric-token OCR for integer, decimal and zero BB stacks, with legacy checkpoint compatibility.
- `perception/event_baseline.py` — paired-frame actor/action/BB-amount event smoke head.
- `perception/boundary_baseline.py` — adjacent-frame hand-reset detector.
- `perception/composite.py` — one BB-only visible-state result with forced abstention on missing fields.
- `perception/temporal_composite.py` — paired-frame state reconstruction with chip/pot conservation, heads-up effective stack and evidence-gated history completeness.
- `perception/evaluate_temporal.py` — leakage-checked held-out evaluator for the complete temporal perception path.
- `perception/hand_tracker.py` — stateful hand IDs, cross-street continuity, complete histories and identity attachment gate.
- `perception/viewport.py` — fingerprinted affine mapping from a detected or manually verified table box into canonical perception geometry.
- `perception/evaluate_hand_tracker.py` — full multi-hand perception/tracker evaluator.
- `player_identity.py` — persistent uncertainty-aware identity registry with collision and duplicate-frame protection.
- `evaluate_identity_registry.py` — labeled-name association evaluator; explicitly does not claim visual OCR.
- `visual_identity_signature.py` — normalized name-band pixel signatures that support pseudonymous profile continuity without claiming readable OCR.
- `evaluate_visual_identity.py` — held-out stability, collision and registry-resolution evaluation for visual signatures.
- `backend_adapter.py` — strict BB-only bridge from complete tracked hands to the coaching core's confidence-bearing visual-observation contract; ambiguous multiway stacks and raise semantics abstain.
- `deadline.py` — deadline budget, adaptive strategy-tier selection, safe fallback and stale/illegal/duplicate controlled-action authorization.
- `evaluate_coaching_bridge.py` — full synthetic pixels-to-identity-to-observation-to-backend interoperability audit.
- `tools/evaluate_partial_reference.py` — frozen partial-ground-truth OOD comparison with safe-abstention checks.
- `data/reference/controlled-reference-v1/` — one immutable user-provided virtual-chip reference explicitly excluded from training and gate counts.
- `synthetic/render_hand_sequences.py` — coherent multi-street, multi-hand sequence renderer.
- `tools/validate_sequences.py` — hand/street/card/history/stack/pot/identity continuity auditor.
- `perception/baselines/synthetic_smoke_v2.model_card.json` — versioned smoke evidence and limitations.
- `perception/baselines/synthetic_audit_v1.model_card.json` — frozen, one-pass synthetic audit evidence.
- `perception/baselines/synthetic_state_v1.model_card.json` — renderer-v2 table-state and composite evidence.
- `perception/baselines/synthetic_stack_event_v1.model_card.json` — frozen stack and temporal-event audits, including failed development history.
- `perception/baselines/synthetic_temporal_composite_v1.model_card.json` — exact held-out validation evidence, integration failures and fresh-test requirement.
- `perception/baselines/synthetic_hand_sequence_v1.model_card.json` — preserved historical complete-hand validation and untouched 50% test failure.
- `perception/baselines/synthetic_hand_sequence_gate_v2.model_card.json` — fresh 2,000-frame model family, exact 24-hand synthetic test, real OOD failure and latency limits.
- `perception/baselines/synthetic_hand_sequence_latency_cache_v1.model_card.json` — matched 24-hand cache optimization audit with unchanged predictions and bounded-memory latency evidence.
- `perception/baselines/synthetic_visual_identity_stack_context_v1.model_card.json` — post-test development evidence for visual signatures, explicit multiway stack context and the gated backend bridge.

## Annotation workflow

Create a local project for one controlled virtual-chip table configuration:

```powershell
python -m apc.annotator.cli init .\apc-data\first-table `
  --project-id first-table-v1 `
  --provider controlled-table `
  --layout six-max `
  --theme dark-four-color `
  --max-seats 6
```

Capture an explicit table rectangle as a read-only 250-frame session:

```powershell
python -m apc.capture.screen_capture C:\captures\controlled-001 `
  --session controlled-001 --region 100,80,1700,980 `
  --frames 250 --interval-ms 200
```

The coordinates are `left,top,right,bottom` across the available displays.
Capture requires an explicit region, records no keyboard or mouse input, performs
no network transmission, and writes `capture_plan.json`, fingerprinted PNGs and
`capture_report.json`. Run a second independent session in a new destination;
the example coordinates are placeholders and must be replaced with the actual
visible table bounds.

For an already captured frame, the automatic locator produces an advisory
normalized table box or an explicit abstention:

```powershell
python -m apc.perception.table_locator C:\captures\controlled-001\frame-000000-000000000ms.png `
  --output .\apc\runs\controlled-001\table-locator.json
```

The current locator is a border/center-contrast geometry baseline. Its frozen
synthetic test detected 192/192 frames with mean IoU 0.841, and its one-pass
real OOD reference reached IoU 0.818. These are development results only:
confidence is capped, uncalibrated and cannot automatically activate viewport
normalization or coaching before controlled-visible evaluation.

Import PNG frames as an immutable capture session:

```powershell
python -m apc.annotator.cli import .\apc-data\first-table `
  C:\frames\frame-001.png C:\frames\frame-002.png `
  --session session-001 --timestamp-ms 0
```

For a recorded folder, import the whole ordered session with natural filename
sorting, a declared capture interval and optional subsampling:

```powershell
python -m apc.annotator.cli import-folder .\apc-data\first-table `
  C:\captures\controlled-001 --session controlled-001 `
  --timestamp-ms 0 --interval-ms 100 --sample-every 1
```

The importer fingerprints and deduplicates PNGs, enforces increasing timestamps
inside each session, and leaves every new frame unverified until annotation
review. Recursive folder scanning is opt-in with `--recursive`.

Generate non-destructive model suggestions to accelerate review:

```powershell
python -m apc.annotator.suggestions .\apc-data\first-table `
  --base-checkpoint .\apc\checkpoints\synthetic-handseq-dev-v1-base.json `
  --card-checkpoint .\apc\checkpoints\synthetic-handseq-dev-v1-card.json `
  --table-state-checkpoint .\apc\checkpoints\synthetic-handseq-dev-v1-table-v5.json `
  --stack-checkpoint .\apc\checkpoints\synthetic-handseq-dev-v1-stack.json
```

Suggestions are stored separately from annotations, fingerprinted against the
immutable source frame, and always marked `review_required`. They are never
auto-applied, never treated as verified labels, and cannot open the training
gate without human review. The workbench displays the suggestion, raw confidence,
checkpoint provenance and abstentions; its explicit apply button only updates an
unsaved draft, resets `verified` to false and records the suggestion fingerprint.

After one frame in a capture session has a verified annotation, reuse only its
stable normalized geometry as review-only drafts for later frames in that same
session:

```powershell
python -m apc.annotator.propagation .\apc-data\first-table VERIFIED_SAMPLE_ID
```

This propagates table, seat, visible card-slot, pot and action-button boxes plus
the stable Hero seat. It deliberately omits cards, stacks, pot values, dealer,
actions, occupancy and player names. Existing annotations and suggestions are
not replaced unless explicitly requested, and propagated drafts never count as
verified frames.

Run the local workbench:

```powershell
python -m apc.annotator.web .\apc-data\first-table --open
```

The workbench previews each frame and validates the canonical JSON annotation
before saving. When at least three capture sessions are completely verified,
export a group-exclusive manifest:

```powershell
python -m apc.annotator.cli export .\apc-data\first-table --version 0.1.0
```

Enforce the full visible-table training minimum before starting a promotable run:

```powershell
python -m apc.tools.validate_dataset `
  .\apc-data\first-table\dataset_manifest.json `
  --require-ready --output .\apc-data\first-table\gate-t-report.json
```

The command returns 0 only for a valid manifest that passes every Gate T
minimum, 1 for an invalid dataset, and 3 for a valid but undersized dataset.
Image verification remains enabled; `--skip-images` is structural diagnosis only
and must not be used for a promotion decision.

After exporting the controlled-visible project, merge it with the synthetic
complete-hand corpus and run Gate T on the combined manifest:

```powershell
python -m apc.tools.merge_datasets `
  .\apc-data\visible-training-combined `
  .\apc\data\processed\synthetic-handseq-gate-v2\dataset_manifest.json `
  .\apc-data\controlled-visible\dataset_manifest.json `
  --dataset-id apc-visible-training-v1 --dataset-version 1.0.0 `
  --report .\apc\runs\visible-training-v1\merge-report.json

python -m apc.tools.validate_dataset `
  .\apc-data\visible-training-combined\dataset_manifest.json --require-ready
```

The merger validates every source image, namespaces dataset/session/table/hand
identities, preserves source fingerprints in provenance, rejects repeated frame
digests, and exports a fresh group-exclusive split. Source ordering does not
change the resulting data fingerprints.

The workbench draws all current labels over the responsive source image and
supports drag-to-create normalized regions for tables, seats, cards, pots and
action buttons, and the visible turn clock. Object-specific fields are merged into the canonical target,
and a synchronization helper derives hero/dealer seats, pot, street and legal
actions before validation. Verified-source, same-session layout propagation is
now available; OCR-assisted numeric and name labeling remains the next
usability improvement.

Raw frames, processed tensors, checkpoints and run outputs are intentionally
ignored by version control. Reproducible manifests, annotations, metrics and
model cards remain versioned.

## Synthetic bootstrap and perception smoke test

Generate a controlled, exactly labeled bootstrap set with three layouts and
two themes:

```powershell
python -m apc.synthetic.render_table .\apc\data\processed\synthetic-v1 `
  --sessions 9 --seed 20260802
```

Fit the smoke baseline on manifest-declared training sessions and evaluate a
held-out session:

```powershell
python -m apc.perception.baseline train `
  .\apc\data\processed\synthetic-v1\dataset_manifest.json `
  --checkpoint .\apc\checkpoints\synthetic-v1.json --seed 20260802

python -m apc.perception.baseline evaluate `
  .\apc\checkpoints\synthetic-v1.json `
  .\apc\data\processed\synthetic-v1\dataset_manifest.json `
  --split test --output .\apc\runs\synthetic-v1\test_metrics.json
```

The evaluator rejects any held-out split containing a training session. The
smoke model reads pixels, but it is deliberately non-promotable and supports
only layout, theme, street and visible legal-action classification. Gate T
also requires at least 500 verified frames from two controlled visible-table
sessions; synthetic scale alone can never open the gate.

Generate paired temporal events and evaluate actor, action and BB amount:

```powershell
python -m apc.synthetic.render_events .\apc\data\processed\synthetic-events-v1 `
  --sessions 18 --seed 764321

python -m apc.perception.event_baseline train `
  .\apc\data\processed\synthetic-events-v1\dataset_manifest.json `
  --checkpoint .\apc\checkpoints\synthetic-events-v1-event.json --seed 764321

python -m apc.perception.event_baseline evaluate `
  .\apc\checkpoints\synthetic-events-v1-event.json `
  .\apc\data\processed\synthetic-events-v1\dataset_manifest.json `
  --split test `
  --base-checkpoint .\apc\checkpoints\synthetic-state-dev-v1-base.json `
  --stack-checkpoint .\apc\checkpoints\synthetic-state-dev-v1-stack-v1.json `
  --output .\apc\runs\synthetic-events-v1\event_test.json
```

This temporal baseline deliberately assumes a fixed event banner. Its audit
proves the paired-frame state-transition machinery, not general screen
recognition.

Run the complete paired-frame evaluator after training compatible table-state,
stack and event heads:

```powershell
python -m apc.perception.evaluate_temporal `
  .\apc\data\processed\synthetic-events-v1\dataset_manifest.json `
  --base-checkpoint .\apc\checkpoints\synthetic-state-dev-v1-base.json `
  --card-checkpoint .\apc\checkpoints\synthetic-state-dev-v1-card.json `
  --table-state-checkpoint .\apc\checkpoints\synthetic-events-v1-table.json `
  --stack-checkpoint .\apc\checkpoints\synthetic-events-v1-stack.json `
  --event-checkpoint .\apc\checkpoints\synthetic-events-v1-event.json `
  --split validation --output .\apc\runs\synthetic-events-v1\temporal-validation.json
```

The temporal composite rejects mismatched cards, seats, actor-stack deltas,
pot deltas and invalid all-in residual stacks. It does not call a pair-local
history complete, does not collapse multiway effective stacks to a misleading
scalar, and never substitutes seat aliases for recognized player identities.

## Complete-hand sequence workflow

Generate and audit coherent hands with carried stacks, cumulative histories,
stable player labels and explicit boundary labels:

```powershell
python -m apc.synthetic.render_hand_sequences `
  .\apc\data\processed\hand-sequences-v1 `
  --sessions 12 --hands-per-session 2 --seed 8675309

python -m apc.tools.validate_sequences `
  .\apc\data\processed\hand-sequences-v1\dataset_manifest.json `
  --output .\apc\runs\hand-sequences-v1\sequence-audit.json
```

Large generation runs use session-isolated seeds and can be safely chunked and
resumed without regenerating completed sessions:

```powershell
python -m apc.synthetic.render_hand_sequences `
  .\apc\data\processed\hand-sequences-large-v2 `
  --sessions 125 --hands-per-session 2 --seed 2026080302 --session-limit 25

python -m apc.synthetic.render_hand_sequences `
  .\apc\data\processed\hand-sequences-large-v2 `
  --sessions 125 --hands-per-session 2 --seed 2026080302 `
  --resume --session-limit 25
```

The immutable `generation_plan.json` rejects parameter drift on resume. A
manifest is exported only when all planned sessions are complete.

The 2,000-frame `synthetic-handseq-gate-v2` corpus passes full frame validation
and complete-hand continuity for 250/250 hands. This satisfies the synthetic
scale, temporal, layout and theme portions of Gate T, but it does not substitute
for the required 500 controlled-visible frames across two sessions.

The stateful evaluator combines base, card, segmented numeric, stack, event and
boundary checkpoints. It generates internal hand IDs because many visible
tables do not expose a client hand number. The historical two-hand checkpoint
failed one untouched hand. Its replacement, trained on the 2,000-frame corpus,
passed 96/96 fresh test transitions and 24/24 complete hands exactly. It remains
non-promotable because the normalized real reference is still only 1/6 exact,
player identities remain unresolved. The frozen model card retains its original
1.322-second p95. A later matched cold benchmark on this laptop reduced p95
from 1.103 seconds to 0.444 seconds using bounded shared image/feature caches,
with the exact prediction fingerprint unchanged; controlled-visible latency and
the 250 ms perception target remain open.

`player_identity.py` can persist confidence-bearing name candidates and attach
unique resolved identities to tracked seats. Its present evaluation uses
verified annotation strings; visual player-name OCR remains a required head.

The composite now also extracts an uncalibrated, pseudonymous signature from
each learned seat name band. On the held-out synthetic validation session,
144 observations covering nine players were stable with no token collisions,
and all nine seat profiles resolved after repeated frames. This preserves a
player map even before text OCR, but it does not reveal or claim to read the
username. Controlled-visible appearance changes remain unevaluated.

## Coaching-core bridge

`backend_adapter.py` maps occupied seats clockwise from the visible dealer,
normalizes structured actions, carries checkpoint and identity evidence, and
builds the exact observation payload accepted by `coach/`. A cross-package
dry run reaches the sample solver node exactly. The result remains explicitly
`observation_ready_uncalibrated` with `recommendation_allowed=false`; passing
the software contract is not treated as perception calibration.

For multiway states, APC preserves one Hero-versus-opponent effective stack in
BB for every active opponent. It derives a scalar automatically only when one
opponent remains. With multiple opponents, the default is abstention unless a
solver bundle explicitly declares the `minimum_active_opponent` reduction.

The end-to-end bridge audit processed eight held-out validation transitions.
After repeated identity evidence, three states crossed the visual contract and
reached the backend; all were expectedly unmatched because the fixture solver
contains heads-up nodes only. Three histories with contribution-only raises
abstained instead of inventing a raise-to size. No row allowed an uncalibrated
recommendation.

## Frozen visible-table OOD reference

One previously supplied virtual-chip screenshot is retained as an immutable,
partial-ground-truth OOD reference. The synthetic checkpoint matched only the
flop street (1/6 verified fields); it missed the six-max layout, Hero's Ah Qh,
the 2s 8d 5s board, the 62.4 BB pot and the six visible seat slots. The first
pass exposed an optional identity-head exception; the composite now converts
that failure into an abstention and rejects duplicate-card output before any
backend handoff. The frame is never used for training, tuning or gate counts.

A one-pass manual table-box profile then normalized the same frozen frame into
canonical geometry. This did not improve semantic accuracy (still 1/6): the
numeric table-state head rejected an invalid token and the composite returned
`abstain_perception_head_failure`. Reusing that frame as a temporal pair also
returned `abstain_frame_perception_rejected` without invoking the event head.
This proves the viewport contract and failure isolation, not generalization.
