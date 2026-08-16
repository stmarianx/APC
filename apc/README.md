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
seats, pot and call price, stacks, player names, turn clocks and paired action events. It remains
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
- `perception/turn_clock_baseline.py` — segmented synthetic countdown OCR producing canonical remaining milliseconds.
- `perception/name_ocr_baseline.py` — batched character-level synthetic player-name OCR feeding the uncertainty-aware identity registry.
- `perception/card_baseline.py` — learned card-slot geometry plus rank/suit pixel heads.
- `perception/table_state_baseline.py` — hero/dealer geometry and closed-vocabulary pot/call-price heads.
- `perception/stack_baseline.py` — segmented renderer-v2 numeric-token OCR for integer, decimal and zero BB stacks, with legacy checkpoint compatibility.
- `perception/event_baseline.py` — paired-frame actor/action/BB-amount event smoke head.
- `perception/boundary_baseline.py` — adjacent-frame hand-reset detector.
- `perception/composite.py` — one BB-only visible-state result with forced abstention on missing fields.
- `perception/temporal_composite.py` — paired-frame state reconstruction with chip/pot conservation, heads-up effective stack and evidence-gated history completeness.
- `perception/evaluate_temporal.py` — leakage-checked held-out evaluator for the complete temporal perception path.
- `perception/evaluate_realtime_composite.py` — separate-corpus joint field, per-head latency and visible-deadline evaluator.
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
- `perception/baselines/synthetic_realtime_composite_v1.model_card.json` — integrated 13-field accuracy, latency and deadline-routing audit.

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

Generate a clock-labeled synthetic audit set, fit the segmented countdown head,
and evaluate it on group-exclusive sessions:

```powershell
python -m apc.synthetic.render_table .\apc\data\processed\synthetic-turn-clock-v2 `
  --sessions 42 --seed 2026081502 --include-turn-clock

python -m apc.perception.turn_clock_baseline train `
  .\apc\data\processed\synthetic-turn-clock-v2\dataset_manifest.json `
  --checkpoint .\apc\checkpoints\synthetic-turn-clock-v2.json --seed 2026081501

python -m apc.perception.turn_clock_baseline evaluate `
  .\apc\checkpoints\synthetic-turn-clock-v2.json `
  .\apc\data\processed\synthetic-turn-clock-v2\dataset_manifest.json `
  --split test --output .\apc\runs\synthetic-turn-clock-v2\test.json
```

The untouched synthetic test reads 16/16 countdowns exactly with 0 ms mean
absolute error and 62.16 ms p95 on this laptop. The optional composite head
publishes canonical remaining milliseconds and visible-timer provenance. This
is fixed-font synthetic evidence only; real timer formats, occlusion,
animations, calibration and deadline-source auditing remain open.

Generate varied stable player names, train the character head and audit both
unseen whole-name decoding and repeated-frame identity resolution:

```powershell
python -m apc.synthetic.render_table .\apc\data\processed\synthetic-name-ocr-v1 `
  --sessions 42 --seed 2026081601 --include-name-ocr

python -m apc.perception.name_ocr_baseline train `
  .\apc\data\processed\synthetic-name-ocr-v1\dataset_manifest.json `
  --checkpoint .\apc\checkpoints\synthetic-name-ocr-v1.json --seed 2026081601

python -m apc.perception.name_ocr_baseline evaluate `
  .\apc\checkpoints\synthetic-name-ocr-v1.json `
  .\apc\data\processed\synthetic-name-ocr-v1\dataset_manifest.json `
  --split test --output .\apc\runs\synthetic-name-ocr-v1\test.json
```

The group-exclusive held-out synthetic test reconstructs 64/64 previously unseen whole names
and every character exactly, then resolves all identities after repeated frames.
Single-decode batched inference reduced p95 from 608.54 ms to 49.79 ms without
changing the prediction fingerprint. The validation identity rate is lower
(89.66%) despite exact text because its uncalibrated confidence gate correctly
retains uncertain names. Arbitrary real fonts, lengths, Unicode and occlusion
remain controlled-visible requirements.

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

Run the full visible-state and deadline audit on a separately generated corpus
that includes both player names and turn clocks:

```powershell
python -m apc.synthetic.render_table .\apc\data\processed\synthetic-realtime-v1 `
  --sessions 42 --seed 2026081602 --include-turn-clock --include-name-ocr

python -m apc.perception.evaluate_realtime_composite `
  .\apc\data\processed\synthetic-realtime-v1\dataset_manifest.json `
  --base-checkpoint .\apc\checkpoints\synthetic-handseq-gate-v2-base.json `
  --card-checkpoint .\apc\checkpoints\synthetic-handseq-gate-v2-card.json `
  --table-state-checkpoint .\apc\checkpoints\synthetic-handseq-gate-v2-table-exemplar.json `
  --stack-checkpoint .\apc\checkpoints\synthetic-handseq-gate-v2-stack.json `
  --turn-clock-checkpoint .\apc\checkpoints\synthetic-turn-clock-v2.json `
  --name-ocr-checkpoint .\apc\checkpoints\synthetic-name-ocr-v1.json `
  --split test --output .\apc\runs\synthetic-realtime-v1\test.json
```

The final separately generated audit reached 100% joint supported-state
accuracy, 188.38 ms validation p95 and 171.34 ms held-out-test p95. Measured
latency is charged against each visible countdown; all 32 frames retained a
bounded-refinement budget and none entered an unsafe/expired state. Batched
numeric/card inference and removal of redundant identity work cut the original
393.79/333.30 ms validation/test p95 values. Threaded heads were measured and
rejected because contention was slower on this laptop. These figures remain
synthetic-only and exclude solver and actuation latency.

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

`recommendation.py` closes the strategy-to-output contract after a backend
solver match. It conditionally normalizes the legal mixed strategy, samples it
reproducibly from a caller-supplied key, converts solver bet fractions and
raise-to amounts into explicit BB commands, and preserves node, solver, match,
latency and deadline provenance. Closed calibration/environment gates,
unmatched states, low-confidence matches and expired deadlines abstain or use
the declared non-GTO safe fallback. Every result sets
`actuation_authorized=false` so execution remains a separate, audited concern.

Run the fixture regression with:

```powershell
python -m apc.evaluate_recommendations coach/examples/sample_solver_export.csv `
  --output apc/runs/recommendation-v1/regression.json
```

The frozen four-node regression produced 4/4 deterministic, BB-only,
provenance-complete recommendations, 4/4 closed-gate abstentions, zero
actuation-authorization violations and 0.681 ms p95 recommendation latency on
this laptop. It starts from exact backend states and therefore does not satisfy
the controlled-visible end-to-end or solver-coverage gates.

## Solver coverage audit

`evaluate_solver_coverage.py` measures an imported solver bundle against Hero
decision contexts reconstructed from completed hands. Its provider-independent
core accepts parsed hand objects; the initial command-line intake uses the
currently supported text hand-history parser. It reports matcher confidence,
observed-action coverage, street/player-count/position slices, and the first
structural exclusion reason for every unmatched decision.

```powershell
python -m apc.evaluate_solver_coverage `
  coach/examples/sample_solver_bundle.json `
  coach/examples/sample_play_money_hand.txt `
  --output apc/runs/solver-coverage-v1/audit.json
```

The frozen fixture audit is valid but fails promotion: only 1/3 Hero decisions
has an exact matching node and covered observed action. The preflop decision
has no three-player node; the turn decision has no suit-isomorphic card node.
The declared expansion gate requires at least 100 decisions and at least 80%
exact state coverage. This small fixture measures the gap; it does not satisfy
that gate or visible-perception evaluation.

## Completed-hand replay dataset

`self_learning/replay_dataset.py` starts Gate S with an immutable dataset
contract for exact solver-matched completed-hand decisions. Each BB-only row
contains the canonical state, imported mixed-strategy target, observed action
and EV-loss feedback when covered, solver provenance, a hand-group split and
its own content fingerprint. Imported targets remain `gto_verified=false`.

```powershell
python -m apc.self_learning.replay_dataset build `
  coach/examples/sample_solver_bundle.json `
  coach/examples/sample_play_money_hand.txt `
  apc/runs/replay-fixture-v1/dataset `
  --dataset-id replay-fixture-v1

python -m apc.self_learning.replay_dataset validate `
  apc/runs/replay-fixture-v1/dataset
```

The builder refuses an existing destination, fingerprints source files,
detects example or manifest tampering, and assigns all examples from a hand to
one deterministic split. The first artifact has one exact training example
from three decisions and deliberately remains `training_eligible=false`: it has
no validation/test groups and therefore cannot enter the candidate trainer;
paired promotion evaluation and checkpoint rollback evidence also remain open.

Replay manifests now compute training eligibility from declared minimum example
and hand-group counts plus non-empty train/validation/test splits. The candidate
trainer in `self_learning/train_candidate.py` accepts only a manifest that
passes those checks. It learns deterministic hashed structured-state features
against legal mixed-strategy targets, evaluates cross-entropy, L1 error,
top-action agreement and top-action regret in BB on every split, and writes a
fingerprinted checkpoint with activation and incumbent replacement disabled.

`self_learning/evaluate_candidate_smoke.py` exercises the entire pipeline using
60 distinct hand IDs cloned from one fixture. It produced 34/8/18 grouped
train/validation/test rows and a deterministic checkpoint. All rows represent
the same underlying poker state, so its perfect top-action agreement and zero
top-action regret demonstrate plumbing only—not generalization or promotion.

## Controlled virtual-chip decision table

`virtual_table.py` provides a platform-independent, internal-only virtual-chip
decision episode for every imported solver node. Policies receive canonical BB
state and legal actions without seeing the EV oracle. One legal action produces
terminal, fingerprinted feedback containing the explicit BB command, imported
EV, best available EV and regret. Illegal actions, duplicate terminal steps,
ambiguous calls and sizes beyond the effective stack are rejected. The provider
contains no screen coordinates, input hooks or external actuation.

```powershell
python -m apc.evaluate_virtual_table `
  coach/examples/sample_solver_bundle.json `
  --output apc/runs/virtual-table-v1/provider_audit.json
```

The frozen audit covered 9/9 nodes and 21/21 actions, rejected every illegal
and duplicate probe, had zero external-actuation violations and measured 0.049
ms p95 step latency. This establishes the decision-provider contract only:
episodes use imported EV rewards rather than sampled terminal chips.

`full_hand_table.py` adds deterministic, complete heads-up no-limit Hold'em
hands with shuffled hidden cards, 0.5/1 BB blinds, legal no-limit action bounds,
all four betting rounds, all-in runouts, exact showdown ranking and zero-sum
settlement. Every observation and completed transition is fingerprinted; all
numeric poker quantities are serialized only in BB. It is an internal
environment with no screen, mouse, keyboard or external-table integration.

```powershell
python -m apc.evaluate_full_hand_table --hands 100 --seed-start 1000 `
  --output apc/runs/full-hand-table-v1/audit.json
```

The frozen audit completed 100/100 hands and 540 actions, covered fold, check,
call, bet, raise and all-in plus every street, and found zero deterministic
replay, unique-card, chip-conservation, zero-sum or external-actuation failures.
Internal step latency was 0.50 ms p95. This is a genuine sampled-outcome
trajectory provider, but it currently supports equal-stack heads-up, no-rake
play only. Multiway pots, side pots, antes, rake and trained policy evaluation
remain separate work.

`self_learning/full_hand_dataset.py` converts those complete hands into an
immutable outcome-learning dataset. It records only Hero-perspective decisions,
the legal action set, the exact behavior command, pre-state and transition
fingerprints, and the sampled terminal return in BB. Opponent hole cards never
enter a decision record. Entire hands are assigned to exactly one split.

```powershell
python -m apc.self_learning.full_hand_dataset build `
  apc/runs/full-hand-dataset-v1 --dataset-id full-hand-trajectory-v1 `
  --hands 100 --hand-seed-start 2000 --minimum-examples 100 `
  --minimum-groups 80
```

The frozen build contains 280 decisions from 100 hand groups with a 222/31/27
train/validation/test split. Its immutable file, examples and dataset-level
fingerprints validate, and all splits are hand-exclusive. The sampled return is
useful for outcome/value-learning experiments, but the deterministic coverage
behavior is neither a solver target nor a GTO or promotion label.

`self_learning/train_value.py` provides the first deterministic terminal-return
value baseline. It uses suit-renaming-invariant hashed state features, trains
only on the training hand groups, reports MAE, RMSE, bias, sign accuracy and
five calibration bins independently for every split, fingerprints the
checkpoint and refuses activation or recommendations. Invalid live states and
any exposed opponent hole cards cause inference abstention.

```powershell
python -m apc.self_learning.train_value train `
  apc/runs/full-hand-dataset-v1 apc/runs/value-model-v1/checkpoint.json
```

The frozen baseline failed honestly: test MAE was 15.03 BB versus 14.81 BB for
the training-mean baseline, a -0.22 BB degradation. It remains uncalibrated and
unusable for policy selection. This identifies the next required architecture:
action-conditioned value estimation evaluated once on a fresh hand corpus;
the failed test split must not be reused for model selection.

`self_learning/train_action_value.py` implements that isolated follow-up. It
crosses structured poker features with an exactly validated visible legal
command, adds rank/suit/board-shape features, selects an epoch using validation
MAE only, and compares untouched test error with both training-action-mean and
global-mean baselines. A fresh 300-hand corpus (`3000–3299`) produced 840
examples split 597/137/106 by complete hand.

This second experiment also failed safely. Validation-selected epoch 3 reached
8.21 BB validation MAE, but untouched test MAE was 9.710 BB versus 9.655 BB for
the action-mean baseline and 9.517 BB for the global baseline. No further
feature or hyperparameter tuning is permitted on that test split. The evidence
indicates that one sampled showdown return per behavior action is too noisy;
the next dataset must average repeated paired rollouts or use separately
verified counterfactual solver values.

`self_learning/paired_rollout_dataset.py` now supplies the first variance-
controlled alternative. For each Hero-button preflop state it evaluates fold,
call, minimum raise and all-in using exactly the same hole cards and board
runout, then continues with a declared check/call opponent policy. All four
counterfactuals share one hand group and split; opponent cards remain absent
from the learning state.

The frozen build contains 2,000 paired deals, 8,000 action examples and all 169
starting-hand classes. Common-random-card pairing reduced raise-versus-call
standard error from 0.0493 BB to 0.0220 BB (55.3%). It reduced all-in-versus-
call error by only 1.0%, exposing the remaining high-variance target rather
than hiding it. This corpus is suitable for a fresh small-action counterfactual
experiment, while all-in requires stratified sampling or normalized targets.

`self_learning/train_paired_value.py` consumes a new, untouched 5,000-deal
paired corpus and learns validation-selected hand-class/action shrinkage values
for Hero-button preflop fold, call and minimum raise. It deliberately abstains
on all-in and any state outside that narrow scope. On the untouched test split
(732 deals per action), call MAE improved from 0.9508 BB to 0.9255 BB and raise
MAE from 1.9017 BB to 1.8510 BB, with all 169 hand classes covered. This is the
first APC sampled-outcome value candidate to pass its declared fresh offline
generalization gate.

The checkpoint remains non-promotable: its values are against one deterministic
check/call continuation policy, not GTO or a calibrated population model. It
cannot recommend, activate, estimate all-in, or generalize postflop. Those
restrictions are enforced by inference and checkpoint validation rather than
left as documentation only.

`self_learning/evaluate_paired_value_confidence.py` bootstraps complete paired
test deals rather than individual action rows. Across 5,000 deterministic
bootstrap resamples, the 95% improvement lower bound remained above zero for
call (+0.00525 BB), minimum raise (+0.01049 BB) and their aggregate (+0.00787
BB). The statistical improvement gate therefore passes with 100% hand-class
coverage.

The same audit prevents a stronger claim: expected absolute value-calibration
error was 0.156 BB for call and 0.312 BB for raise, with extreme-bin gaps of
0.525 BB and 1.050 BB. The checkpoint remains explicitly uncalibrated and
cannot recommend or activate. Validated table lookup itself is fast (0.0015 ms
p95), but this excludes perception and end-to-end coaching latency.

`self_learning/calibrate_paired_value.py` then froze explicit calibration limits
before opening another disjoint 5,000-deal corpus (`20000–24999`). A constrained
affine layer was fit on 1,506 validation examples and evaluated once on 1,582
test examples. It met the declared expected-calibration-error and maximum-bin-
gap limits, but worsened MAE by 0.0220 BB for call and 0.0440 BB for raise,
exceeding the allowed 0.01 BB regression. The overall calibration gate therefore
failed; the wrapper remains uncalibrated and cannot recommend or activate.

This negative result is retained rather than tuning against the opened test
split. Any next calibration method must be selected on validation data and use
another untouched corpus for final evidence.

`self_learning/postflop_paired_rollout_dataset.py` extends common-random-card
counterfactuals through flop, turn and river. Every complete hand contributes a
Hero in-position state on each street and evaluates check, minimum bet and
all-in against identical hidden cards/runout. The frozen build contains 2,000
hands, 6,000 paired states, 18,000 examples and 91 public texture classes; all
nine rows from a hand stay in one split and no opponent cards enter decisions.

Pairing again reduced minimum-bet comparison standard error by 55.3%, while
all-in improved only 1.0%. Flop, turn and river payoff summaries were identical:
with an opponent that always checks and calls, moving the same chip amount on a
different street cannot teach timing strategy. The corpus establishes postflop
state/action plumbing, but a useful next dataset needs continuation policies
that fold, bet and raise, plus out-of-position trunks.

`self_learning/postflop_policy_rollout_dataset.py` adds three explicit opponent
continuations: always check/call, always fold when pressured, and a board-aware
selective policy that folds air, calls one pair, and bets or raises two-pair or
better. The frozen 1,000-hand build contains 27,000 examples across 9,000
street/policy states and all 91 texture classes. The selective policy produced
2,028 folds, 3,315 calls, 1,790 bets and 657 raises.

Unlike the check/call-only corpus, timing now matters: selective-policy minimum-
bet value shifted from +0.239 BB on the flop to −0.011 BB on the turn and
−0.184 BB on the river. These are probe-policy outcomes—not GTO recommendations
or learned opponent profiles—but they provide the first postflop dataset where
street, texture and opponent response jointly affect the learning target.

`self_learning/train_postflop_policy_value.py` consumes that immutable corpus
without reopening its complete-hand splits. It selects a visible-feature
abstraction and shrinkage only on validation hands, then evaluates check versus
minimum bet once on 140 untouched test hands (1,260 policy states and 2,520
action values). The selected made-hand/board-texture abstraction reduced MAE
from the street/profile/action mean baseline's 1.2105 BB to 1.0370 BB and chose
the higher realized action 81.03% of the time. Its realized policy value gained
0.0778 BB per state; a 2,000-sample complete-hand paired bootstrap gave a 95%
interval of [0.0397, 0.1167] BB. Exact abstraction coverage was 98.81%.

The same audit reports descriptive value-calibration errors (0.0909 BB EACE,
0.2327 BB maximum bin gap) but deliberately leaves confidence uncalibrated.
Validated lookup p95 was 2.6422 ms across 500 calls against a 5 ms gate; this
excludes perception, profile inference and strategy integration. The checkpoint
supports only check/minimum-bet decisions against the three declared synthetic
opponent probes. It is offline evidence, not a GTO model, learned-player model,
recommendation provider or promotion candidate.

`coach/src/poker_coach/opponent_model.py` now persists exact Beta-posterior
player evidence in a content-addressed JSON store. Unique event IDs prevent a
replayed hand from incrementing a profile twice; atomic writes and optimistic
revision checks reject corrupt snapshots and stale writers. Sparse profiles
map to a uniform, observe-only prior. Evidenced fold, aggression and showdown
posteriors map to a probability mixture over the three declared probe policies,
with approximate 95% uncertainty propagated into every policy weight.

`self_learning/profile_conditioned_postflop.py` applies that mixture to the
postflop value candidate. An offline action is exposed only when the profile
evidence gate passes, all six policy/action abstractions are covered, and the
same action remains best throughout the mixture uncertainty envelope. The
frozen audit used 450 unique profile events, three distinct archetypes and all
420 untouched test states (1,260 profile/state evaluations). Stable-action
coverage was 98.81% for the overfolder, 90.24% for the sticky-passive profile
and 70.00% for the aggressive-selective profile; uncertain cases abstained.
All outputs remained non-authorizing and bridge p95 latency was 3.7562 ms.

This closes profile persistence and proves an uncertainty-aware synthetic-
policy bridge. It does not establish a learned opponent classifier, real-player
calibration, GTO correctness, or permission to emit evaluated coaching advice.

`self_learning/postflop_position_rollout_dataset.py` removes the prior BTN-only
training restriction. Each of 1,000 seeds now produces card-matched Hero BTN
and Hero BB trunks on flop, turn and river. Every position/street state is
paired across check, minimum bet and all-in against all three opponent probes;
all 54 rows from the seed remain in one split. The frozen corpus contains
54,000 examples, 6,000 position states, 18,000 policy states and 87 texture
classes, with no opponent private cards in decision inputs.

Position behaves as a real feature only where the continuation can react to
action order. Check/call and fold-to-pressure probes have exactly zero BTN/BB
payoff difference. Against the selective probe, BB-minus-BTN minimum-bet
advantage is +0.046 BB on the flop, +0.093 BB on the turn and +0.176 BB on the
river because an in-position opponent can stab after Hero checks. This closes
the basic out-of-position data-path gap but does not yet cover raised pots,
facing-bet Hero nodes, multiple bet sizes or learned opponent populations.

`self_learning/evaluate_paired_policy.py` adds deterministic candidate inference
and paired node-bootstrap confidence intervals on this provider. Inference
renormalizes only a fully supported legal-action set and abstains if any action
is outside the checkpoint vocabulary. The first audit compares the smoke
candidate with a clearly labelled uniform-action reference, not an incumbent.
It covers only 1/9 nodes: the other eight require bet sizes absent from training.
The covered node improves reference EV by 0.0783 BB, but one node, 11.11%
coverage and a non-incumbent comparator categorically fail promotion. The
declared gate requires at least 30 independent nodes, at least 90% action-set
coverage, a paired 95% lower bound above zero and a real incumbent comparator.

`self_learning/checkpoint_registry.py` keeps activation separate from immutable
candidate checkpoint contents. The content-addressed local registry requires an
expected revision for every mutation. Registration never changes the active
fingerprint. Promotion requires fingerprinted evidence naming both candidate
and incumbent, a passing paired incumbent comparison, passing safety
non-regression, the aggregate promotion gate and explicit registry activation
authorization. Rollback restores the exact promotive predecessor and verifies
its stored bytes and semantic checkpoint fingerprint before reporting success.

```powershell
python -m apc.self_learning.checkpoint_registry status <registry-folder>
```

Tests cover registration without activation, strict fixture promotion, failed
and tampered evidence, stale revisions, registry/artifact tampering and exact
rollback. This proves lifecycle mechanics only; no current APC candidate has
real evidence that satisfies the promotion contract.

Candidate inference independently validates game/player metadata, BB stack,
pot and call values, board and hole-card integrity, canonical action history,
legal-action agreement, rake and utility declarations. Invalid states abstain
without probabilities. A structurally valid supported state returns
`prediction_ready_uncalibrated`: it may be used in offline evaluation, but
`recommendation_allowed`, calibrated confidence and activation all remain
false. `self_learning/evaluate_candidate_safety.py` tests ten deterministic
invalid-state mutations. The smoke candidate accepted none and produced zero
recommendation/activation violations across valid and adversarial paths. This
is standalone safety evidence; paired incumbent non-regression remains open.

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
