# APC model, data and readiness specification

Version: 0.2.0
Status: foundation contract  
Units: BB

## 1. Product objective

APC is both the product name and the name of its trainable neural-network
model. It is an AI poker coach for controlled or explicitly permitted
virtual-chip environments. It must turn visible table-frame sequences into an
auditable poker state, combine learned policy/value estimates with a
solver-backed baseline and evidence-gated player adjustments, and produce a
low-latency recommendation expressed entirely in big blinds.

Deterministic lookup tables, linear smoke models and abstraction-based value
checkpoints in this repository are APC baselines, teachers and regression
controls. They are not the final APC neural model and must never be presented
as if they were.

APC is platform-agnostic. Each visual theme is handled by a calibration or
provider profile that implements one canonical observation contract. No
strategy component may depend on screen coordinates, colors or a particular
client name.

APC is also machine-portable. Development begins with bounded CPU-first runs on
the available 8 GB laptop, while datasets, checkpoints and run reports use
relative paths and cryptographic fingerprints so training can move to a
stronger machine without changing the model contract. Public source visibility
during development does not require downloaded dependency caches or large
training checkpoints to be committed. A clean-clone and artifact-integrity gate
is required before moving training to another machine.

## 2. Required outputs

For every accepted decision state APC returns:

1. normalized hand and table identity;
2. hero cards, board, positions, stacks, pot, price and legal actions;
3. a field-level confidence and source evidence trail;
4. baseline action frequencies and EVs when a solver node covers the state;
5. an explicit uncovered-state or fallback status otherwise;
6. opponent evidence, sample size and uncertainty;
7. any bounded exploit overlay separately from the baseline;
8. a recommended action, size in BB, explanation and latency audit;
9. abstention when critical perception or strategy evidence is insufficient.

## 3. APC components

### 3.1 Course knowledge curriculum

The 38 Negreanu videos are not treated as a source of exact equilibrium
frequencies. Their 9,011 timestamped transcript segments, contact sheets and
63-node/79-edge concept graph become a provenance-aware instructional corpus.
Training examples must retain video id, timestamp, concept ids, extracted
claim, assumptions and whether the target is a heuristic, exploit, math fact or
solver-verifiable statement.

The curriculum supports explanation, retrieval, scenario generation and
concept tagging. Exact GTO targets come only from declared solver outputs or a
reproducible self-play solver configuration.

### 3.2 Visual perception provider

The first trainable provider uses a multi-stage, multi-head design:

- layout detector: table boundary, seats, board, pot and control region;
- card detector/classifier: rank, suit, card back, empty slot and occlusion;
- text detector/OCR: player name, stack, pot, bet and button label text;
- seat head: occupied, hero, active, folded, sitting out and all-in state;
- dealer/button head;
- action-control head: available action, enabled state and displayed amount;
- event head: observed actor/action/amount between stabilized frames;
- calibration head: field confidence and out-of-distribution score.

Pixel inference must emit the existing normalized visual-observation contract.
It must never emit a strategic recommendation directly.

### 3.3 Temporal state tracker

The tracker stabilizes distinct frames, preserves hand/board/action prefixes,
checks chip and pot progression, prevents card identity changes, and rejects
revision rollback. A rejected transition must not mutate the last accepted
state. Animation frames, occlusions and ambiguous OCR remain pending rather
than becoming invented actions.

### 3.4 APC neural model

The versioned neural contract is `neural/config_v1.json`, validated by
`neural/contract.py`. APC uses a modular multimodal temporal architecture so
components can be pretrained independently and jointly fine-tuned later:

- compact convolutional encoder for normalized visible table-frame sequences;
- masked transformer for canonical state and observed-action sequences;
- uncertainty-bearing player-profile encoder;
- gated fusion of visual, structured-state and opponent evidence;
- perception, temporal-consistency, legal-action policy, action-value BB,
  state-value BB, opponent-tendency and uncertainty/abstention heads.

Every policy output is masked by the visible legal actions. Bet and raise
choices carry both an action class and a BB/pot-fraction size representation.
Opponent private cards are forbidden from decision inputs. Course material is
used for concept/explanation auxiliaries; GTO labels require fingerprinted
solver output or a reproducible solver/self-play configuration.

APC may ingest stabilized live observations and update player posteriors during
a virtual-chip session. Neural policy weights remain frozen during a hand.
Only completed hands enter the content-addressed replay buffer, and weight
updates create a new offline candidate that must pass calibration, safety,
paired-incumbent and rollback gates before activation. This is how APC evolves
from live action without turning one noisy or adversarial hand into an
immediate uncontrolled weight change.

### 3.5 Strategy and coaching layer

The strategy layer consumes only accepted canonical states. Priority order:

1. exact or suit-isomorphic solver node with matching configuration;
2. versioned blueprint plus certificate-checked refinement;
3. labeled educational baseline for the controlled trainer;
4. abstention from a GTO claim when none of the above covers the state.

Opponent adjustments retain the solver strategy as the baseline. Small samples
are observe-only. Directional overlays require posterior evidence and display
their opportunity count and uncertainty interval.

### 3.6 Self-learning layer

Self-learning has two distinct loops:

- **online experience:** stabilized visible actions, player posteriors and range
  beliefs update during a virtual-chip session without modifying neural policy
  weights;
- **offline neural improvement:** completed trajectories enter a versioned
  replay buffer for solver imitation, self-play, policy/value learning,
  continual-learning regression mixing and calibration.

No candidate checkpoint replaces the active checkpoint automatically. It must
pass the fixed regression suite, held-out perception set, strategy evaluation,
calibration gates and paired virtual-chip evaluation against the incumbent.

### 3.7 Deadline-aware decision and controlled action layer

APC must read or receive the actual per-turn deadline; it must not assume a
fixed 30-second clock. Every accepted decision carries a monotonic observation
time, state revision/fingerprint, legal actions, deadline, safety margin and
actuation reserve. The scheduler selects the strongest available strategy tier
that fits the remaining compute budget: bounded refinement, cached exact node,
cached blueprint or fast policy. If none fits, the controlled trainer uses an
explicit check-then-fold fallback when legal. It never invents a call or size.

Any action interface is downstream of strategy and is enabled only for a
controlled virtual-chip training table. It supports fold, check, call, bet,
raise and all-in through a provider-independent command contract. Before an
action can be sent, the gate rechecks the state revision and fingerprint,
observation age, legal actions, unambiguous BB sizing, deadline safety margin
and duplicate authorization token. Perception/strategy code contains no screen
coordinates and cannot directly click a client control.

## 4. Canonical visual annotation

`schemas/frame_annotation.schema.json` is authoritative. Each item binds an
immutable image digest to:

- capture session, sequence index and timestamp;
- provider-independent environment metadata;
- normalized bounding boxes;
- seat occupancy, names, stacks and statuses;
- hero and community cards;
- pot and to-call amounts;
- dealer position;
- legal action buttons;
- observed action events;
- the canonical poker state used as the training target;
- annotation provenance and verification status.

All chip quantities are normalized to BB. Raw display text may be retained only
as evidence for OCR evaluation.

## 5. Dataset rules

1. Only controlled or explicitly permitted virtual-chip sources are eligible.
2. Frames from the same capture session may appear in only one split.
3. Near-duplicate frames and adjacent animation frames remain in the same split.
4. Test sessions must include unseen combinations of layout, scale or theme.
5. Card, seat, stack, action and street distributions are audited before use.
6. Corrections create a new annotation revision; image digests never change.
7. Raw frames and checkpoints stay outside version control; manifests, labels,
   metrics, configuration hashes and model cards are versioned.
8. Player identifiers used for model training are pseudonymized unless a name
   crop is needed for OCR evaluation; persistent opponent mapping stays local.

## 6. Initial dataset target

The first supervised run requires at least:

- 2,000 verified labeled frames;
- 8 independent capture sessions;
- 3 table layouts or seat counts;
- 2 visual themes;
- every street and every legal action class;
- at least 500 frames in temporally adjacent sequences;
- at least 500 verified frames from two controlled or explicitly permitted
  visible-table sessions (synthetic frames may augment but cannot satisfy this);
- group-exclusive train/validation/test splits;
- at least 10% double-audited annotations;
- zero known cross-split image or perceptual-hash duplicates.

These are minimums for beginning model development, not proof of generality.

## 7. Training stages

### Stage A — course grounding

Build timestamped concept examples and evaluations from the verified course
corpus. This trains explanation and concept retrieval, not the equilibrium
policy.

### Stage B — perception pretraining

Train layout, card, OCR, seat and action heads with grouped splits. Preserve
per-field confidence and evaluate exact-match state reconstruction.

### Stage C — temporal learning

Train/evaluate event recognition across frame sequences and combine it with
hard state-transition constraints.

### Stage D — APC policy/value pretraining and strategy integration

Train APC's legal-action policy and BB value heads from fingerprinted solver
targets and the existing tabular teacher baselines. Feed only stable canonical
states into the range, equity, solver, profile and explanation engine. Measure
abstention, calibration and exact solver coverage.

### Stage E — virtual-chip self-play

Generate controlled trajectories from frozen game configurations and ingest
eligible completed live-observation hands. Train a new APC neural checkpoint
from the versioned replay buffer, retain incumbent regression samples, evaluate
best responses or exploitability where tractable, and promote only
evidence-backed improvements.

## 8. Readiness gates

### Gate T — ready to train on visible tables

All of the following must be evidenced:

- annotation and dataset schemas validate;
- capture/annotation/export tools operate end to end;
- minimum dataset target is met;
- the controlled visible-table minimum is met independently of synthetic data;
- split leakage audit passes;
- reproducible training configuration and deterministic seed are recorded;
- baseline training smoke test produces a checkpoint and metrics;
- a held-out evaluation command runs without reading training labels.

Until Gate T passes, APC is **not ready for visible-table training**.

### Gate P — perception model ready for integration

- card rank+suit exact accuracy >= 99.5%;
- stack, bet, pot and to-call exact normalized-value accuracy >= 99.0%;
- seat, dealer, action and control macro F1 >= 98.0%;
- complete critical-state exact accuracy >= 97.0%;
- expected calibration error <= 2.0%;
- critical-field false-confidence rate <= 0.1%;
- out-of-distribution and low-confidence frames abstain;
- perception p95 latency <= 250 ms on declared target hardware.

### Gate C — evaluated coaching prototype

- accepted states pass all temporal invariants;
- BB normalization and legal-action filtering are exact on the test corpus;
- strategy provenance is present for every non-abstaining recommendation;
- solver coverage is reported rather than inferred;
- end-to-end p95 latency <= 500 ms on declared target hardware;
- recommendation regression and virtual-chip scenario evaluations pass;
- player adjustments remain bounded and uncertainty-gated.

### Gate S — self-learning promotion

- replay dataset and training configuration are immutable and fingerprinted;
- candidate passes perception, calibration and strategy regressions;
- candidate does not increase invalid-state acceptance or unsafe confidence;
- paired virtual-chip evaluation improves the declared objective with a
  confidence interval excluding zero;
- rollback to the incumbent checkpoint is tested.

For an APC neural checkpoint, Gate S additionally requires a pinned neural
runtime, architecture/configuration fingerprint, deterministic initialization
seed, replay-buffer fingerprint, legal-mask invariants, catastrophic-forgetting
regression suite and a paired evaluation against the declared active APC
checkpoint. Live experience collection alone can never satisfy promotion.

## 9. Evaluation reports

Every training run produces:

- dataset manifest and split fingerprints;
- source revision and environment information;
- model architecture/configuration hash;
- random seeds and target hardware;
- per-head and complete-state metrics;
- calibration and abstention curves;
- latency distribution;
- error slices by layout, theme, resolution, street and action;
- known limitations and checkpoint promotion decision.

## 10. Current status

The course corpus, mathematical engine, annotation/capture pipeline, synthetic
perception baselines, temporal tracker, virtual table, Bayesian profiles,
solver interchange and offline value teachers exist. The APC neural
architecture and completed-live-hand replay contracts are versioned. PyTorch
2.13 CPU is installed in the development runtime, and four full-corpus APC
neural candidates have demonstrated deterministic state/profile training,
pickle-free fingerprinted weights and sub-8 ms p95 strategy inference. V4 was
frozen before a one-shot, disjoint 600-hand sealed audit; its 4.3829 BB MAE,
29.29% action accuracy and 2.3903 BB regret remain below the teacher's 3.4254
BB, 68.50% and 1.7825 BB on the same sealed states.
Completed virtual-chip hands can now enter an immutable content-addressed
replay buffer with deterministic hand-group splits, priority sampling and
incumbent-regression retention.
They remain unpromoted: the visual branch is untrained, the raised-postflop
teacher remains stronger, confidence is uncalibrated, and targets are not
verified GTO labels. The authoritative evidence and missing gates are recorded in
`readiness.json`.
