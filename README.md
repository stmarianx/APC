# APC — AI Poker Coach

> Development repository: rebuildable runtime caches and large checkpoints are
> kept separate from source. See `apc/PORTABILITY_AND_RELEASE.md` for the
> laptop-to-training-machine workflow.

This workspace builds APC, a platform-agnostic visual poker coaching model for
controlled or explicitly permitted virtual-chip environments. It contains three
verified layers:

1. `analysis/` - complete 38-video course analysis, primary-source research, transcript evidence, mathematical GTO foundation and the interactive concept graph.
2. `coach/` - a working local training-table and saved-hand-review application backed by exact poker math, player tendencies and provenance-aware strategy contracts.
3. `apc/` - the model, visual-dataset, training and readiness contract for turning visible table frames into confidence-gated poker states and BB recommendations.

## Verified status

- Research: 38/38 videos, 82,599 transcript words, 9,011 timestamped segments, 38 visual contact sheets, 63 graph nodes and 79 edges.
- Application core: 165 passing tests across 38 modules, including CFR convergence, solver adapters, multi-street traversal, editable and Bayesian-conditioned ranges, saved-hand opponent range timelines, revealed-card calibration, exact/deterministic range-matchup analysis, hidden-strategy solver practice, revisioned normalized table states, temporal transition invariants, growing hand-history capture, calibrated visual observations, exact board/blocker reasoning, uncertainty-gated exploit insights, durable study scheduling, recursive folder ingestion, compact opponent-profile export, BB session trends and HTTP integration.
- APC foundation: versioned model specification, visible-frame annotation schema including turn-clock targets, grouped dataset manifest, browser-verified graphical annotation workbench, explicit-region read-only screen acquisition, automatic abstention-capable table localization, immutable ordered-folder capture/import pipeline, fingerprinted review-only model suggestions, same-session verified layout propagation, portable artifact manifests, deadline-aware controlled-action authorization, deterministic cross-dataset merging, fail-closed executable training gate, deterministic resumable synthetic renderers, base/card/segmented-numeric/stack/event/boundary/turn-clock/name pixel trainers, conservation-checked complete-hand tracker, persistent uncertainty-aware identity registry, pseudonymous visual identity signatures, explicit multiway stack context, fingerprinted viewport calibration, strict perception-to-backend bridge, frozen synthetic/OOD audits and 129 passing APC tests. The dedicated synthetic countdown head reads 16/16 untouched test clocks exactly at 62.16 ms p95 and feeds canonical remaining milliseconds into the composite, but remains non-promotable. The character-level name head reconstructed 64/64 group-exclusive held-out unseen names exactly, resolved every repeated-frame test identity and reduced p95 from 608.54 ms to 49.79 ms without changing predictions; arbitrary controlled-visible names remain uncalibrated. A separately generated integrated audit reached 100% joint accuracy across 13 supported fields, 188.38/171.34 ms validation/test p95, and preserved bounded-refinement budget on all 32 frames; this is still synthetic-only and excludes solver/actuation latency. A fresh 2,000-frame/250-hand synthetic corpus passes full image, annotation, split, identity and temporal-continuity audits; Gate T now fails only because controlled-visible data remains 0/500 frames and 0/2 sessions. The fresh semantic checkpoint passed 96/96 untouched test transitions and 24/24 complete hands with exact histories, visible states, effective-stack maps and boundaries, replacing—but not erasing—the historical 1/2-hand failure. A matched cold benchmark on this laptop preserved the exact prediction fingerprint while reducing synthetic complete-hand p95 from 1.103 s to 0.444 s; controlled-visible latency and the 250 ms perception target remain open. The table locator detected 192/192 frozen synthetic-test frames at mean IoU 0.841 and reached IoU 0.818 on the one-pass real OOD frame, but remains explicitly uncalibrated. The bridge reaches an exact sample solver node but remains calibration-gated. The frozen user-provided screenshot’s fresh semantic checkpoint still matched only 1/6 verified fields; synthetic correctness and localization success do not establish real card/state recognition, and both original and normalized frames remain excluded from training. The generated knowledge curriculum contains 214 timestamped course-evidence examples plus 10 verified mathematical primitives with group-exclusive lesson splits and no unsupported exact-strategy labels. APC is not yet ready for arbitrary visible-table training; Gate T remains open in `apc/readiness.json`.
- Training app: six live-feedback decision spots from preflop through river, explicitly labeled as educational baselines pending solver import.
- River Solver Lab: solver-generated approximate Nash frequencies for a configurable one-bet river abstraction.
- Full-tree bridge: versioned per-hand solution bundles with configuration hashes, EVs and idempotent persistence.
- Canonical solver cache: all 24 equivalent suit renamings reuse one strategy node without collapsing different flush or blocker structures.
- Multi-street interchange: a strict one-action-per-row CSV adapter groups preflop-through-river solver exports into validated strategy nodes.
- Solution traversal: durable node IDs form auditable root-to-river paths with preserved branches and explicit ambiguous-parent reporting.
- Live table assistant: normalized in-progress states are revision-checked, matched suit-isomorphically to imported nodes, filtered by current legal actions and returned with pot odds, SPR, confidence, provenance and auditable decision EV.
- Live text capture: a stateful adapter tails a growing PokerStars history file or the newest file in a folder, detects hero turns from betting-round order, gates partial writes and advances the Live Assistant automatically without importing incomplete hands into the durable library.
- Visual-provider bridge: calibrated screen regions, screenshot hashes and per-field confidence are audited; low-confidence reads are blocked, two distinct consistent frames are required, and temporal chip/board/card/action invariants reject impossible transitions without mutating the last accepted solver state.
- Structural reasoning: every normalized state now reports pairedness, suit texture, straight-window coverage, hero made hand, private draws and nut-flush blockers while withholding range/nut-advantage claims until explicit ranges exist.
- Range Matchup Lab: two weighted ranges produce blocker-adjusted equity, exact or deterministic sampled runouts, a 95% interval, made-hand distributions and explicitly range-relative current nut shares.
- Bayesian range inference: an observed action reweights imported exact combos by their solver frequency and reports action evidence, entropy reduction, effective-combo count, total-variation shift and combo-level Bayes factors.
- Opponent range timelines: saved non-Hero actions are matched to compatible imported public nodes, conditioned automatically, and carried across streets only with identical combo coverage and provenance; skipped nodes remain visible as inference gaps.
- Showdown calibration: revealed opponent cards score range beliefs using support coverage, log loss, multiclass Brier score, top-rank accuracy and forecast-versus-outcome reliability buckets.
- Range explorer: imported exact combos populate an honest 13×13 coverage matrix with editable 100%/50%/excluded weights and recalculated action mixes.
- Player adjustments: Bayesian tendency evidence produces observe-only early signals or directional exploit overlays while retaining the solver as baseline.
- Personalized review: matched hero decisions become confidence-weighted EV-loss drills.
- Session diagnosis: reviews report total/average EV loss, the worst decision, street-level concentration and exact solver coverage.
- Study progression: drill ratings persist mastery, streak, interval, due date and immutable attempt history.
- Continuous saved-hand intake: completed PokerStars text hands are imported idempotently from a selected folder; partial writes wait, unchanged files skip, and cumulative player-style profiles refresh automatically.
- Durable opponent map: the CLI can export a compact BB-only JSON snapshot with style labels, Bayesian estimates, confidence intervals and uncertainty-gated exploit insights while preserving the separate hand-authored evidence notes.
- Sample play-money hand: 12 parsed actions, 9 pre-action decision snapshots and exact pot reconciliation (`290 = 280 award + 10 rake`).
- Browser verification: training feedback, CFR solving and saved-hand analysis all complete without console errors.

See `analysis/research_verification.json` and `coach/verification.json` for machine-readable audits.

## Launch the application

```powershell
Set-Location 'C:\Users\st_ma\Documents\Negreanu\coach'
$env:PYTHONPATH = (Resolve-Path '.\src')
python -m poker_coach.web --open --database '.\poker_coach_lab.sqlite3'
```

## Current workflows

```text
saved text hand histories
  -> incremental folder scan and completed-hand gate
  -> durable deduplicated hand library
  -> parser and validation
  -> replayed decision states
  -> exact range/equity and decision math
  -> Bayesian player tendencies
  -> optional cached solution frequencies/EVs
  -> explainable report

independent training scenario
  -> hidden reference strategy
  -> user decision
  -> immediate frequency, EV-loss and math feedback
  -> session score and review queue

river abstraction
  -> full-tree counterfactual regret minimization
  -> average strategy and best responses
  -> exploitability-audited bluff and defense frequencies
```

PokerStars desktop clients can save text hand histories under Settings -> Playing History -> Hand History. The current regional adapter was validated against a 242-completed-hand PokerStars.RO snapshot spanning tournament, six-player cash and nine-player cash files with no parse errors or incomplete blocks. Future client-format changes still require regression validation.

## Next implementation milestones

1. Collect and double-audit the minimum controlled virtual-chip visible-table dataset required by Gate T.
2. Collect controlled-visible examples for variable-font name, numeric and turn-clock calibration.
3. Train card, numeric, dealer, seat, name and event heads and evaluate them against Gate P.
4. Calibrate the integrated perception-to-solver bridge and evaluate BB recommendations end to end.
