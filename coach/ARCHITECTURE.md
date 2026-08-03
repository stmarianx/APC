# Local poker coach architecture

## Product contract

Every recommendation must be reproducible from a complete information state and return:

- legal actions;
- baseline action frequencies;
- action EVs and EV loss;
- range and board assumptions;
- opponent-model evidence and uncertainty;
- source/provenance;
- the concepts responsible for the explanation.

No strategy is defined until variant, positions, stacks, blinds/antes, rake or payouts, cards, action history and available sizes are fixed.

## Data flow

```text
saved hand-history folder or pasted history
  -> completed-hand boundary detector
  -> idempotent SQLite library
  -> source parser
  -> canonical HandHistory events
  -> state replayer and validator
  -> range/equity + solution adapter + player posterior
  -> action EV/frequency comparison
  -> explanation and study queue
  -> independent training table
```

## Core modules

1. `models.py`: immutable source-of-truth cards, players, actions and hands.
2. `pokerstars.py`: text-to-event adapter; raw lines remain attached to actions.
3. `decision_math.py`: exact context-free equations with explicit assumptions.
4. `profiles.py`: opportunity-based Bayesian tendency estimates with contextual splits.
5. `features.py`: position mapping and opportunity extraction from completed hands.
6. `replay.py`: stacks, contributions, pots, call prices, effective stacks and decision snapshots.
7. `equity.py`: exact hand evaluation and exhaustive known-hand/weighted-range equity.
8. `ranges.py`: standard starting-hand notation, weights, blockers and combination generation.
9. `isomorphism.py`: exhaustive 24-permutation suit canonicalization that retains board arrival order and blocker/flush relationships.
10. `solutions.py`: deterministic suit-isomorphic spot keys plus validated frequency/EV/provenance contracts.
11. `explanations.py`: replayed state, decision math and optional solution comparison.
12. `storage.py`: idempotent SQLite hand imports, legacy-fingerprint migration, incremental file-state tracking and persistent solution cache.
13. `trainer.py`: provenance-aware scenarios, reproducible sessions, EV-loss grading and live feedback.
14. `river_solver.py`: full-tree CFR for a configurable one-bet river abstraction, including best-response exploitability audit.
15. `solver_import.py`: strict versioned per-hand solution bundles, configuration fingerprints and idempotent store imports.
16. `solver_adapters.py`: auto-detected JSON plus grouped one-action-per-row CSV interchange adapters.
17. `solution_tree.py`: unambiguous ancestor linkage, branch preservation, root-to-node traversal and ambiguity reporting.
18. `range_strategy.py`: public-node grouping, starting-hand classification, exact-combo coverage and per-class action aggregation.
19. `exploit.py`: uncertainty-gated posterior tendency rules and directional exploit overlays.
20. `matching.py`: normalized suit-isomorphic hand-to-solution matching, observed sizing resolution, EV loss and prioritized drill generation.
21. `study.py`: deterministic spaced-review scheduling, mastery, streaks and due-state contracts.
22. `ingest.py`: recursive saved-folder discovery, completed-summary gating, encoding fallback, file stability checks and scan diagnostics.
23. `board_texture.py`: exact board pairing, suit and straight-window structure plus hero made-hand, draw and blocker interaction without unsupported range-advantage claims.
24. `range_matchup.py`: blocker-adjusted weighted range equity, exact or deterministic sampled runouts, confidence intervals and explicitly range-relative current nut shares.
25. `range_inference.py`: Bayesian combo-weight updates from imported per-combo action frequencies, including action evidence, entropy, KL divergence and total-variation shift.
26. `range_timeline.py`: public-state matching for non-Hero saved actions, automatic Bayesian range updates, conservative posterior carry and explicit unmatched-action gaps.
27. `range_calibration.py`: revealed-combo mapping through suit isomorphism, probability trajectories, log loss, multiclass Brier score, rank accuracy and reliability buckets.
28. `state_transition.py`: auditable temporal gate for revision, table/hand identity, normalized actions, immutable private/configuration fields, board/history prefixes and plausible chip-flow progression.
29. `web.py`: persistent local API plus training table, River Solver Lab, solution bridge, continuous saved-hand intake and hand/drill review UI.
30. Future vendor-specific converters and population-model calibration.

## Trust boundaries

- A hand visible during an active file append remains pending until its summary marker arrives; unstable files are retried.

- Imported text is untrusted and must parse or fail explicitly—never silently invent cards, stacks or actions.
- Automatic course transcripts support explanations but are not hand-state truth.
- Opponent statistics require opportunity counts and posterior uncertainty.
- Solver recommendations must retain configuration hashes: game, rake, stacks, ranges and action abstraction.
- Private cards are part of per-hand solution fingerprints; two hands at the same public node cannot overwrite one another.
- Literal suit names are not strategic identity. Fingerprints canonicalize all global suit renamings while retaining exact suit-sharing relationships between board and private cards.
- Drill generation reports exact/close/approximate match confidence and never hides unmatched decisions.
- Board-only structure never masquerades as range or nut advantage; those labels require explicit ranges derived from position and action history.
- Range matchup claims identify the supplied notations, blocker-adjusted weights, enumeration/sampling method, sample count, confidence interval and range-relative nut definition.
- Bayesian range conditioning is restricted to imported exact combos at one public node; an observed action with zero prior support fails instead of manufacturing a posterior.
- Opponent timelines match public information only. A posterior carries across streets only when exact-combo coverage and solver provenance are identical; missing intermediate nodes are counted and do not contribute invented likelihood evidence.
- Imported combo posteriors are the current sparse Public Belief State representation. Revealed-card calibration measures this belief without converting unsupported hands into synthetic solver coverage.
- A visually stable read is still untrusted until it passes temporal invariants. Rejected revisions retain structured evidence and violations while the last accepted solver state remains unchanged.
- Future real-time subgame refinement must preserve parent-blueprint counterfactual-value constraints or label the result as an unsafe local approximation, and it must expose a latency-triggered cached-blueprint fallback.
- Negreanu-derived exploit heuristics may adjust a separately labeled opponent model or training curriculum; they must not alter the reference game's chip utility and still be called GTO.
- The live-feedback loop consumes only application-owned training states. Saved third-party hands enter through the post-hand text importer.
