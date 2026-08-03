# Poker Coach Lab

This package contains a usable local poker-training application and the analysis engine beneath it. It provides immediate feedback on its own training table and reconstructs saved PokerStars-style text histories for post-hand review.

## Current capabilities

- Browser training table with six bundled preflop-to-river decision spots.
- Interactive River Solver Lab that computes approximate Nash strategies with CFR.
- Strict external solution-bundle import with per-hand fingerprints and SQLite persistence.
- JSON and tabular CSV solver adapters, including a four-street per-action export example and auto-detecting CLI.
- Persistent solver node IDs plus deterministic multi-street forest traversal, branching, path lookup and ambiguity reporting.
- Editable 13×13 range explorer with exact-node coverage and weighted aggregate action mixes.
- Exact/near-state matching from reviewed decisions to imported solution nodes.
- Suit-isomorphic solution matching and fingerprints: renamed suits share one node while flush/blocker structure remains distinct.
- Personalized drill queues ranked by confidence-weighted EV loss.
- Session-level leak summaries with measured EV loss, worst decision, street concentration and solver coverage.
- Persistent mastery, streak, due date and immutable attempt history for every drill.
- Reproducible sessions, legal-action validation, scoring and EV-loss feedback.
- Hidden strategy until the user acts, followed by frequencies, EVs, mathematical context and explanations.
- Explicit strategy provenance: the bundled spots are educational baselines, not solver-verified outputs.
- Local JSON API for health, scenarios, sessions, decisions and saved-hand analysis.
- Validated PokerStars-style text parsing and deterministic state replay.
- Pot, stack, call-price, effective-stack and pot-reconciliation audits.
- Exact 5-7 card evaluation and exhaustive equity against known hands or weighted ranges.
- Range notation, blocker filtering, pot odds, call/bluff EV, MDF, polar ratios and SPR.
- Exact board-texture reasoning: pairing, suit structure, straight-window coverage, made hands, private draws and nut-flush blockers, with unsupported range-advantage claims explicitly withheld.
- Range Matchup Lab with weighted notation, blocker-adjusted combination removal, exact small-tree enumeration, deterministic large-tree sampling, 95% intervals and current range-relative nut shares.
- Bayesian action conditioning in the Range Explorer: editable combo priors update through imported solver frequencies with action evidence, entropy reduction, effective-combo count and posterior shifts.
- Saved-hand opponent range timelines: compatible public solver nodes condition exact-combo ranges on observed actions, carry supported posteriors across streets, and expose missing-node gaps and provenance resets.
- Showdown calibration: revealed opponent cards score the preceding exact-combo posteriors with support coverage, log loss, multiclass Brier score, rank accuracy and reliability buckets.
- Bayesian player tendencies segmented by opportunity, position and stack bucket.
- Uncertainty-gated exploit overlays with posterior evidence, observe-only small samples and directional adjustments for established leaks.
- SQLite persistence for idempotent hand imports and fingerprinted solver solutions.
- Incremental folder scanning for PokerStars saved histories, including incomplete-write detection, unchanged-file skipping and cumulative player profiles.
- Recursive CLI folder import plus compact BB-only JSON opponent maps with Bayesian estimates, intervals and uncertainty-gated adjustments.

## Launch the application

From `C:\Users\st_ma\Documents\Negreanu\coach`:

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src')
python -m poker_coach.web --open --database '.\poker_coach_lab.sqlite3'
```

The server binds to `127.0.0.1:8765` by default and stores hands, solver nodes, drills and study history in `poker_coach_lab.sqlite3`. Open `http://127.0.0.1:8765/` if the browser does not open automatically. Use `Ctrl+C` to stop it.

The three application views are:

1. **Training table** - make a decision without seeing the reference strategy, then inspect frequency, EV loss, pot geometry, concepts and provenance.
2. **River Solver Lab** - change pot and bet sizes, run CFR, and inspect solver-generated bluff/defense frequencies plus exploitability.
   The same view can load, validate and import versioned external solution bundles.
3. **Hand-history review** - paste histories or point the app at the folder where PokerStars saves English text hands. Scan once or enable the five-second auto-scan while the app is open; completed hands enter the durable library and refresh player-style profiles, solution matches and drills. Rate drills `again`, `hard`, `good` or `easy` to update mastery and the next due date.

The folder scanner waits for `*** SUMMARY ***` before importing a hand, so a file being appended is safe to inspect. It remembers file size and modification time in SQLite, skips unchanged files, and uses hand IDs to prevent duplicates. The path and auto-scan preference remain local to this browser.

## Analyze from the command line

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src')
python -m poker_coach.cli 'C:\path\to\saved-hand-history.txt'
```

Persist multiple imports and analyze the accumulated database:

```powershell
python -m poker_coach.cli 'C:\path\to\saved-hand-history.txt' --database '.\coach.sqlite3'
```

Recursively scan a PokerStars history folder and generate a compact statistical opponent map:

```powershell
python -m poker_coach.cli 'C:\path\to\HandHistory' `
  --database '.\poker_coach_lab.sqlite3' `
  --recursive --profiles-only `
  --output '.\opponent_profiles.generated.json'
```

Folder scanning requires `--database` so completed hands are deduplicated and unchanged files are skipped. The generated map reports all quantities in BB and keeps statistical estimates separate from `opponent_profiles.json`, which contains hand-authored observations.

Import solver-produced per-hand nodes into the persistent solution cache:

```powershell
python -m poker_coach.solver_cli '.\examples\sample_solver_bundle.json' --database '.\coach.sqlite3'
```

Import a one-action-per-row multi-street CSV export through the same validator:

```powershell
python -m poker_coach.solver_cli '.\examples\sample_solver_export.csv' --database '.\coach.sqlite3'
```

Each bundle declares `schema_version`, solver source/version, and one or more nodes. Every node supplies the complete normalized game key, public board, private hero cards, action history, action abstraction, normalized frequencies and EVs. Invalid cards, non-finite values, frequency sums other than one, duplicate node ids and duplicate fingerprints fail explicitly. Fingerprints use `suit_isomorphism_v1`: all 24 global suit renamings collapse to one canonical key, but the imported cards remain unchanged for provenance and display.

The CSV column contract and converter target are specified in [SOLVER_EXPORT_FORMAT.md](C:/Users/st_ma/Documents/Negreanu/coach/SOLVER_EXPORT_FORMAT.md).

## Verification

```powershell
$env:PYTHONPATH = (Resolve-Path '.\src')
python verify_core.py
```

The verifier executes the full unit and HTTP integration suite and rewrites `verification.json`. The current suite covers domain validation, parsing, replay, math, equity, ranges, all 24 suit renamings, blocker/flush non-collisions, legacy fingerprint migration, solver convergence, exploitability, solution imports, state matching, temporal state invariants, sizing resolution, drill generation, spaced review, SQLite reopen persistence, incremental folder ingestion, incomplete appended hands, training sessions, web assets, API error handling and saved-hand analysis.

## Strategy provenance

The six bundled decision frequencies are course-derived educational baselines with mathematical sanity checks. They are deliberately labeled `educational_baseline` and `solver_verified: false`. River Solver Lab output is solver-generated for its explicitly displayed one-bet abstraction and includes an exploitability audit. The existing `SolutionKey` and `SolvedSpot` contracts remain ready for full no-limit solver exports, which must retain their configuration fingerprint and source version.

## Next milestones

1. Import solver-verified strategies and replace or supplement the educational baselines.
2. Expand the offline leak report with position/street filters and session-to-session trends.
3. Continue regional hand-history regression validation as the desktop client evolves.
4. Expand training-table calibration with quantified recognition error in permitted test environments.

See `ARCHITECTURE.md` for the component and trust-boundary design.
