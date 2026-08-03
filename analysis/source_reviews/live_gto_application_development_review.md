# Review: Live GTO Poker Application Development

Source reviewed: `C:\Users\st_ma\Downloads\Live GTO Poker Application Development.docx`

The supplied report was structurally inspected as a 141-paragraph, three-table Word document. LibreOffice was unavailable, so page-render verification could not be performed. Architectural decisions below rely on extracted headings, paragraphs, tables, relationships and media structure rather than inferred page layout.

## Useful ideas adopted

1. **Public Belief State as the strategic state.** The report correctly emphasizes that public cards, action history and beliefs over private states—not a single guessed hand—must drive imperfect-information decisions. The coach's imported exact-combo range posterior is therefore treated as a sparse, provenance-bound Public Belief State. `range_timeline.py` updates it from observed actions; `range_calibration.py` scores it only when cards are revealed.

2. **Temporal validation of perception.** Per-frame OCR confidence is insufficient. Stack, pot, street, board and action changes need cross-frame invariants so impossible transitions are rejected. The existing two-frame stability gate is a foundation; a future state-transition validator should add chip-conservation and board-prefix checks before a visual observation can become strategic input.

3. **Blueprint plus safe subgame refinement.** Imported full-tree strategies and low-latency cached nodes form the blueprint layer. Any future turn/river re-solving service must retain the parent blueprint's counterfactual-value constraints or explicitly label itself an unsafe local approximation. Latency budgets and a deterministic fallback to the cached blueprint should be part of that service contract.

4. **Action abstraction should include small sizes.** The solver interchange already permits arbitrary declared sizes. Solver-verified exports should include strategically relevant 20%, 25%, 33% and 40% pot branches where the game tree and rake model justify them, rather than assuming only half-pot, pot and all-in.

5. **Showdown/equity auxiliary evaluation.** The report's auxiliary-head idea reinforces the need to measure whether belief predictions correspond to revealed cards. The implemented calibration layer reports exact-combo support, probability trajectories, log loss, multiclass Brier score, rank accuracy and reliability buckets.

6. **Separation of perception, state, strategy and presentation.** The proposed asynchronous boundaries align with the current normalized visual-provider contract and local strategy API. Heavy solvers can later move behind the same versioned state/solution schemas without coupling screen calibration to solver internals.

## Ideas retained only with qualifications

- A Nash equilibrium is non-exploitable only within the solved game's assumptions. Heads-up, zero-sum guarantees do not automatically transfer to multiplayer, rake, omitted bet sizes or an approximate value network.
- ReBeL and Deep CFR are research directions, not drop-in claims. Model checkpoints require exploitability or best-response evaluation, reproducible training manifests and held-out calibration before they can be labeled solver verified.
- Opponent-policy shifts may support exploitative recommendations, but only with opportunity counts, uncertainty shrinkage, bounded deviations and a solver baseline. Raw empirical frequencies must not silently replace equilibrium beliefs.
- OCR techniques must be calibration-driven. Fixed RGB thresholds and generic Tesseract settings are examples, not robust universal contracts across themes, scaling, animation and localization.
- Mixed-strategy sampling needs reproducibility and auditability in training. Cryptographic randomness is not a mathematical requirement.

## Ideas rejected from the product architecture

- **Reward inflation for selected hand classes is not GTO.** Changing rewards for suited connectors changes the game being solved. Negreanu-style heuristics belong in a separately labeled exploit or curriculum layer, never inside the equilibrium reference utility.
- **Purchased or scraped bulk player histories are not an acceptable evidence source.** Player models should be derived from hands legitimately available in the user's own saved history and should retain sample-size uncertainty.
- **Input humanization and detection-evasion behavior are not coaching features.** The architecture remains focused on state analysis, recommendations, training and auditable review rather than autonomous click execution or concealment.

## Resulting implementation priorities

1. Finish showdown calibration and expose it in Hand-history Review.
2. Add cross-frame state-transition invariants to the calibrated visual-provider bridge.
3. Define a versioned blueprint/subgame-solver service contract with latency and safe-fallback metadata.
4. Expand solver-import fixtures to test micro-bet abstractions and counterfactual-value provenance.
