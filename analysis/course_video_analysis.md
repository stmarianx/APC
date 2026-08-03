# Video-by-video course analysis

This index records the strategic role of every local lesson. Time-coded transcript evidence and representative visual frames are stored separately. The “coach primitive” column translates teaching content into a testable component for the future system.

| # | Lesson | Duration | Strategic takeaway | Coach primitive |
|---:|---|---:|---|---|
| 1 | Introduction | 2.87m | Course spans technical decisions, tournaments, live reads, and repeatable table thinking. | Curriculum/ontology root |
| 2 | Understanding Position | 12.50m | Acting later creates an information and equity-realization advantage; ranges widen as fewer players remain. | Position-aware state/range model |
| 3 | Hand Ranges and Board Texture | 27.88m | Reason range-vs-range, count value/bluff combinations, narrow ranges from action, and connect board texture to betting frequency. | Weighted range + texture feature engine |
| 4 | Ranges Hand Review | 12.24m | Apply range construction to observed hands rather than guessing one exact holding. | Hand replayer with posterior range trace |
| 5 | Game Theory and Math | 11.16m | Use an unexploitable baseline, then exploit observed errors; compare pot odds, frequencies, and price. | GTO/exploit split + exact EV math |
| 6 | C-Betting | 15.94m | Continuation betting depends on range interaction, position, board, sizing, and opponent continuation. | Flop strategy and EV comparison |
| 7 | Check-Raising | 7.14m | Check-raising combines strong value with credible bluffs/semi-bluffs and protects out-of-position checks. | Check/raise range constructor |
| 8 | Three-Betting | 9.16m | Re-raising ranges and sizes depend on positions, stack depth, opener tendencies, and value/bluff structure. | Preflop tree and sizing model |
| 9 | Three-Betting Hand Review | 4.40m | Translate 3-bet concepts into complete hand-line analysis. | Preflop review drill |
| 10 | Detecting and Executing the Bluff | 14.52m | Bluff when the line is credible, blockers and fold equity cooperate, and the target can fold. | Bluff-candidate and fold-equity scorer |
| 11 | Executing the Bluff Hand Reviews | 30.43m | Evaluate multi-street bluff plans against actual ranges and runouts. | Street-by-street bluff replay |
| 12 | Bet Sizing | 7.08m | Size according to purpose, range shape, price offered, and future pot geometry. | Pot/stack geometry calculator |
| 13 | Overbetting | 10.55m | Overbets are polarized tools supported by nut advantage and capped opposing ranges. | Polar sizing/indifference module |
| 14 | Multi-way Dynamics | 13.79m | Extra players reduce fold equity, strengthen continuing ranges, and tighten value/bluff thresholds. | N-player state and multiway warning layer |
| 15 | Mixed Strategy | 18.50m | Randomized frequencies reduce predictability and arise when actions have similar EV. | Frequency strategy + RNG for simulator only |
| 16 | Mixed Strategy Hand Review | 4.90m | Recognize practical spots where more than one action belongs in the strategy. | Mixed-action review and EV-loss display |
| 17 | Pre- and Postflop Mistakes | 10.14m | Diagnose leaks by separating preflop range errors from postflop line errors. | Leak taxonomy and attribution |
| 18 | Tournament Strategy: Early and Middle Stages | 12.41m | Adjust to stack depth, blind growth, table composition, and accumulation needs. | Tournament phase/stack model |
| 19 | Tournament Strategy: On the Bubble | 10.28m | Stack coverage and payout pressure create asymmetric risk and leverage. | ICM/risk-premium analysis |
| 20 | Tournament Strategy: Late Stages and Final Table | 10.29m | Pay jumps, short stacks, and relative stack sizes reshape ranges and all-in thresholds. | Final-table ICM and stack-role model |
| 21 | Universal Tournament Strategy | 9.55m | Combine structure, field, position, stack, and opponent information rather than using phase labels alone. | Tournament context schema |
| 22 | Cash Games | 11.68m | Cash utility is chip-based but depends on effective depth, rake, rebuy, and table quality. | Cash configuration and rake-aware EV |
| 23 | Masking Tells | 12.19m | Standardize physical and timing behavior so information leakage is minimized. | Training-only behavioral checklist |
| 24 | Spotting Tells, Part 1 | 11.34m | Establish a player baseline before interpreting deviations. | Observation reliability model |
| 25 | Spotting Tells, Part 2 | 9.31m | Combine multiple weak physical signals with betting evidence, never one tell in isolation. | Multisignal opponent evidence |
| 26 | Spotting Tells Hand Reviews | 14.06m | Test tells against shown-down hands and the action line. | Evidence calibration/review |
| 27 | Table Talk | 10.49m | Conversation can reveal comfort, intent, and tilt but also manipulates observers. | Optional live-session note taxonomy |
| 28 | How to Think at the Poker Table | 11.24m | Use a repeatable sequence: state, ranges, price, future streets, opponent, action. | Explainable decision checklist |
| 29 | Managing and Exploiting Tilt | 19.18m | Protect one's own policy from emotional drift and cautiously exploit observable deviations. | Session-quality and deviation flags |
| 30 | Table Image and Metagame | 15.88m | Opponents react to perceived history; current optimal action depends on those beliefs. | History-conditioned player model |
| 31 | Table Image and Metagame Hand Reviews | 13.49m | Connect table history to concrete range and frequency adjustments. | Meta-aware hand replay |
| 32 | Player Profiling | 14.27m | Classify tendencies only as a starting prior; update from opportunity-based evidence. | Bayesian player-style profile |
| 33 | Game Selection | 13.07m | Expected profit depends on table quality as well as technical edge. | Offline session/table evaluation |
| 34 | Bankroll Management | 11.31m | Match stakes to edge, variance, and risk tolerance; results need large samples. | Variance, confidence, and bankroll model |
| 35 | Off-Felt Training | 13.84m | Improvement comes from deliberate review, study, drills, and correction loops. | Study queue and spaced drills |
| 36 | Life as a Poker Player | 14.34m | Sustainable performance requires discipline, routine, and realistic variance expectations. | Responsible-use/session controls |
| 37 | Closing | 3.16m | Consolidate the course into repeatable practice rather than isolated tips. | Learning roadmap |
| 38 | Bonus Material: Online Play | 5.79m | Online play emphasizes speed, volume, digital hand histories, and reduced physical information. | Hand-history-first offline workflow |

## Synthesis

The course's strongest contribution is a human decision framework:

1. Locate the exact table context.
2. Assign ranges from position and action.
3. Recompute those ranges on the board.
4. Compare value, bluff, and sizing branches.
5. Update from opponent evidence.
6. Review the decision separately from the outcome.

Its main gap for an AI coach is mathematical precision. The course introduces GTO, combinations, pot odds, and mixed play, but it does not provide the formal extensive-form model, exact range-weighted EVs, solver convergence logic, exploitability metrics, or a full ICM engine. Those gaps are supplied in `gto_math_foundation.md`.

## Evidence-quality note

Automatic transcripts are a retrieval aid, not authoritative subtitles. Low-confidence segments—especially montage-heavy hand reviews—must be checked against the contact sheets and, when needed, the source video before being used as training labels. The future coach should preserve this provenance rule: no strategy claim without a recoverable hand state, assumptions, and source.

### Verified transcript audit

The final canonical transcript pass covers all 38 lesson ordinals exactly once. It contains 82,599 words across 9,011 timestamped segments and 214 representative evidence excerpts. Every transcript uses independent decoding windows (`condition_on_previous_text=false`) to prevent montage or music transitions from contaminating later windows. Mean per-lesson ASR log-probability ranges from -0.4904 to -0.3684; average speech coverage is 81.0%; no lesson crosses the index's low-confidence, sparse-speech, or low-coverage thresholds.

These measurements establish strong retrieval coverage, not perfect verbatim accuracy. Cards, stacks, sizes, and action order remain source-of-truth fields that require visual or structured-hand verification before a hand review becomes a training example.

## Visual-audit findings

Every lesson was sampled at 12 evenly spaced timestamps and reviewed in paired contact sheets. The visual evidence supports the curriculum map above and adds several details that speech recognition alone can miss:

- Videos 2–5 explicitly show position diagrams, range matrices, board/range examples, the hybrid GTO/exploitative principle, and a pot-odds formula.
- Videos 7–16 alternate conceptual instruction with complete table examples. Visible slides emphasize balancing check-raises with calls, narrowing an opponent's range after a 3-bet call, blocker-aware overbet selection, more polarized multiway betting, and opponent-conditioned mixed frequencies.
- Video 17's visible leak taxonomy includes open-limping, folding too often heads-up, overattaching to premium hands, calling too often on rivers, and requiring stronger holdings as prior aggression increases.
- Videos 18–22 separate tournament stages from cash play. The visible reminders reinforce relative stacks over an undifferentiated “average stack” and discourage routine open-limping in cash games.
- Videos 23–32 use physical demonstrations and archival hands for tells, table talk, image, metagame, and profiling. These observations are contextual evidence rather than deterministic hand-strength labels; player tendencies visibly change with the opponent and situation.
- Videos 33–38 move from strategy selection to sustainable practice: judge game selection by hours and decision quality rather than short-term results, manage bankroll and variance, use meditation/visualization in off-felt training, treat poker as a business, and adapt online study to digital hand histories.

The contact sheets are evidence maps, not substitutes for the source footage. Exact hand reconstruction should come from the original video or a structured hand history whenever cards, stacks, action order, or sizing are material.
