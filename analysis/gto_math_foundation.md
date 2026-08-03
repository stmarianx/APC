# Poker strategy and GTO mathematical foundation

## Scope and evidence

- Primary local corpus: 38 Daniel Negreanu poker lessons, 7.673 hours, 6.687 GB.
- Format: H.264 video with AAC audio; no embedded subtitle tracks.
- Local evidence products: media manifest; 38 corrected time-coded transcripts totaling 82,599 words and 9,011 segments; a 214-excerpt evidence index; and evenly sampled contact sheets for every lesson.
- Game focus inferred from the course: no-limit Texas Hold'em, spanning cash games, tournaments, multiway pots, live reads, and professional practice.
- Important limitation: the course is practical and heuristic. It is not a solver specification and does not establish exact equilibrium frequencies for fully specified games. Formal GTO claims below are grounded in game theory and poker-AI research, not inferred from teaching anecdotes.

## Course concept map

| Module | Videos | Main contribution | Formalization needed for a coach |
|---|---:|---|---|
| Strategic primitives | 1–5 | Position, ranges, board texture, hand review, introductory game theory/math | State schema, weighted ranges, Bayesian updates, equity and EV |
| Betting-tree decisions | 6–17 | C-betting, check-raising, 3-betting, bluffing, sizing, overbets, multiway play, mixed strategy, mistakes | Legal action tree, pot geometry, fold equity, indifference, mixed frequencies |
| Format-dependent utility | 18–22 | Early/middle, bubble, late/final-table, universal tournament, cash-game adjustments | Chip EV, payout utility, ICM, risk premium, rake, stack depth |
| Human information and exploitation | 23–32 | Masking/spotting tells, table talk, thought process, tilt, image, metagame, profiling | Opponent model, observation reliability, posterior uncertainty, constrained exploitation |
| Professional process | 33–38 | Game selection, bankroll, off-felt training, poker life, online-play bonus | Variance, risk of ruin, study loop, hand-history ingestion and evaluation |

## 1. State, uncertainty, and utility

Represent a decision as an information state rather than as a single visible hand:

\[
s=(H_i,B,P,S,\text{positions},\text{action history},\text{rules})
\]

Here, \(H_i\) is the acting player's private hand, \(B\) is the public board, \(P\) is the pot, \(S\) contains effective stacks, and `rules` includes blinds, antes, rake, legal bet sizes, number of players, and tournament payouts. A GTO answer is undefined until these parameters are fixed.

An extensive-form game can be written as:

\[
G=(N,H,Z,A,P_c,\mathcal I,u)
\]

where \(N\) is the set of players; \(H\) histories; \(Z\) terminal histories; \(A(h)\) legal actions; \(P_c\) chance; \(\mathcal I\) information sets; and \(u_i(z)\) player utility.

A behavior strategy gives an action distribution at each information set:

\[
\sigma_i(I,a)\ge0,\qquad \sum_{a\in A(I)}\sigma_i(I,a)=1
\]

The output of a solver is therefore normally a mixed distribution—not one deterministic action.

## 2. Combinatorics, ranges, blockers, and equity

Texas Hold'em begins with \(\binom{52}{2}=1326\) two-card combinations:

- Pocket pair: 6 combinations.
- Suited non-pair hand: 4 combinations.
- Offsuit non-pair hand: 12 combinations.
- One opponent has \(\binom{50}{2}=1225\) possible hole-card combinations before conditioning on action.

A weighted range \(R\) assigns weight \(w_h\in[0,1]\) to every compatible hand \(h\). Blockers remove incompatible combinations and change the conditional distribution.

Bayesian range updating after action \(a\):

\[
P(h\mid a,s)=\frac{P(a\mid h,s)P(h\mid s)}{\sum_{h'}P(a\mid h',s)P(h'\mid s)}
\]

Weighted equity of hero hand \(x\) against range \(R\) on board \(B\):

\[
\operatorname{Eq}(x,R\mid B)=
\frac{\sum_{h\in R}w_h\,\operatorname{Eq}(x,h\mid B)}{\sum_{h\in R}w_h}
\]

For \(o\) clean outs with one card to come and \(U\) unseen cards:

\[
P(\text{hit})=\frac{o}{U}
\]

With two cards to come after the flop:

\[
P(\text{hit by river})=1-\frac{47-o}{47}\frac{46-o}{46}
\]

The familiar rules of two and four are mental approximations, not replacements for exact combinatorics or range-vs-range simulation.

## 3. Expected value and pot geometry

For action \(a\):

\[
EV(a\mid s)=\sum_o P(o\mid a,s)\,u(o)
\]

Facing a bet \(B\) into pot \(P\), a call costs \(B\) and produces a final pot \(P+2B\). Ignoring future betting, rake, and ties, break-even call equity is:

\[
q_{\text{call}}=\frac{B}{P+2B}
\]

For a bluff risking \(B\) to win \(P\), break-even fold frequency is:

\[
f_{\text{bluff}}=\frac{B}{P+B}
\]

If \(x=B/P\) is the bet size as a fraction of pot, the idealized river polar model gives:

\[
\text{MDF}=\frac{1}{1+x},\quad
\frac{\text{bluffs}}{\text{value}}=\frac{x}{1+x},\quad
\text{bluff share}=\frac{x}{1+2x}
\]

The equality between bluff share and break-even call equity is an indifference result for a simplified heads-up river model. It is not a universal street-by-street defense rule.

| Bet size | Break-even call equity | Bluff break-even folds | MDF | Bluff:value |
|---:|---:|---:|---:|---:|
| 25% pot | 16.7% | 20.0% | 80.0% | 0.20 |
| 33% pot | 19.9% | 24.8% | 75.2% | 0.25 |
| 50% pot | 25.0% | 33.3% | 66.7% | 0.33 |
| 75% pot | 30.0% | 42.9% | 57.1% | 0.43 |
| 100% pot | 33.3% | 50.0% | 50.0% | 0.50 |
| 150% pot | 37.5% | 60.0% | 40.0% | 0.60 |
| 200% pot | 40.0% | 66.7% | 33.3% | 0.67 |

Stack-to-pot ratio controls future betting geometry:

\[
SPR=\frac{\text{effective stack at street start}}{\text{pot at street start}}
\]

Position changes equity realization. A useful diagnostic is:

\[
EQR=\frac{\text{realized EV-derived equity}}{\text{raw showdown equity}}
\]

Being in position, retaining nut hands, and controlling the betting sequence often increases realization; being capped or out of position often reduces it.

## 4. Range interaction and betting strategy

Board texture matters because it transforms both players' distributions. The important objects are:

- Range advantage: one range has higher average equity.
- Nut advantage: one range contains more of the strongest combinations.
- Range connectivity: how many combinations interact with the board.
- Cappedness: whether prior actions make top-end hands unlikely.
- Blocker effects: how a candidate hand changes the opponent's value and bluff combinations.

These quantities feed bet sizing and frequency:

- Small bets can leverage broad range advantage and deny equity cheaply.
- Large bets and overbets usually require a more polarized distribution and nut advantage.
- Check-raises and 3-bets need both value and appropriate bluff/semi-bluff candidates.
- Multiway pots reduce fold equity, strengthen continuing ranges, and generally demand tighter value/bluff thresholds.

The coach should describe a line through action EVs and frequency—not by asserting that a hand class “always” takes one action.

## 5. Nash equilibrium, regret, and CFR

A strategy profile \(\sigma^*\) is a Nash equilibrium if no player gains by unilateral deviation:

\[
u_i(\sigma_i^*,\sigma_{-i}^*)\ge u_i(\sigma_i',\sigma_{-i}^*)
\quad\forall i,\sigma_i'
\]

For a history \(h\), reach probability is the product of action probabilities along the path. CFR separates player \(i\)'s contribution from everyone else:

\[
\pi^\sigma(h)=\pi_i^\sigma(h)\pi_{-i}^\sigma(h)
\]

Counterfactual value of action \(a\) at information set \(I\):

\[
v_i^\sigma(I,a)=
\sum_{h\in I}\pi_{-i}^\sigma(h)
\sum_{z\sqsupset h\cdot a}\pi^\sigma(h\cdot a,z)u_i(z)
\]

Instantaneous and cumulative counterfactual regret:

\[
r_i^t(I,a)=v_i^{\sigma^t}(I,a)-v_i^{\sigma^t}(I),\qquad
R_i^T(I,a)=\sum_{t=1}^T r_i^t(I,a)
\]

Regret matching chooses the next strategy from positive regret:

\[
\sigma_i^{T+1}(I,a)=
\frac{[R_i^T(I,a)]_+}{\sum_b[R_i^T(I,b)]_+}
\]

If there is no positive regret, use a fallback distribution such as uniform play. In two-player zero-sum games, the average strategy—not necessarily the final iterate—converges toward Nash equilibrium as regret vanishes.

Exploitability evaluates how much a best responder can gain. One convention is NashConv:

\[
\operatorname{NashConv}(\sigma)=
\sum_i\left[u_i(BR_i(\sigma_{-i}),\sigma_{-i})-u_i(\sigma)\right]
\]

Always state the reporting convention because some tools divide the two-player NashConv by two and label that value “exploitability.”

## 6. Scaling poker solvers

Full no-limit hold'em is too large for naive traversal. Practical systems combine:

1. Card/information abstraction: bucket strategically similar private/public states.
2. Action abstraction: restrict the continuous bet-size space.
3. CFR variants: CFR+, MCCFR, discounting, pruning, or neural approximation.
4. Blueprint strategy: solve a coarse full-game policy offline.
5. Subgame resolving: refine the reached public state online in a permitted simulator.
6. Value networks/public belief states: approximate continuation values while preserving uncertainty over private hands.

Research landmarks:

- CFR introduced counterfactual regret decomposition for large imperfect-information games.
- CFR+ essentially solved heads-up limit hold'em.
- DeepStack combined continual resolving with learned counterfactual values.
- Libratus combined a blueprint, nested endgame solving, and self-improvement.
- Pluribus combined self-play/MCCFR, abstraction, and real-time search for six-player poker, without a general multiplayer Nash guarantee.
- Deep CFR replaced hand-built abstraction with neural approximation of CFR behavior.
- ReBeL used public belief states to combine self-play reinforcement learning and search with two-player zero-sum convergence guarantees.

## 7. Tournament utility, ICM, and risk premium

Cash-game chip utility is approximately linear before rake and bankroll preferences. Tournament utility is nonlinear because chips convert to payouts through stack distributions and prize structure.

In the Independent Chip Model, probability of winning is proportional to chip share:

\[
P(i\text{ finishes 1st})=\frac{s_i}{\sum_j s_j}
\]

For an ordered finish prefix \((p_1,\ldots,p_k)\):

\[
P(p_1,\ldots,p_k)=
\prod_{r=1}^k
\frac{s_{p_r}}{\sum_j s_j-\sum_{m<r}s_{p_m}}
\]

ICM equity is the payout-weighted sum of finish probabilities:

\[
V_i^{ICM}=\sum_k P(i\text{ finishes }k)\,\text{payout}_k
\]

If losing an all-in costs tournament utility \(L\) and winning gains \(G\), the break-even win probability is:

\[
q_{ICM}=\frac{L}{L+G}
\]

The excess of this threshold over the chip-EV threshold is one way to express risk premium. Bubble factor, opponent stack coverage, pay jumps, and bounties alter the utility function and therefore optimal ranges.

## 8. Variance, bankroll, and evidence quality

For per-hand result \(X\) with mean \(\mu\) and variance \(\sigma^2\), after \(n\) independent observations:

\[
E\left[\sum X\right]=n\mu,\qquad
\operatorname{Var}\left(\sum X\right)=n\sigma^2,\qquad
SE(\bar X)=\frac{\sigma}{\sqrt n}
\]

Poker hands are not perfectly independent when opponents adapt, but the equations explain why short-term results are weak evidence of strategic quality. The coach should report sample size, uncertainty, and decision EV separately from realized outcome.

For a simple binary bet with net odds \(b:1\), win probability \(p\), and loss probability \(q=1-p\), full Kelly is:

\[
f^*=\frac{bp-q}{b}
\]

Practical bankroll policy normally uses a fraction of Kelly and explicit stop/risk constraints because estimates are uncertain and poker returns are not a stationary binary game.

## 9. GTO baseline plus constrained exploitation

Pure best response to a noisy opponent model can be highly exploitable. A coach should keep three strategies distinct:

- \(\pi_{GTO}\): low-exploitability baseline.
- \(\pi_{BR}(\hat\pi_{opp})\): best response to an estimated opponent policy.
- \(\pi_{safe}\): exploitative policy constrained by model uncertainty or exploitability budget.

A useful formulation is:

\[
\max_{\pi} E[u(\pi,\hat\pi_{opp})]
\quad\text{subject to}\quad
\operatorname{Exploitability}(\pi)\le\varepsilon
\]

Observed tells, timing, table talk, and player profiles update \(\hat\pi_{opp}\); they do not override mathematical consistency. Reliability and sample size should control how far the policy may deviate from baseline.

For a binary tendency such as “folded to flop c-bet,” use a shrinkage estimator rather than a raw percentage. With prior \(\theta\sim\operatorname{Beta}(\alpha,\beta)\), \(k\) observed folds in \(n\) opportunities gives:

\[
\theta\mid D\sim\operatorname{Beta}(\alpha+k,\beta+n-k),\qquad
E[\theta\mid D]=\frac{\alpha+k}{\alpha+\beta+n}
\]

The posterior interval, not just the mean, should govern exploit strength. Style features derived from the user's own hands may include VPIP, PFR, 3-bet, fold-to-3-bet, c-bet, fold-to-c-bet, aggression frequency, showdown rate, sizing distributions, and position/stack-depth splits. Every statistic needs an opportunity count and should be segmented only when the sample remains adequate.

## 10. Product interpretation of the PokerListings training guide

The linked PokerListings guide groups 13 products into six apps and seven training sites:

- Apps: TOK Learn, DTO, Tournament Cruncher, Poker Cruncher, Poker Dealmaker ICM, and SnapShove.
- Sites/resources: Run It Once, PokerCoaching, Upswing Poker, Tournament Poker Edge, Raise Your Edge, the Negreanu/Ivey Masterclasses, and PokerListings' pot-odds guide.

The durable product lesson is the learning-mode taxonomy, not the article's rankings or prices:

- Quizzes for recall and fundamentals.
- Scenario trainers with immediate action feedback.
- Equity, pot-odds, ICM, deal, and push/fold calculators.
- Structured video curricula.
- Hand review and simulations.

Our coach should combine these modes around a single state model and evidence trail: every recommendation should show assumptions, action frequencies, action EVs, EV loss versus baseline, and the concepts responsible for the result.

## 11. Recommended coach boundary and architecture

PokerStars' current rules prohibit real-time action advice, advanced range/ICM/Nash calculations while its software is open, and datamining or mass-shared opponent databases. The terms define play-money games as part of the service, and the card-room rules say they apply to all games. Therefore the PokerStars-compatible workflow is:

1. Save the user's own play-money hand histories.
2. Close the PokerStars client.
3. Import and normalize those histories.
4. Reconstruct decisions and player-style features only from hands the user played.
5. Analyze decisions offline against cached solutions or a solver.
6. Replay the spots in an independent training table with live feedback.

Target architecture:

\[
\text{own hand history}
\rightarrow\text{parser/validator}
\rightarrow\text{canonical state}
\rightarrow\begin{cases}
\text{range/equity engine}\\
\text{solution lookup/resolver}\\
\text{opponent model with uncertainty}
\end{cases}
\rightarrow\text{EV/frequency explanation}
\rightarrow\text{training drill}
\]

A platform adapter may enable live integration only when the platform supplies an explicit API or written authorization for that exact use.

## Sources

- PokerListings, “13 Best Poker Training Apps & Sites for 2026”: https://www.pokerlistings.com/blog/poker-training-apps-sites
- PokerListings, pot odds and equity: https://www.pokerlistings.com/poker-strategies/texas-holdem/how-to-calculate-pot-odds-and-equity-equity
- Zinkevich et al., “Regret Minimization in Games with Incomplete Information”: https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html
- Tammelin et al., “Solving Heads-Up Limit Texas Hold'em”: https://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf
- Moravčík et al., “DeepStack”: https://poker.cs.ualberta.ca/publications/17science.pdf
- Brown and Sandholm, “Libratus”: https://pubmed.ncbi.nlm.nih.gov/29249696/
- Brown and Sandholm, “Pluribus”: https://noambrown.github.io/papers/19-Science-Superhuman.pdf
- Brown et al., “Deep Counterfactual Regret Minimization”: https://proceedings.mlr.press/v97/brown19b
- Brown et al., “ReBeL”: https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html
- Google DeepMind, OpenSpiel algorithms: https://github.com/google-deepmind/open_spiel/blob/master/docs/algorithms.md
- PokerStars third-party tools policy: https://www.pokerstars.com/poker/room/prohibited/
- PokerStars play-money hand histories: https://www.pokerstars.com/help/articles/hh-pm-older-7-days/40172/
