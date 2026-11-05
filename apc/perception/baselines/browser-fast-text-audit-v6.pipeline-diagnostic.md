# Post-audit stage attribution and frame-key correction

The sealed v6 result remains unchanged. This separate diagnostic reuses its frames with the exact fast/wide v5 stack and four checkpoint hashes, without training or threshold changes. Results are post-audit reference analysis, not a new independent audit.

All three misses are locator/refinement-stage misses (no >=0.5 IoU match before verification):

- `h1-flop/frame.jpg`: board Qc.
- `h2-flop/frame.jpg`: hero 2h.
- `h2-turn/frame.jpg`: hero As.

The other 57 appearances are recognized correctly. No card is lost specifically at verification or glyph recognition in this run. Oracle crops in the sealed audit recognize all 60. The diagnostic does not yet isolate which decoder/refinement/recovery operation fails on each missing card.

The three false proposals in `negative-3/frame.jpg` all route to the incumbent bounded-surface verifier, with scores approximately 0.9984, 0.9845 and 0.8794. All have candidate=null because glyph extraction finds no complete usable ink pair. The newer text-fragment v5 ambiguous expert does not decide these cases. Scores are uncalibrated, no confirmed identity is emitted, and stronger bounded-expert development must preserve true compact-card retention.

Frame identity fix: diagnostic rows now include collection-relative frame_key plus hand_group/street. Benchmark repeatability and cross-variant checks group by collection-relative path, not basename. Different captures can legitimately share `frame.jpg`; they must not be conflated. Duplicate identical paths are refused in benchmarking. A regression uses distinct same-basename paths with different predictions and verifies correct grouping. Historical timing reports used unique basenames and remain unchanged; sealed v6 scoring processed actual paths independently, so its counts are unaffected.

Next: trace missing-card geometry/recovery, use new synthetic placement cases for development, and evaluate bounded-verifier changes separately from the ambiguous expert. Consider temporal evidence integration only with explicit uncertainty/abstention: previous hero observations are useful evidence, not permission to invent unseen board cards. Audit images remain excluded from training. Full-table/profile/strategy/latency gates remain open.
