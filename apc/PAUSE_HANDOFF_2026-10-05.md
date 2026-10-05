# APC cutoff handoff — 2026-10-05, 11:00 Europe/Bucharest

Stop development at the user-requested cutoff. Resume only after explicit user
authorization. The full APC objective is unchanged and remains incomplete.

Latest development commit before this handoff: `56513be`, pushed to origin/main.
Unrelated local `apc/perception/name_ocr_baseline.py` edits were preserved and
excluded from these commits.

Completed this work session:
- Spatial annotation targets, visibility masks and session-isolated loading.
- Region recognizer, reproducible training runner, confusion/audit metrics.
- Pixel-based card proposals and end-to-end candidate reading with abstention.
- Crop-shift diagnosis, jitter experiment and ink-normalized neural recognition.
- Optional rank/suit subregion labels, paired-glyph training, corner-symbol renderer.

Evidence and limitations are recorded in `perception/REGION_TRAINING.md`.
Full suite passed 262 tests before the final glyph-schema/design changes; final
targeted suite passed 22 tests. Synthetic whole-token recognition reached 180/180
cards on detected boxes, but the frozen unfamiliar table audit failed. Corner
paired recognition reached 14/20 cards with supplied glyph regions. No visible
table training/coaching readiness gate has passed.

Local ignored datasets/checkpoints remain under `apc/runs/`, including
`card-glyph-v1`, `card-glyph-pair-v1`, `corner-card-glyph-pair-v1`, and the
corresponding generated dataset directories. All training/test jobs started in
this work session returned terminal completion before this handoff.

Next required work:
- Independent rank/suit glyph localization, varied fonts, symbols and artwork.
- Verified controlled-visible dataset sessions; do not count synthetic/frozen audit frames.
- Calibrated uncertainty, fresh held-out domain audits and full-pipeline latency.
- Seat/card-role/numeric/action integration with temporal tracking and backend.
- Full goal-level readiness evaluation before notifying the user of readiness.
