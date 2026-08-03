# Negreanu course research workspace

This folder contains the first-phase evidence and strategy foundation for an offline poker coach.

The tested implementation foundation now lives in `../coach/`; the workspace-level `../README.md` links both phases.

## Primary outputs

- `course_video_analysis.md` — video-by-video curriculum analysis and coach primitives.
- `gto_math_foundation.md` — mathematical model spanning ranges, equity, EV, pot geometry, equilibrium, CFR, ICM, bankroll, and uncertainty-aware exploitation.
- `transcript_evidence_index.md` / `.json` — canonical 38-lesson searchable evidence index with decoder provenance and transcript quality statistics.
- `media_manifest.csv` / `.json` — source inventory and technical metadata for all 38 videos.
- `contact_sheets/` — 12 evenly sampled frames per source video.
- `contact_sheet_gallery/` — paired lesson galleries used for the full visual audit.
- `transcripts/` — time-coded automatic transcripts in text and structured JSON.
- `research_verification.json` — machine-readable completion audit for sources, transcripts, evidence, visual samples, reports, and graph structure.
- `verify_research_artifacts.py` — repeatable strict verifier used to produce that audit.

## Evidence policy

Automatic speech recognition is used for retrieval, not as ground truth. Strategy claims should retain source video, timestamp, recovered table state, assumptions, and confidence. Hand-review footage must be reconstructed from visible cards/actions or a structured hand history before it becomes a training label.

The verified corpus contains 38/38 corrected transcripts, 82,599 words, 9,011 timestamped segments, and 214 indexed evidence excerpts. The strict index builder rejects missing or duplicate lesson ordinals.

## Intended product path

1. Parse the user's own saved hand histories into a canonical game state.
2. Calculate action EV, frequencies, and EV loss offline.
3. Maintain uncertainty-aware player-style profiles from hands the user participated in.
4. Turn recurring leaks into drills on an independent training table.
5. Permit any live integration only where the target environment explicitly authorizes it.
