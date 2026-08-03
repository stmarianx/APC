from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = ANALYSIS_DIR / "models"
TRANSCRIPT_DIR = ANALYSIS_DIR / "transcripts"
DOMAIN_PROMPT = (
    "Daniel Negreanu teaches no-limit Texas Hold'em poker strategy. "
    "Poker terms include GTO, Nash equilibrium, range, equity, pot odds, implied odds, "
    "expected value, EV, blockers, bluff, value bet, continuation bet, c-bet, check-raise, "
    "three-bet, four-bet, overbet, polarized, condensed, capped range, ICM, bubble, "
    "big blind, small blind, under the gun, cutoff, button, flop, turn, river, and showdown."
)


def slug_for(path: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe the local poker-course videos.")
    parser.add_argument("--start", type=int, default=1, help="First 1-based video ordinal")
    parser.add_argument("--end", type=int, default=38, help="Last 1-based video ordinal")
    parser.add_argument("--model", default="distil-small.en")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batched", action="store_true", help="Use batched inference for speed")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Condition windows on prior text; disabled by default to prevent failure cascades",
    )
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_outputs(
    video: Path,
    info: object,
    segments: list[dict[str, object]],
    elapsed: float,
    output_suffix: str = "",
    decoding: dict[str, object] | None = None,
) -> None:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{slug_for(video)}{output_suffix}"
    payload = {
        "file": video.name,
        "language": getattr(info, "language", "en"),
        "language_probability": getattr(info, "language_probability", None),
        "duration_seconds": getattr(info, "duration", None),
        "duration_after_vad_seconds": getattr(info, "duration_after_vad", None),
        "transcription_elapsed_seconds": round(elapsed, 3),
        "decoding": decoding or {},
        "segments": segments,
    }
    (TRANSCRIPT_DIR / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    def stamp(seconds: float) -> str:
        minutes, remainder = divmod(seconds, 60)
        return f"{int(minutes):02d}:{remainder:05.2f}"

    plain_lines = [
        f"[{stamp(float(segment['start']))}-{stamp(float(segment['end']))}] "
        f"{str(segment['text']).strip()}"
        for segment in segments
    ]
    (TRANSCRIPT_DIR / f"{stem}.txt").write_text("\n".join(plain_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    videos = sorted(
        (path for path in ROOT.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.name.lower(),
    )
    selected = videos[max(args.start - 1, 0) : min(args.end, len(videos))]
    if not selected:
        raise SystemExit("No videos selected")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model} on CPU (int8, {args.threads} threads)", flush=True)
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        cpu_threads=args.threads,
        num_workers=1,
        download_root=str(MODEL_DIR),
    )
    transcriber = BatchedInferencePipeline(model=model) if args.batched else model

    for index, video in enumerate(selected, start=args.start):
        output_path = TRANSCRIPT_DIR / f"{slug_for(video)}{args.output_suffix}.json"
        if output_path.exists() and not args.overwrite:
            print(f"SKIP {index:02d}/38 {video.name}", flush=True)
            continue
        print(f"START {index:02d}/38 {video.name}", flush=True)
        started = time.perf_counter()
        options = dict(
            audio=str(video),
            language="en",
            task="transcribe",
            beam_size=args.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=None,
            condition_on_previous_text=args.condition_on_previous_text,
            word_timestamps=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=6,
            compression_ratio_threshold=2.2,
            temperature=0.0,
        )
        if args.batched:
            options["batch_size"] = args.batch_size
        segment_iter, info = transcriber.transcribe(**options)
        segments = []
        for segment in segment_iter:
            segments.append(
                {
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": segment.text.strip(),
                    "avg_logprob": round(float(segment.avg_logprob), 5),
                    "no_speech_prob": round(float(segment.no_speech_prob), 5),
                    "compression_ratio": round(float(segment.compression_ratio), 5),
                }
            )
        elapsed = time.perf_counter() - started
        write_outputs(
            video,
            info,
            segments,
            elapsed,
            args.output_suffix,
            decoding={
                "model": args.model,
                "device": "cpu",
                "compute_type": "int8",
                "threads": args.threads,
                "beam_size": args.beam_size,
                "batched": args.batched,
                "condition_on_previous_text": args.condition_on_previous_text,
                "vad_filter": True,
            },
        )
        media_seconds = float(getattr(info, "duration", 0.0) or 0.0)
        speed = media_seconds / elapsed if elapsed else 0.0
        print(
            f"DONE  {index:02d}/38 {elapsed / 60:.2f} min, "
            f"{speed:.2f}x realtime, {len(segments)} segments",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
