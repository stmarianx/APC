from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
TRANSCRIPT_DIR = ANALYSIS_DIR / "transcripts"
EXPECTED_ORDINALS = set(range(1, 39))

TOPICS: dict[str, tuple[str, ...]] = {
    "position": ("position", "button", "cutoff", "under the gun", "small blind", "big blind"),
    "ranges": ("range", "combination", "combo", "narrow", "wide", "premium hand"),
    "board texture": ("board", "texture", "flop", "turn", "river", "rainbow", "wet", "dry"),
    "equity and odds": ("equity", "pot odds", "odds", "outs", "expected value", "ev ", "percentage"),
    "bet sizing": ("bet size", "sizing", "half pot", "pot-sized", "overbet", "minimum bet"),
    "aggression": ("continuation bet", "c-bet", "check raise", "check-raise", "three bet", "3-bet", "four bet"),
    "bluffing": ("bluff", "blocker", "fold equity", "semi-bluff", "polarized"),
    "mixed strategy": ("mixed strategy", "frequency", "random", "unpredictable", "balance", "game theory"),
    "multiway": ("multi-way", "multiway", "three players", "four players", "heads up"),
    "tournaments": ("tournament", "bubble", "final table", "payout", "icm", "short stack", "big blinds"),
    "cash games": ("cash game", "cash games", "buy-in", "deep stack"),
    "tells and live reads": ("tell", "body language", "eye contact", "breathing", "pulse", "physical"),
    "player modeling": ("profile", "opponent", "tight", "loose", "aggressive", "passive", "weak player"),
    "psychology and meta": ("tilt", "emotion", "table image", "metagame", "table talk", "ego"),
    "game and bankroll": ("game selection", "bankroll", "risk", "variance", "stakes", "losing session"),
    "study process": ("study", "training", "review", "mistake", "practice", "learn", "work on your game"),
}


def count_phrase(text: str, phrase: str) -> int:
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE))


def topic_counts(text: str) -> Counter[str]:
    return Counter(
        {
            topic: sum(count_phrase(text, phrase) for phrase in phrases)
            for topic, phrases in TOPICS.items()
        }
    )


def timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def evidence_for(segments: list[dict[str, object]], ranked_topics: list[str]) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    used_text: set[str] = set()
    for topic in ranked_topics[:6]:
        phrases = TOPICS[topic]
        candidates = []
        for segment in segments:
            text = str(segment["text"]).strip()
            score = sum(count_phrase(text, phrase) for phrase in phrases)
            if score and len(text) >= 35 and text not in used_text:
                candidates.append((score, len(text), segment))
        if candidates:
            _, _, segment = max(candidates, key=lambda item: (item[0], item[1]))
            text = str(segment["text"]).strip()
            used_text.add(text)
            evidence.append(
                {
                    "topic": topic,
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": text,
                }
            )
    return evidence


def main() -> int:
    records = []
    seen_ordinals: set[int] = set()
    for transcript_path in sorted(TRANSCRIPT_DIR.glob("*.json")):
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        ordinal_match = re.match(r"^(\d+)", str(payload["file"]))
        if not ordinal_match:
            raise SystemExit(f"Cannot derive lesson ordinal from {payload['file']}")
        ordinal = int(ordinal_match.group(1))
        if ordinal in seen_ordinals:
            raise SystemExit(f"Duplicate transcript for lesson {ordinal}: {transcript_path.name}")
        seen_ordinals.add(ordinal)
        segments = payload["segments"]
        full_text = " ".join(str(segment["text"]) for segment in segments)
        counts = topic_counts(full_text)
        ranked = [topic for topic, count in counts.most_common() if count > 0]
        logprobs = [float(segment["avg_logprob"]) for segment in segments]
        duration_seconds = float(payload.get("duration_seconds") or 0.0)
        word_count = len(re.findall(r"\b\w+\b", full_text))
        speech_seconds = sum(
            max(0.0, float(segment["end"]) - float(segment["start"])) for segment in segments
        )
        mean_logprob = round(sum(logprobs) / len(logprobs), 4) if logprobs else None
        words_per_minute = round(word_count / (duration_seconds / 60), 1) if duration_seconds else 0.0
        speech_coverage = round(speech_seconds / duration_seconds, 3) if duration_seconds else 0.0
        quality_flags = []
        if mean_logprob is not None and mean_logprob < -0.8:
            quality_flags.append("low ASR confidence")
        if words_per_minute < 35:
            quality_flags.append("sparse speech or montage")
        if speech_coverage < 0.25:
            quality_flags.append("low speech coverage")
        records.append(
            {
                "ordinal": ordinal,
                "file": payload["file"],
                "transcript_file": transcript_path.name,
                "duration_seconds": duration_seconds,
                "word_count": word_count,
                "words_per_minute": words_per_minute,
                "speech_coverage": speech_coverage,
                "segment_count": len(segments),
                "mean_avg_logprob": mean_logprob,
                "quality_flags": quality_flags,
                "decoding": payload.get("decoding", {}),
                "topic_counts": dict(counts.most_common()),
                "evidence": evidence_for(segments, ranked),
            }
        )

    missing = sorted(EXPECTED_ORDINALS - seen_ordinals)
    unexpected = sorted(seen_ordinals - EXPECTED_ORDINALS)
    if missing or unexpected or len(records) != len(EXPECTED_ORDINALS):
        raise SystemExit(
            "Transcript corpus is not canonical: "
            f"found {len(records)}, missing={missing}, unexpected={unexpected}"
        )
    records.sort(key=lambda record: int(record["ordinal"]))

    (ANALYSIS_DIR / "transcript_evidence_index.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    flagged_count = sum(bool(record["quality_flags"]) for record in records)
    independent_count = sum(
        record["decoding"].get("condition_on_previous_text") is False
        for record in records
        if record["decoding"]
    )
    lines = [
        "# Transcript evidence index",
        "",
        f"- Canonical lessons: {len(records)}/38",
        f"- Independent-window corrected transcripts: {independent_count}",
        f"- Lessons with one or more quality flags: {flagged_count}",
        "",
    ]
    for record in records:
        ordinal = int(record["ordinal"])
        duration_minutes = float(record["duration_seconds"] or 0.0) / 60
        top_topics = [
            f"{topic} ({count})"
            for topic, count in list(record["topic_counts"].items())[:6]
            if count > 0
        ]
        lines.extend(
            [
                f"## {ordinal:02d}. {record['file']}",
                "",
                f"- Duration: {duration_minutes:.2f} min",
                f"- Transcript: {record['word_count']} words; {record['segment_count']} segments; "
                f"{record['words_per_minute']} words/min; {record['speech_coverage']:.1%} speech coverage; "
                f"mean ASR log-probability {record['mean_avg_logprob']}",
                f"- Quality flags: {', '.join(record['quality_flags']) if record['quality_flags'] else 'none'}",
                f"- Decoder: {json.dumps(record['decoding'], ensure_ascii=False) if record['decoding'] else 'legacy baseline'}",
                f"- Strongest detected topics: {', '.join(top_topics) if top_topics else 'none'}",
                "",
            ]
        )
        for item in record["evidence"]:
            lines.append(
                f"- [{timestamp(float(item['start']))}] **{item['topic']}** — {item['text']}"
            )
        lines.append("")
    (ANALYSIS_DIR / "transcript_evidence_index.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Indexed {len(records)} transcripts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
