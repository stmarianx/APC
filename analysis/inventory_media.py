from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import av


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent


def seconds_for(container: av.container.InputContainer) -> float:
    if container.duration is not None:
        return float(container.duration / av.time_base)
    durations = [
        float(stream.duration * stream.time_base)
        for stream in container.streams
        if stream.duration is not None and stream.time_base is not None
    ]
    return max(durations, default=0.0)


def stream_record(stream: av.stream.Stream) -> dict[str, object]:
    codec = stream.codec_context
    record: dict[str, object] = {
        "index": stream.index,
        "type": stream.type,
        "codec": codec.name,
        "language": stream.metadata.get("language"),
        "bit_rate": codec.bit_rate,
    }
    if stream.type == "video":
        record.update(
            width=codec.width,
            height=codec.height,
            average_rate=float(stream.average_rate) if stream.average_rate else None,
        )
    elif stream.type == "audio":
        record.update(
            sample_rate=codec.sample_rate,
            channels=codec.channels,
        )
    return record


def main() -> None:
    videos = sorted(
        (path for path in ROOT.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.name.lower(),
    )
    manifest: list[dict[str, object]] = []
    for ordinal, path in enumerate(videos, start=1):
        with av.open(str(path)) as container:
            streams = [stream_record(stream) for stream in container.streams]
            duration_seconds = seconds_for(container)
        record = {
            "ordinal": ordinal,
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "duration_seconds": round(duration_seconds, 3),
            "duration_minutes": round(duration_seconds / 60, 2),
            "streams": streams,
        }
        manifest.append(record)
        print(f"{ordinal:02d}/38  {duration_seconds / 60:6.2f} min  {path.name}", flush=True)

    total_seconds = sum(float(item["duration_seconds"]) for item in manifest)
    payload = {
        "video_count": len(manifest),
        "total_duration_seconds": round(total_seconds, 3),
        "total_duration_hours": round(total_seconds / 3600, 3),
        "videos": manifest,
    }
    json_path = OUTPUT_DIR / "media_manifest.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = OUTPUT_DIR / "media_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ordinal", "file", "size_bytes", "duration_seconds", "duration_minutes"],
        )
        writer.writeheader()
        for item in manifest:
            writer.writerow({key: item[key] for key in writer.fieldnames})

    print(f"TOTAL  {total_seconds / 3600:.3f} hours across {len(manifest)} videos")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    sys.exit(main())
