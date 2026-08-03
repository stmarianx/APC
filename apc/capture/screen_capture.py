from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageGrab

from apc.annotator.project import utc_now
from apc.tools.validate_dataset import canonical_sha256


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.right - self.left < 320 or self.bottom - self.top < 240:
            raise ValueError("Capture region must be at least 320x240 pixels")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def bbox(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    @classmethod
    def parse(cls, value: str) -> "CaptureRegion":
        try:
            parts = [int(part.strip()) for part in value.split(",")]
        except ValueError as error:
            raise ValueError("region must contain four comma-separated integers") from error
        if len(parts) != 4:
            raise ValueError("region must use left,top,right,bottom")
        return cls(*parts)


@dataclass(frozen=True)
class CapturePlan:
    session_id: str
    region: CaptureRegion
    frames: int
    interval_ms: int

    def __post_init__(self) -> None:
        if not self.session_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in self.session_id
        ):
            raise ValueError("session_id must use letters, numbers, dot, underscore or dash")
        if not 1 <= self.frames <= 10000:
            raise ValueError("frames must be between 1 and 10000")
        if self.interval_ms < 50:
            raise ValueError("interval_ms must be at least 50")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "session_id": self.session_id,
            "region": asdict(self.region),
            "frames": self.frames,
            "interval_ms": self.interval_ms,
            "mode": "read_only_pixels_no_input_control",
        }


Grabber = Callable[[tuple[int, int, int, int]], Image.Image]
Sleeper = Callable[[float], None]


def _default_grabber(bbox: tuple[int, int, int, int]) -> Image.Image:
    return ImageGrab.grab(bbox=bbox, all_screens=True)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def capture_frames(
    output: str | Path,
    plan: CapturePlan,
    *,
    grabber: Grabber | None = None,
    sleeper: Sleeper = time.sleep,
) -> dict[str, object]:
    root = Path(output).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Capture output must be new or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    plan_payload = plan.payload()
    plan_payload["created_at"] = utc_now()
    plan_payload["plan_sha256"] = canonical_sha256(plan.payload())
    _write_json(root / "capture_plan.json", plan_payload)

    take = grabber or _default_grabber
    rows: list[dict[str, object]] = []
    for index in range(plan.frames):
        timestamp_ms = index * plan.interval_ms
        image = take(plan.region.bbox())
        if not isinstance(image, Image.Image):
            raise ValueError("Screen grabber must return a Pillow image")
        image = image.convert("RGB")
        if image.size != (plan.region.width, plan.region.height):
            raise ValueError(
                f"Captured image is {image.width}x{image.height}; expected "
                f"{plan.region.width}x{plan.region.height}"
            )
        filename = f"frame-{index:06d}-{timestamp_ms:09d}ms.png"
        path = root / filename
        image.save(path, format="PNG", optimize=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "index": index,
                "timestamp_ms": timestamp_ms,
                "filename": filename,
                "sha256": digest,
                "width": image.width,
                "height": image.height,
            }
        )
        if index + 1 < plan.frames:
            sleeper(plan.interval_ms / 1000)

    report = {
        "schema_version": "1.0.0",
        "session_id": plan.session_id,
        "completed_at": utc_now(),
        "plan_sha256": plan_payload["plan_sha256"],
        "captured_frames": len(rows),
        "frames": rows,
        "frames_sha256": canonical_sha256(rows),
        "policy": {
            "read_only_pixels": True,
            "input_control": False,
            "network_transmission": False,
            "explicit_region_required": True,
        },
    }
    _write_json(root / "capture_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture an explicit visible-table screen region as fingerprinted PNG frames."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--region", required=True, type=CaptureRegion.parse)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--interval-ms", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = capture_frames(
            args.output,
            CapturePlan(
                session_id=args.session,
                region=args.region,
                frames=args.frames,
                interval_ms=args.interval_ms,
            ),
        )
        sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
