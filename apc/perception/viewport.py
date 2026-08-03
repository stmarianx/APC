from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apc.tools.validate_dataset import canonical_sha256


SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class NormalizedBox:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("normalized box coordinates must be finite")
        if (
            self.x < 0
            or self.y < 0
            or self.width <= 0
            or self.height <= 0
            or self.x + self.width > 1
            or self.y + self.height > 1
        ):
            raise ValueError("normalized box must stay inside the image")

    @classmethod
    def from_dict(cls, payload: object) -> "NormalizedBox":
        if not isinstance(payload, dict):
            raise ValueError("normalized box must be an object")
        try:
            return cls(*(float(payload[key]) for key in ("x", "y", "width", "height")))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("normalized box requires x, y, width and height") from error

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def pixels(self, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
        width, height = image_size
        return (
            self.x * width,
            self.y * height,
            self.width * width,
            self.height * height,
        )


@dataclass(frozen=True)
class ViewportCalibration:
    profile_id: str
    source_size: tuple[int, int]
    observed_table_box: NormalizedBox
    canonical_size: tuple[int, int]
    canonical_table_box: NormalizedBox
    source_kind: str = "verified_manual_table_box"

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must be non-empty")
        for size, name in ((self.source_size, "source_size"), (self.canonical_size, "canonical_size")):
            if len(size) != 2 or any(isinstance(value, bool) or int(value) < 64 for value in size):
                raise ValueError(f"{name} must contain width and height of at least 64 pixels")
        if self.source_kind not in {"verified_manual_table_box", "detected_table_box"}:
            raise ValueError("source_kind must declare verified manual or detected table geometry")

    @property
    def affine(self) -> dict[str, float]:
        source_left, source_top, source_width, source_height = self.observed_table_box.pixels(self.source_size)
        target_left, target_top, target_width, target_height = self.canonical_table_box.pixels(self.canonical_size)
        scale_x, scale_y = target_width / source_width, target_height / source_height
        return {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "translate_x": target_left - source_left * scale_x,
            "translate_y": target_top - source_top * scale_y,
        }

    def map_source_box(self, box: NormalizedBox) -> NormalizedBox:
        source_left, source_top, source_width, source_height = box.pixels(self.source_size)
        transform = self.affine
        target_width, target_height = self.canonical_size
        mapped = NormalizedBox(
            (source_left * transform["scale_x"] + transform["translate_x"]) / target_width,
            (source_top * transform["scale_y"] + transform["translate_y"]) / target_height,
            source_width * transform["scale_x"] / target_width,
            source_height * transform["scale_y"] / target_height,
        )
        return mapped

    def material(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "source_size": list(self.source_size),
            "observed_table_box": self.observed_table_box.to_dict(),
            "canonical_size": list(self.canonical_size),
            "canonical_table_box": self.canonical_table_box.to_dict(),
            "source_kind": self.source_kind,
            "affine": self.affine,
        }

    def to_dict(self) -> dict[str, object]:
        material = self.material()
        material["profile_sha256"] = canonical_sha256(material)
        return material

    def save(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "ViewportCalibration":
        payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        expected = payload.pop("profile_sha256", None)
        if payload.get("schema_version") != SCHEMA_VERSION or expected != canonical_sha256(payload):
            raise ValueError("viewport calibration fingerprint is invalid")
        profile = cls(
            str(payload["profile_id"]),
            tuple(int(value) for value in payload["source_size"]),
            NormalizedBox.from_dict(payload["observed_table_box"]),
            tuple(int(value) for value in payload["canonical_size"]),
            NormalizedBox.from_dict(payload["canonical_table_box"]),
            str(payload["source_kind"]),
        )
        if any(
            not math.isclose(float(payload["affine"][key]), value, rel_tol=0, abs_tol=1e-12)
            for key, value in profile.affine.items()
        ):
            raise ValueError("viewport calibration affine audit is inconsistent")
        return profile


def normalize_viewport(
    image_path: str | Path,
    calibration: ViewportCalibration,
    output_path: str | Path,
) -> dict[str, object]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Viewport normalization requires Pillow") from error
    source_path = Path(image_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    if source.size != calibration.source_size:
        raise ValueError(
            f"viewport source size {source.size} does not match calibration {calibration.source_size}"
        )
    transform = calibration.affine
    scaled_size = (
        max(1, round(source.width * transform["scale_x"])),
        max(1, round(source.height * transform["scale_y"])),
    )
    scaled = source.resize(scaled_size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", calibration.canonical_size, "black")
    offset = (round(transform["translate_x"]), round(transform["translate_y"]))
    canvas.paste(scaled, offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": calibration.profile_id,
        "profile_sha256": calibration.to_dict()["profile_sha256"],
        "source_image_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "normalized_image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_size": list(source.size),
        "canonical_size": list(canvas.size),
        "affine": transform,
        "scaled_size": list(scaled_size),
        "paste_offset": list(offset),
        "output_path": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize a visible table using a fingerprinted viewport calibration.")
    parser.add_argument("image", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = normalize_viewport(args.image, ViewportCalibration.load(args.calibration), args.output)
        if args.report:
            report = args.report.expanduser().resolve()
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
