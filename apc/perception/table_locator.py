from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from apc.perception.viewport import NormalizedBox


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    rows: list[tuple[int, int, int, int, int]] = []
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_x), int(start_y))])
        visited[start_y, start_x] = True
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        area = 0
        while queue:
            x, y = queue.popleft()
            area += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for next_x, next_y in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if (
                    0 <= next_x < width
                    and 0 <= next_y < height
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_x, next_y))
        rows.append((min_x, min_y, max_x + 1, max_y + 1, area))
    return rows


def detect_table_box(
    image_path: str | Path,
    *,
    analysis_width: int = 320,
) -> dict[str, object]:
    started = time.perf_counter()
    path = Path(image_path).expanduser().resolve()
    with Image.open(path) as opened:
        source = opened.convert("RGB")
    scale = min(1.0, analysis_width / source.width)
    analysis_size = (
        max(64, round(source.width * scale)),
        max(64, round(source.height * scale)),
    )
    resized = source.resize(analysis_size, Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    height, width, _ = pixels.shape
    border_width = max(2, round(min(width, height) * 0.015))
    border_pixels = np.concatenate(
        (
            pixels[:border_width].reshape(-1, 3),
            pixels[-border_width:].reshape(-1, 3),
            pixels[:, :border_width].reshape(-1, 3),
            pixels[:, -border_width:].reshape(-1, 3),
        )
    )
    background = np.median(border_pixels, axis=0)
    center_patch = pixels[
        round(height * 0.38) : round(height * 0.62),
        round(width * 0.38) : round(width * 0.62),
    ]
    center = np.median(center_patch.reshape(-1, 3), axis=0)
    separation = float(np.linalg.norm(center - background))
    base = {
        "schema_version": "1.0.0",
        "image": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": source.width,
            "height": source.height,
        },
        "analysis_size": list(analysis_size),
        "diagnostics": {
            "background_rgb": [round(float(value), 3) for value in background],
            "center_rgb": [round(float(value), 3) for value in center],
            "center_border_separation": separation,
        },
    }
    if not math.isfinite(separation) or separation < 12:
        return {
            **base,
            "status": "abstain_low_table_contrast",
            "table_box": None,
            "confidence": 0.0,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    distance_center = np.linalg.norm(pixels - center, axis=2)
    distance_background = np.linalg.norm(pixels - background, axis=2)
    center_radius = min(95.0, max(24.0, separation * 0.9))
    mask = (
        (distance_center <= center_radius)
        & (distance_background >= max(8.0, separation * 0.28))
    )
    candidates: list[tuple[float, NormalizedBox, dict[str, float]]] = []
    component_diagnostics: list[dict[str, float]] = []
    for left, top, right, bottom, area in _components(mask):
        box_width, box_height = right - left, bottom - top
        width_ratio, height_ratio = box_width / width, box_height / height
        aspect = box_width / max(box_height, 1)
        area_ratio = area / (width * height)
        center_x = (left + right) / (2 * width)
        center_y = (top + bottom) / (2 * height)
        center_distance = math.hypot(center_x - 0.5, center_y - 0.5)
        component_diagnostics.append(
            {
                "component_area_ratio": area_ratio,
                "box_width_ratio": width_ratio,
                "box_height_ratio": height_ratio,
                "aspect_ratio": aspect,
                "center_distance": center_distance,
            }
        )
        if not (
            0.35 <= width_ratio <= 0.98
            and 0.20 <= height_ratio <= 0.90
            and 1.5 <= aspect <= 3.5
            and area_ratio >= 0.08
            and center_distance <= 0.24
        ):
            continue
        target_table_aspect = 2.25
        desired_height = box_width / target_table_aspect
        if desired_height < box_height:
            vertical_center = (top + bottom) / 2
            top = max(0, round(vertical_center - desired_height / 2))
            bottom = min(height, round(vertical_center + desired_height / 2))
            box_height = bottom - top
            height_ratio = box_height / height
            aspect = box_width / max(box_height, 1)
        margin_x, margin_y = 1 / width, 1 / height
        x = max(0.0, left / width - margin_x)
        y = max(0.0, top / height - margin_y)
        right_normalized = min(1.0, right / width + margin_x)
        bottom_normalized = min(1.0, bottom / height + margin_y)
        box = NormalizedBox(
            x,
            y,
            right_normalized - x,
            bottom_normalized - y,
        )
        geometry_score = max(0.0, 1.0 - center_distance / 0.24)
        score = area_ratio * (0.6 + 0.4 * geometry_score)
        candidates.append(
            (
                score,
                box,
                {
                    "component_area_ratio": area_ratio,
                    "box_width_ratio": width_ratio,
                    "box_height_ratio": height_ratio,
                    "aspect_ratio": aspect,
                    "center_distance": center_distance,
                },
            )
        )
    if not candidates:
        base["diagnostics"]["largest_components"] = sorted(
            component_diagnostics,
            key=lambda row: row["component_area_ratio"],
            reverse=True,
        )[:5]
        return {
            **base,
            "status": "abstain_no_table_geometry",
            "table_box": None,
            "confidence": 0.0,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
    score, box, geometry = max(candidates, key=lambda row: row[0])
    contrast_score = min(1.0, separation / 80.0)
    confidence = min(0.75, contrast_score * min(1.0, score / 0.32))
    base["diagnostics"].update(geometry)
    base["diagnostics"]["candidate_count"] = len(candidates)
    return {
        **base,
        "status": "detected_uncalibrated",
        "table_box": box.to_dict(),
        "confidence": confidence,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "limitations": [
            "Heuristic border/center segmentation is not a trained object detector.",
            "Confidence is uncalibrated and cannot open the recommendation gate.",
            "Controlled-visible evaluation is required before automatic viewport use.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect a central poker-table viewport with safe abstention.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = detect_table_box(args.image)
        rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0 if result["table_box"] is not None else 3
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
