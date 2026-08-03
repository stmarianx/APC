from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from apc.perception.table_locator import detect_table_box
from apc.perception.viewport import NormalizedBox
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


def box_iou(left: NormalizedBox, right: NormalizedBox) -> float:
    intersection_left = max(left.x, right.x)
    intersection_top = max(left.y, right.y)
    intersection_right = min(left.x + left.width, right.x + right.width)
    intersection_bottom = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, intersection_right - intersection_left) * max(
        0.0, intersection_bottom - intersection_top
    )
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_table_locator(
    manifest_path: str | Path,
    *,
    split: str = "test",
) -> dict[str, object]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    validation = validate_manifest(manifest_file, require_images=True)
    if not validation["valid"]:
        raise ValueError("Dataset manifest is invalid")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    sessions = set(manifest["splits"][split])
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    for relative in manifest["annotation_files"]:
        annotation_path = (manifest_file.parent / relative).resolve()
        annotation: dict[str, Any] = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )
        if annotation["capture_session_id"] not in sessions:
            continue
        image_path = (annotation_path.parent / annotation["image"]["path"]).resolve()
        prediction = detect_table_box(image_path)
        latencies.append(float(prediction["latency_ms"]))
        expected = NormalizedBox.from_dict(annotation["objects"]["table"])
        predicted_payload = prediction["table_box"]
        iou = (
            box_iou(expected, NormalizedBox.from_dict(predicted_payload))
            if predicted_payload is not None
            else 0.0
        )
        rows.append(
            {
                "sample_id": annotation["sample_id"],
                "capture_session_id": annotation["capture_session_id"],
                "image_sha256": annotation["image"]["sha256"],
                "status": prediction["status"],
                "confidence": prediction["confidence"],
                "expected_table_box": expected.to_dict(),
                "predicted_table_box": predicted_payload,
                "iou": iou,
            }
        )
    detected = [row for row in rows if row["predicted_table_box"] is not None]
    ordered_latency = sorted(latencies)
    p95_index = max(0, math.ceil(len(ordered_latency) * 0.95) - 1) if ordered_latency else 0
    return {
        "schema_version": "1.0.0",
        "dataset_id": manifest["dataset_id"],
        "split": split,
        "frames": len(rows),
        "detected_frames": len(detected),
        "detection_rate": len(detected) / len(rows) if rows else 0.0,
        "mean_iou": statistics.fmean(float(row["iou"]) for row in rows) if rows else 0.0,
        "iou_at_0_8": sum(float(row["iou"]) >= 0.8 for row in rows) / len(rows) if rows else 0.0,
        "latency_ms_p95": ordered_latency[p95_index] if ordered_latency else None,
        "predictions_sha256": canonical_sha256(rows),
        "rows": rows,
        "limitations": [
            "Synthetic evaluation does not establish controlled-visible calibration.",
            "Test-split results are frozen audit evidence and must not be used for tuning.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate APC table localization on a manifest split.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_table_locator(args.manifest, split=args.split)
        rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
