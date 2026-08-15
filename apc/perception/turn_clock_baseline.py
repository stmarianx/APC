from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apc.perception.baseline import (
    _image_path,
    _manifest_annotations,
    _percentile,
)
from apc.perception.card_baseline import fit_exemplars
from apc.perception.table_state_baseline import (
    _mean_box,
    _numeric_training_rows,
    _predict_numeric_token,
)
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


SCHEMA_VERSION = "1.0.0"
MODEL_KIND = "synthetic_segmented_turn_clock_pixel_baseline"
CLOCK_FEATURE = {"size": [48, 14], "representation": "local_contrast"}
CLOCK_REGION_WIDTH_PX = 100.0
CLOCK_PREFIX_PX = 14.0


def train_turn_clock_baseline(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    *,
    seed: int = 2026081501,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    train_sessions = {str(value) for value in manifest["splits"]["train"]}
    rows = [
        (path, item)
        for path, item in annotations
        if str(item["capture_session_id"]) in train_sessions
        and isinstance(item["objects"].get("turn_clock"), dict)
    ]
    if not rows:
        raise ValueError("training split has no visible turn-clock examples")
    shape_rows: list[tuple[list[float], str]] = []
    character_rows: list[tuple[list[float], str]] = []
    boxes: list[dict[str, object]] = []
    for annotation_path, annotation in rows:
        clock = annotation["objects"]["turn_clock"]
        token = str(int(clock["remaining_ms"]) // 1000)
        shape, characters = _numeric_training_rows(
            image_path=_image_path(annotation_path, annotation),
            container=clock["box"],
            token=token,
            region_feature=CLOCK_FEATURE,
            character_polarity="bright",
            region_width_px=CLOCK_REGION_WIDTH_PX,
            prefix_px=CLOCK_PREFIX_PX,
        )
        shape_rows.append(shape)
        character_rows.extend(characters)
        boxes.append(clock["box"])
    material: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "model_role": "synthetic_turn_clock_ocr_not_for_promotion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "training": {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "dataset_fingerprints": manifest["fingerprints"],
            "capture_sessions": sorted(train_sessions),
            "frames": len(rows),
            "character_examples": len(character_rows),
        },
        "geometry": {"turn_clock": _mean_box(boxes)},
        "features": {
            "clock": CLOCK_FEATURE,
            "region_width_px": CLOCK_REGION_WIDTH_PX,
            "prefix_px": CLOCK_PREFIX_PX,
            "character_polarity": "bright",
        },
        "shape_model": fit_exemplars(shape_rows),
        "character_model": fit_exemplars(character_rows),
    }
    material["checkpoint_sha256"] = canonical_sha256(material)
    output = Path(checkpoint_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(material, indent=2) + "\n", encoding="utf-8")
    return material


def load_turn_clock_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("model_kind") != MODEL_KIND:
        raise ValueError("unsupported APC turn-clock checkpoint")
    material = dict(payload)
    expected = material.pop("checkpoint_sha256", None)
    if expected != canonical_sha256(material):
        raise ValueError("turn-clock checkpoint fingerprint does not match its contents")
    return payload


def predict_turn_clock(
    checkpoint: dict[str, Any], image_path: str | Path
) -> dict[str, object]:
    path = Path(image_path).expanduser().resolve()
    seconds, confidence = _predict_numeric_token(
        image_path=path,
        container=checkpoint["geometry"]["turn_clock"],
        region_feature=checkpoint["features"]["clock"],
        character_polarity=checkpoint["features"]["character_polarity"],
        region_width_px=float(checkpoint["features"]["region_width_px"]),
        prefix_px=float(checkpoint["features"]["prefix_px"]),
        shape_model=checkpoint["shape_model"],
        character_model=checkpoint["character_model"],
    )
    return {
        "remaining_seconds": int(seconds),
        "remaining_ms": int(seconds) * 1000,
        "confidence": confidence,
        "clock_box": dict(checkpoint["geometry"]["turn_clock"]),
    }


def evaluate_turn_clock_baseline(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out split must be validation or test")
    checkpoint = load_turn_clock_checkpoint(checkpoint_path)
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = {str(value) for value in manifest["splits"][split]}
    training_sessions = {str(value) for value in checkpoint["training"]["capture_sessions"]}
    overlap = eval_sessions & training_sessions
    if overlap:
        raise ValueError(f"held-out evaluation leaks training sessions: {sorted(overlap)}")
    rows = [
        (path, item)
        for path, item in annotations
        if str(item["capture_session_id"]) in eval_sessions
        and isinstance(item["objects"].get("turn_clock"), dict)
    ]
    if not rows:
        raise ValueError("held-out split has no visible turn-clock examples")
    exact = 0
    absolute_errors: list[int] = []
    latencies: list[float] = []
    errors: list[dict[str, object]] = []
    digest_rows: list[dict[str, object]] = []
    for annotation_path, annotation in rows:
        started = time.perf_counter()
        prediction = predict_turn_clock(
            checkpoint, _image_path(annotation_path, annotation)
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        expected = int(annotation["objects"]["turn_clock"]["remaining_ms"])
        predicted = int(prediction["remaining_ms"])
        exact += int(expected == predicted)
        absolute_errors.append(abs(expected - predicted))
        if expected != predicted:
            errors.append(
                {
                    "sample_id": annotation["sample_id"],
                    "expected_ms": expected,
                    "predicted_ms": predicted,
                }
            )
        digest_rows.append(
            {"sample_id": annotation["sample_id"], "remaining_ms": predicted}
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_kind": "held_out_synthetic_turn_clock_ocr",
        "promotion_eligible": False,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": [],
        "frames": len(rows),
        "metrics": {
            "exact_remaining_ms_accuracy": exact / len(rows),
            "mean_absolute_error_ms": sum(absolute_errors) / len(rows),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "errors": errors[:20],
        "prediction_sha256": canonical_sha256(digest_rows),
        "limitations": [
            "Synthetic fixed-font whole-second countdown only; not real-table OCR evidence.",
            "Subsecond transitions, occlusion, animation and arbitrary timer formats are unsupported.",
            "Confidence is uncalibrated and the checkpoint is not promotion eligible.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate APC's turn-clock baseline.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--seed", type=int, default=2026081501)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            result = train_turn_clock_baseline(
                args.manifest, args.checkpoint, seed=args.seed
            )
            result = {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "training_frames": result["training"]["frames"],
                "character_classes": result["character_model"]["class_counts"],
            }
        else:
            result = evaluate_turn_clock_baseline(
                args.checkpoint, args.manifest, split=args.split
            )
            if args.output:
                output = args.output.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
