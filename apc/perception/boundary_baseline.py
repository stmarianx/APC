from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apc.perception.baseline import (
    _image_path,
    _manifest_annotations,
    _percentile,
    fit_centroids,
    predict_centroids,
)
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


SCHEMA_VERSION = "1.0.0"
MODEL_KIND = "temporal_hand_boundary_pixel_baseline"
DIFF_SIZE = (32, 18)
CURRENT_SIZE = (24, 14)


def _boundary_feature(previous_image: Path, current_image: Path) -> list[float]:
    try:
        from PIL import Image, ImageChops
    except ImportError as error:
        raise RuntimeError("APC boundary perception requires Pillow") from error
    with Image.open(previous_image) as previous_source, Image.open(current_image) as current_source:
        previous = previous_source.convert("RGB")
        current = current_source.convert("RGB")
        if previous.size != current.size:
            raise ValueError("boundary frame pair must have identical dimensions")
        difference = ImageChops.difference(previous, current).convert("L")
        resampling = getattr(Image, "Resampling", Image).BOX
        reduced = difference.resize(DIFF_SIZE, resampling)
        flattened = reduced.get_flattened_data() if hasattr(reduced, "get_flattened_data") else reduced.getdata()
        diff = [value / 255.0 for value in flattened]
        current_reduced = current.convert("L").resize(CURRENT_SIZE, resampling)
        current_flattened = current_reduced.get_flattened_data() if hasattr(current_reduced, "get_flattened_data") else current_reduced.getdata()
        current_feature = [value / 255.0 for value in current_flattened]
    return diff + current_feature


def _transitions(
    annotations: list[tuple[Path, dict[str, Any]]],
) -> list[tuple[Path, dict[str, Any], Path, dict[str, Any], bool]]:
    sessions: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for row in annotations:
        sessions[str(row[1]["capture_session_id"])].append(row)
    result = []
    for session, rows in sorted(sessions.items()):
        ordered = sorted(rows, key=lambda row: int(row[1]["sequence_index"]))
        for previous, current in zip(ordered, ordered[1:]):
            label = current[1]["state"].get("hand_start")
            if not isinstance(label, bool):
                raise ValueError(f"boundary dataset {session} is missing boolean state.hand_start")
            result.append((*previous, *current, label))
    return result


def train_boundary_baseline(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    *,
    seed: int = 8675309,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    train_sessions = {str(value) for value in manifest["splits"]["train"]}
    selected = [row for row in annotations if str(row[1]["capture_session_id"]) in train_sessions]
    transitions = _transitions(selected)
    rows = []
    counts = {"boundary": 0, "continuation": 0}
    for previous_path, previous, current_path, current, boundary in transitions:
        label = "boundary" if boundary else "continuation"
        counts[label] += 1
        rows.append(
            (
                _boundary_feature(
                    _image_path(previous_path, previous),
                    _image_path(current_path, current),
                ),
                label,
            )
        )
    if not all(counts.values()):
        raise ValueError(f"boundary training requires both classes, got {counts}")
    material: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "model_role": "synthetic_hand_boundary_baseline_not_for_promotion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "training": {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "dataset_fingerprints": manifest["fingerprints"],
            "capture_sessions": sorted(train_sessions),
            "transitions": len(rows),
            "class_counts": counts,
        },
        "feature": {"diff_size": list(DIFF_SIZE), "current_size": list(CURRENT_SIZE)},
        "boundary_model": fit_centroids(rows),
    }
    material["checkpoint_sha256"] = canonical_sha256(material)
    output = Path(checkpoint_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(material, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return material


def load_boundary_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("model_kind") != MODEL_KIND:
        raise ValueError("unsupported APC hand-boundary checkpoint")
    material = dict(payload)
    expected = material.pop("checkpoint_sha256", None)
    if expected != canonical_sha256(material):
        raise ValueError("hand-boundary checkpoint fingerprint does not match its contents")
    return payload


def predict_boundary(
    checkpoint: dict[str, Any],
    previous_image: str | Path,
    current_image: str | Path,
) -> dict[str, object]:
    label, confidence = predict_centroids(
        _boundary_feature(
            Path(previous_image).expanduser().resolve(),
            Path(current_image).expanduser().resolve(),
        ),
        checkpoint["boundary_model"],
    )
    return {
        "hand_start": label == "boundary",
        "label": label,
        "confidence": confidence,
    }


def evaluate_boundary_baseline(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "validation",
) -> dict[str, object]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation split must be validation or test")
    checkpoint = load_boundary_checkpoint(checkpoint_path)
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = {str(value) for value in manifest["splits"][split]}
    train_sessions = {str(value) for value in checkpoint["training"]["capture_sessions"]}
    overlap = eval_sessions & train_sessions
    if overlap:
        raise ValueError(f"held-out boundary evaluation leaks training sessions: {sorted(overlap)}")
    selected = [row for row in annotations if str(row[1]["capture_session_id"]) in eval_sessions]
    transitions = _transitions(selected)
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    latencies: list[float] = []
    errors: list[dict[str, object]] = []
    digest_rows = []
    for previous_path, previous, current_path, current, expected in transitions:
        started = time.perf_counter()
        prediction = predict_boundary(
            checkpoint,
            _image_path(previous_path, previous),
            _image_path(current_path, current),
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        observed = bool(prediction["hand_start"])
        key = "tp" if expected and observed else "fn" if expected else "fp" if observed else "tn"
        confusion[key] += 1
        transition_id = f"{previous['sample_id']}->{current['sample_id']}"
        if observed != expected:
            errors.append(
                {
                    "transition": transition_id,
                    "expected_hand_start": expected,
                    "predicted": prediction,
                }
            )
        digest_rows.append({"transition": transition_id, "prediction": prediction})
    total = sum(confusion.values())
    positives = confusion["tp"] + confusion["fn"]
    predicted_positives = confusion["tp"] + confusion["fp"]
    negatives = confusion["tn"] + confusion["fp"]
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "held_out_hand_boundary_smoke",
        "promotion_eligible": False,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": [],
        "transitions": total,
        "positive_boundaries": positives,
        "metrics": {
            "accuracy": (confusion["tp"] + confusion["tn"]) / total,
            "precision": confusion["tp"] / predicted_positives if predicted_positives else 0.0,
            "recall": confusion["tp"] / positives if positives else 0.0,
            "specificity": confusion["tn"] / negatives if negatives else 0.0,
            "confusion": confusion,
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "errors": errors[:20],
        "prediction_sha256": canonical_sha256(digest_rows),
        "limitations": [
            "Synthetic two-hand capture sessions only.",
            "Session-initial hands use capture-start evidence and are not part of adjacent-frame boundary evaluation.",
            "Confidence is uncalibrated and this checkpoint is not promotion eligible.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate APC's temporal hand-boundary baseline.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--seed", type=int, default=8675309)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            result = train_boundary_baseline(args.manifest, args.checkpoint, seed=args.seed)
            summary: dict[str, object] = {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "training": result["training"],
            }
        else:
            summary = evaluate_boundary_baseline(args.checkpoint, args.manifest, split=args.split)
            if args.output:
                output = args.output.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
