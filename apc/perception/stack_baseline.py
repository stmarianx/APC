from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apc.perception.baseline import (
    BaselineCheckpoint,
    _image_path,
    _manifest_annotations,
    _percentile,
    extract_feature,
    fit_centroids,
    predict_centroids_batch,
    predict_image,
)
from apc.perception.card_baseline import fit_exemplars, predict_exemplars_batch
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


SCHEMA_VERSION = "1.0.0"
MODEL_KIND = "segmented_stack_numeric_token_pixel_baseline"
LEGACY_MODEL_KIND = "segmented_stack_digit_pixel_baseline"
LENGTH_FEATURE = {"size": [56, 18], "representation": "bright_mask"}
DIGIT_FEATURE = {"size": [12, 20], "representation": "local_contrast"}
STACK_LINE_RELATIVE = (0.18, 0.55, 0.82, 0.90)
DIGIT_CELL_WIDTH = 6 / 160
DECIMAL_CELL_WIDTH = 2 / 160
STACK_SUFFIX_WIDTH = 14 / 160
DIGIT_VERTICAL = (0.58, 0.88)


def _relative_box(
    box: dict[str, object], relative: tuple[float, float, float, float]
) -> dict[str, float]:
    x, y, width, height = (float(box[key]) for key in ("x", "y", "width", "height"))
    left, top, right, bottom = relative
    return {
        "x": x + left * width,
        "y": y + top * height,
        "width": (right - left) * width,
        "height": (bottom - top) * height,
    }


def _config(box: dict[str, object], feature: dict[str, object]) -> dict[str, object]:
    result = dict(feature)
    result["crop"] = [
        float(box["x"]),
        float(box["y"]),
        float(box["x"]) + float(box["width"]),
        float(box["y"]) + float(box["height"]),
    ]
    return result


STACK_TOKEN_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
MIN_TOKEN_CHARACTERS = 1
MAX_TOKEN_CHARACTERS = 5


def digit_box(
    seat_box: dict[str, object],
    *,
    digit_count: int,
    digit_index: int,
    decimal_index: int | None = None,
) -> dict[str, float]:
    if not MIN_TOKEN_CHARACTERS <= digit_count <= MAX_TOKEN_CHARACTERS:
        raise ValueError("synthetic stack OCR supports one to five numeric-token characters")
    if not 0 <= digit_index < digit_count:
        raise ValueError("digit index is outside the stack string")
    if decimal_index is not None and not 0 < decimal_index < digit_count - 1:
        raise ValueError("decimal index must be inside the numeric token")
    widths = [
        DECIMAL_CELL_WIDTH if index == decimal_index else DIGIT_CELL_WIDTH
        for index in range(digit_count)
    ]
    text_width = sum(widths) + STACK_SUFFIX_WIDTH
    start = 0.5 - text_width / 2
    character_start = start + sum(widths[:digit_index])
    return _relative_box(
        seat_box,
        (
            character_start,
            DIGIT_VERTICAL[0],
            character_start + widths[digit_index],
            DIGIT_VERTICAL[1],
        ),
    )


def _token_shape(token: str) -> str:
    if "." not in token:
        return f"integer:{len(token)}"
    left, right = token.split(".", 1)
    return f"decimal:{len(left)}:{len(right)}"


def _shape_geometry(shape: str) -> tuple[int, int | None]:
    pieces = shape.split(":")
    if len(pieces) == 2 and pieces[0] == "integer":
        count = int(pieces[1])
        return count, None
    if len(pieces) == 3 and pieces[0] == "decimal":
        left, right = int(pieces[1]), int(pieces[2])
        return left + 1 + right, left
    raise ValueError(f"unsupported stack token shape: {shape!r}")


def _mean_box(boxes: list[dict[str, object]]) -> dict[str, float]:
    if not boxes:
        raise ValueError("cannot learn stack geometry from zero boxes")
    return {
        key: sum(float(box[key]) for box in boxes) / len(boxes)
        for key in ("x", "y", "width", "height")
    }


def _learn_seat_geometry(rows: list[tuple[Path, dict[str, Any]]]) -> dict[str, object]:
    collected: dict[str, dict[int, list[dict[str, object]]]] = {}
    for _, annotation in rows:
        layout = str(annotation["environment"]["layout_id"])
        layout_rows = collected.setdefault(layout, {})
        for index, seat in enumerate(annotation["objects"]["seats"]):
            layout_rows.setdefault(index, []).append(seat["box"])
    return {
        layout: [_mean_box(indexes[index]) for index in range(max(indexes) + 1)]
        for layout, indexes in sorted(collected.items())
    }


def train_stack_baseline(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    *,
    seed: int = 118092,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    train_sessions = set(str(value) for value in manifest["splits"]["train"])
    rows = [(path, item) for path, item in annotations if str(item["capture_session_id"]) in train_sessions]
    length_rows: list[tuple[list[float], str]] = []
    character_rows: list[tuple[list[float], str]] = []
    for annotation_path, annotation in rows:
        image_path = _image_path(annotation_path, annotation)
        for seat in annotation["objects"]["seats"]:
            token = str(seat["stack_bb"])
            if not STACK_TOKEN_RE.fullmatch(token) or not MIN_TOKEN_CHARACTERS <= len(token) <= MAX_TOKEN_CHARACTERS:
                raise ValueError(f"stack baseline requires a one-to-five-character non-negative BB token, got {token!r}")
            line_box = _relative_box(seat["box"], STACK_LINE_RELATIVE)
            length_rows.append((extract_feature(image_path, _config(line_box, LENGTH_FEATURE)), _token_shape(token)))
            decimal_index = token.find(".") if "." in token else None
            for index, character in enumerate(token):
                box = digit_box(
                    seat["box"],
                    digit_count=len(token),
                    digit_index=index,
                    decimal_index=decimal_index,
                )
                character_rows.append((extract_feature(image_path, _config(box, DIGIT_FEATURE)), character))
    material: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "model_role": "synthetic_segmented_stack_ocr_not_for_promotion",
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
        "geometry": {"seats": _learn_seat_geometry(rows)},
        "features": {
            "length": LENGTH_FEATURE,
            "digit": DIGIT_FEATURE,
            "stack_line_relative": list(STACK_LINE_RELATIVE),
            "digit_cell_width": DIGIT_CELL_WIDTH,
            "decimal_cell_width": DECIMAL_CELL_WIDTH,
            "suffix_width": STACK_SUFFIX_WIDTH,
            "digit_vertical": list(DIGIT_VERTICAL),
        },
        "shape_model": fit_centroids(length_rows),
        "character_model": fit_exemplars(character_rows),
    }
    material["checkpoint_sha256"] = canonical_sha256(material)
    output = Path(checkpoint_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(material, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return material


def load_stack_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("model_kind") not in {MODEL_KIND, LEGACY_MODEL_KIND}:
        raise ValueError("unsupported APC stack checkpoint")
    material = dict(payload)
    expected = material.pop("checkpoint_sha256", None)
    if expected != canonical_sha256(material):
        raise ValueError("stack checkpoint fingerprint does not match its contents")
    return payload


def predict_stacks(
    checkpoint: dict[str, Any],
    base_checkpoint: BaselineCheckpoint,
    image_path: str | Path,
    *,
    base_prediction: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    path = Path(image_path).expanduser().resolve()
    base = base_prediction or predict_image(base_checkpoint, path)
    layout = str(base["layout_id"]["value"])
    seat_boxes = checkpoint["geometry"]["seats"].get(layout)
    if not seat_boxes:
        raise ValueError(f"stack geometry has no learned layout: {layout}")
    line_features = [
        extract_feature(
            path,
            _config(
                _relative_box(
                    seat_box,
                    tuple(checkpoint["features"]["stack_line_relative"]),
                ),
                checkpoint["features"]["length"],
            ),
        )
        for seat_box in seat_boxes
    ]
    shape_model = (
        checkpoint["length_model"]
        if checkpoint["model_kind"] == LEGACY_MODEL_KIND
        else checkpoint["shape_model"]
    )
    shape_predictions = predict_centroids_batch(line_features, shape_model)
    seat_specs: list[dict[str, object]] = []
    character_features: list[list[float]] = []
    character_slots: list[tuple[int, int]] = []
    character_model = (
        checkpoint["digit_model"]
        if checkpoint["model_kind"] == LEGACY_MODEL_KIND
        else checkpoint["character_model"]
    )
    for seat_no, (seat_box, shape_prediction) in enumerate(
        zip(seat_boxes, shape_predictions), start=1
    ):
        raw_shape, length_confidence = shape_prediction
        if checkpoint["model_kind"] == LEGACY_MODEL_KIND:
            digit_count, decimal_index = int(raw_shape), None
            shape = f"integer:{digit_count}"
        else:
            shape = raw_shape
            digit_count, decimal_index = _shape_geometry(shape)
        characters: list[str | None] = [None] * digit_count
        confidences = [float(length_confidence)]
        spec_index = len(seat_specs)
        for index in range(digit_count):
            if index == decimal_index:
                characters[index] = "."
                confidences.append(length_confidence)
                continue
            box = digit_box(
                seat_box,
                digit_count=digit_count,
                digit_index=index,
                decimal_index=decimal_index,
            )
            character_features.append(
                extract_feature(path, _config(box, checkpoint["features"]["digit"]))
            )
            character_slots.append((spec_index, index))
        seat_specs.append(
            {
                "seat_no": seat_no,
                "seat_box": seat_box,
                "shape": shape,
                "digit_count": digit_count,
                "characters": characters,
                "confidences": confidences,
            }
        )
    for (spec_index, character_index), (character, confidence) in zip(
        character_slots,
        predict_exemplars_batch(
            character_features,
            character_model,
            allowed_labels=set("0123456789"),
        ),
    ):
        seat_specs[spec_index]["characters"][character_index] = character
        seat_specs[spec_index]["confidences"].append(confidence)
    predictions = []
    for spec in seat_specs:
        if any(character is None for character in spec["characters"]):
            raise ValueError("stack OCR left an unresolved character slot")
        token = "".join(str(character) for character in spec["characters"])
        if not STACK_TOKEN_RE.fullmatch(token):
            raise ValueError(
                f"stack OCR produced an invalid BB token at seat {spec['seat_no']}: {token!r}"
            )
        predictions.append(
            {
                "seat_no": spec["seat_no"],
                "stack_bb": token,
                "confidence": min(spec["confidences"]),
                "character_count": spec["digit_count"],
                "token_shape": spec["shape"],
                "seat_box": dict(spec["seat_box"]),
            }
        )
    return predictions


def evaluate_stack_baseline(
    checkpoint_path: str | Path,
    base_checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "validation",
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation split must be validation or test")
    checkpoint = load_stack_checkpoint(checkpoint_path)
    base_checkpoint = BaselineCheckpoint.load(base_checkpoint_path)
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = set(str(value) for value in manifest["splits"][split])
    training_sessions = set(str(value) for value in checkpoint["training"]["capture_sessions"])
    overlap = eval_sessions & training_sessions & base_checkpoint.training_sessions
    if overlap:
        raise ValueError(f"held-out evaluation leaks training sessions: {sorted(overlap)}")
    rows = [(path, item) for path, item in annotations if str(item["capture_session_id"]) in eval_sessions]
    exact = length_exact = total = frame_exact = 0
    latencies: list[float] = []
    errors: list[dict[str, object]] = []
    digest_rows = []
    for annotation_path, annotation in rows:
        started = time.perf_counter()
        predictions = predict_stacks(checkpoint, base_checkpoint, _image_path(annotation_path, annotation))
        latencies.append((time.perf_counter() - started) * 1000.0)
        expected = [str(seat["stack_bb"]) for seat in annotation["objects"]["seats"]]
        predicted = [str(item["stack_bb"]) for item in predictions]
        matches = [left == right for left, right in zip(expected, predicted)]
        exact += sum(matches)
        length_exact += sum(len(left) == int(item["character_count"]) for left, item in zip(expected, predictions))
        total += len(expected)
        frame_exact += int(len(expected) == len(predicted) and all(matches))
        if not all(matches):
            errors.append({"sample_id": annotation["sample_id"], "expected": expected, "predicted": predicted})
        digest_rows.append({"sample_id": annotation["sample_id"], "stacks_bb": predicted})
    if not rows or not total:
        raise ValueError("held-out split has no stack examples")
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "held_out_segmented_stack_ocr_smoke",
        "promotion_eligible": False,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "base_checkpoint_sha256": base_checkpoint.payload["checkpoint_sha256"],
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": [],
        "frames": len(rows),
        "stack_examples": total,
        "metrics": {
            "character_count_accuracy": length_exact / total,
            "stack_exact_accuracy": exact / total,
            "complete_frame_stack_accuracy": frame_exact / len(rows),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "errors": errors[:20],
        "prediction_sha256": canonical_sha256(digest_rows),
        "limitations": [
            "Synthetic renderer v2 fixed-width numeric-token segmentation only.",
            "All-in text labels, occlusion and arbitrary fonts are not supported.",
            "Confidence is uncalibrated and this checkpoint is not promotion eligible.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate APC's segmented stack OCR baseline.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--seed", type=int, default=118092)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("base_checkpoint", type=Path)
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            result = train_stack_baseline(args.manifest, args.checkpoint, seed=args.seed)
            summary = {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "training_frames": result["training"]["frames"],
                "character_examples": result["training"]["character_examples"],
                "character_class_counts": result["character_model"]["class_counts"],
            }
        else:
            result = evaluate_stack_baseline(
                args.checkpoint, args.base_checkpoint, args.manifest, split=args.split
            )
            if args.output:
                output = args.output.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            summary = result
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
