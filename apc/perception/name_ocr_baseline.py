from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apc.perception.baseline import (
    BaselineCheckpoint,
    _image_path,
    _manifest_annotations,
    _percentile,
    predict_image,
)
from apc.perception.card_baseline import fit_exemplars
from apc.perception.table_state_baseline import (
    TIGHT_GLYPH_CANVAS,
    TIGHT_GLYPH_SIZE,
    _mean_box,
)
from apc.player_identity import NameObservation, PlayerIdentityRegistry
from apc.synthetic.render_table import NAME_OCR_CHARSET, NAME_OCR_LENGTH
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


SCHEMA_VERSION = "1.0.0"
MODEL_KIND = "synthetic_fixed_advance_player_name_character_baseline"
SEAT_WIDTH_PX = 160.0
SEAT_HEIGHT_PX = 68.0
CHARACTER_ADVANCE_PX = 6.0
CHARACTER_CROP_WIDTH_PX = 7.0
NAME_ROW_TOP_PX = 4.0
NAME_ROW_HEIGHT_PX = 26.0


def _pil_numpy() -> tuple[Any, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("APC player-name OCR requires Pillow and NumPy") from error
    return Image, np


def _tight_glyph_feature_from_image(
    image: Any,
    box: dict[str, object],
    *,
    polarity: str,
) -> list[float]:
    Image, _ = _pil_numpy()
    if polarity not in {"bright", "dark"}:
        raise ValueError("glyph polarity must be bright or dark")
    width, height = image.size
    left = max(0, min(width - 1, round(float(box["x"]) * width)))
    top = max(0, min(height - 1, round(float(box["y"]) * height)))
    right = max(
        left + 1,
        min(width, round((float(box["x"]) + float(box["width"])) * width)),
    )
    bottom = max(
        top + 1,
        min(height, round((float(box["y"]) + float(box["height"])) * height)),
    )
    region = image.crop((left, top, right, bottom))
    if polarity == "bright":
        mask = region.point(lambda value: 255 if value >= 150 else 0)
    else:
        mask = region.point(lambda value: 255 if value <= 100 else 0)
    ink = mask.getbbox()
    canvas = Image.new("L", TIGHT_GLYPH_CANVAS)
    if ink is not None:
        glyph = mask.crop(ink)
        x = max(0, (TIGHT_GLYPH_CANVAS[0] - glyph.width) // 2)
        y = max(0, (TIGHT_GLYPH_CANVAS[1] - glyph.height) // 2)
        canvas.paste(glyph, (x, y))
    resampling = getattr(Image, "Resampling", Image).BOX
    reduced = canvas.resize(TIGHT_GLYPH_SIZE, resampling)
    flattened = (
        reduced.get_flattened_data()
        if hasattr(reduced, "get_flattened_data")
        else reduced.getdata()
    )
    return [value / 255.0 for value in flattened]


def _predict_exemplars_batch(
    features: list[list[float]],
    model: dict[str, object],
    *,
    allowed_labels: set[str],
) -> list[tuple[str, float]]:
    _, np = _pil_numpy()
    dimension = int(model["feature_dimension"])
    if any(len(feature) != dimension for feature in features):
        raise ValueError("player-name feature dimension does not match checkpoint")
    exemplars = [
        row for row in model["exemplars"] if str(row["label"]) in allowed_labels
    ]
    if not exemplars:
        raise ValueError("player-name checkpoint has no allowed character exemplars")
    feature_matrix = np.asarray(features, dtype="float32")
    exemplar_matrix = np.asarray([row["feature"] for row in exemplars], dtype="float32")
    squared = (
        np.sum(feature_matrix * feature_matrix, axis=1, keepdims=True)
        + np.sum(exemplar_matrix * exemplar_matrix, axis=1)[None, :]
        - 2.0 * feature_matrix @ exemplar_matrix.T
    ) / max(1, dimension)
    distances = np.sqrt(np.maximum(squared, 0.0))
    labels = [str(row["label"]) for row in exemplars]
    predictions: list[tuple[str, float]] = []
    for row in distances:
        order = np.argsort(row)
        best_index = int(order[0])
        best_label = labels[best_index]
        best_distance = float(row[best_index])
        alternative = next(
            (float(row[int(index)]) for index in order if labels[int(index)] != best_label),
            best_distance,
        )
        confidence = (
            1.0
            if alternative == 0 and best_distance == 0
            else max(0.0, min(1.0, (alternative - best_distance) / max(alternative, 1e-12)))
        )
        predictions.append((best_label, confidence))
    return predictions


def _validate_synthetic_name(value: object) -> str:
    name = str(value)
    if len(name) != NAME_OCR_LENGTH or any(
        character not in NAME_OCR_CHARSET for character in name
    ):
        raise ValueError(
            f"synthetic OCR name must contain exactly {NAME_OCR_LENGTH} supported characters"
        )
    return name


def name_character_box(
    seat_box: dict[str, object],
    character_index: int,
) -> dict[str, float]:
    if not 0 <= character_index < NAME_OCR_LENGTH:
        raise ValueError("player-name character index is out of range")
    x, y, width, height = (
        float(seat_box[key]) for key in ("x", "y", "width", "height")
    )
    rendered_width = CHARACTER_ADVANCE_PX * NAME_OCR_LENGTH
    left_px = (SEAT_WIDTH_PX - rendered_width) / 2 + CHARACTER_ADVANCE_PX * character_index
    return {
        "x": x + width * left_px / SEAT_WIDTH_PX,
        "y": y + height * NAME_ROW_TOP_PX / SEAT_HEIGHT_PX,
        "width": width * CHARACTER_CROP_WIDTH_PX / SEAT_WIDTH_PX,
        "height": height * NAME_ROW_HEIGHT_PX / SEAT_HEIGHT_PX,
    }


def _learn_geometry(
    rows: list[tuple[Path, dict[str, Any]]],
) -> dict[str, list[dict[str, float]]]:
    layouts: dict[str, dict[int, list[dict[str, object]]]] = {}
    for _, annotation in rows:
        layout = str(annotation["environment"]["layout_id"])
        seats = layouts.setdefault(layout, {})
        for index, seat in enumerate(annotation["objects"]["seats"]):
            seats.setdefault(index, []).append(seat["box"])
    return {
        layout: [_mean_box(indexes[index]) for index in range(max(indexes) + 1)]
        for layout, indexes in sorted(layouts.items())
    }


def train_name_ocr_baseline(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    *,
    seed: int = 2026081503,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    train_sessions = {str(value) for value in manifest["splits"]["train"]}
    rows = [
        (path, annotation)
        for path, annotation in annotations
        if str(annotation["capture_session_id"]) in train_sessions
    ]
    character_rows: list[tuple[list[float], str]] = []
    training_names: set[str] = set()
    for annotation_path, annotation in rows:
        image_path = _image_path(annotation_path, annotation)
        Image, _ = _pil_numpy()
        with Image.open(image_path) as opened:
            image = opened.convert("L")
            for seat in annotation["objects"]["seats"]:
                name = _validate_synthetic_name(seat.get("player_name"))
                training_names.add(name)
                for index, character in enumerate(name):
                    character_rows.append(
                        (
                            _tight_glyph_feature_from_image(
                                image,
                                name_character_box(seat["box"], index),
                                polarity="bright",
                            ),
                            character,
                        )
                    )
    model = fit_exemplars(character_rows)
    missing_classes = set(NAME_OCR_CHARSET) - set(model["class_counts"])
    if missing_classes:
        raise ValueError(f"training split misses name characters: {sorted(missing_classes)}")
    material: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "model_role": "synthetic_character_name_ocr_not_for_promotion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "training": {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "dataset_fingerprints": manifest["fingerprints"],
            "capture_sessions": sorted(train_sessions),
            "frames": len(rows),
            "unique_names": len(training_names),
            "character_examples": len(character_rows),
            "training_names_sha256": canonical_sha256(sorted(training_names)),
        },
        "geometry": _learn_geometry(rows),
        "decoder": {
            "charset": NAME_OCR_CHARSET,
            "name_length": NAME_OCR_LENGTH,
            "seat_width_px": SEAT_WIDTH_PX,
            "seat_height_px": SEAT_HEIGHT_PX,
            "character_advance_px": CHARACTER_ADVANCE_PX,
            "character_crop_width_px": CHARACTER_CROP_WIDTH_PX,
            "name_row_top_px": NAME_ROW_TOP_PX,
            "name_row_height_px": NAME_ROW_HEIGHT_PX,
            "polarity": "bright",
        },
        "character_model": model,
    }
    material["checkpoint_sha256"] = canonical_sha256(material)
    output = Path(checkpoint_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(material, indent=2) + "\n", encoding="utf-8")
    return material


def load_name_ocr_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("model_kind") != MODEL_KIND:
        raise ValueError("unsupported APC player-name OCR checkpoint")
    material = dict(payload)
    expected = material.pop("checkpoint_sha256", None)
    if expected != canonical_sha256(material):
        raise ValueError("player-name OCR checkpoint fingerprint does not match its contents")
    return payload


def _predict_layout_names(
    checkpoint: dict[str, Any],
    image_path: Path,
    layout: str,
) -> list[dict[str, object]]:
    if layout not in checkpoint["geometry"]:
        raise ValueError(f"player-name geometry has no learned layout: {layout}")
    frame_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    rows: list[dict[str, object]] = []
    allowed = set(str(checkpoint["decoder"]["charset"]))
    Image, _ = _pil_numpy()
    features: list[list[float]] = []
    with Image.open(image_path) as opened:
        image = opened.convert("L")
        for seat_index, seat_box in enumerate(checkpoint["geometry"][layout]):
            for character_index in range(int(checkpoint["decoder"]["name_length"])):
                features.append(
                    _tight_glyph_feature_from_image(
                        image,
                        name_character_box(seat_box, character_index),
                        polarity=str(checkpoint["decoder"]["polarity"]),
                    )
                )
    predictions = _predict_exemplars_batch(
        features,
        checkpoint["character_model"],
        allowed_labels=allowed,
    )
    name_length = int(checkpoint["decoder"]["name_length"])
    for seat_index, seat_box in enumerate(checkpoint["geometry"][layout]):
        start = seat_index * name_length
        seat_predictions = predictions[start : start + name_length]
        character_confidences = [confidence for _, confidence in seat_predictions]
        rows.append(
            {
                "seat_no": seat_index + 1,
                "player_name": "".join(character for character, _ in seat_predictions),
                "confidence": sum(character_confidences) / len(character_confidences),
                "minimum_character_confidence": min(character_confidences),
                "seat_box": dict(seat_box),
                "frame_sha256": frame_sha256,
            }
        )
    return rows


def predict_player_names(
    checkpoint: dict[str, Any],
    image_path: str | Path,
    *,
    base_checkpoint: BaselineCheckpoint | None = None,
    base_prediction: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    path = Path(image_path).expanduser().resolve()
    if base_prediction is None:
        if base_checkpoint is None:
            raise ValueError("player-name prediction requires base layout evidence")
        base_prediction = predict_image(base_checkpoint, path)
    layout = str(base_prediction["layout_id"]["value"])
    return {
        "layout_id": base_prediction["layout_id"],
        "player_names": _predict_layout_names(checkpoint, path, layout),
    }


def evaluate_name_ocr_baseline(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out split must be validation or test")
    checkpoint = load_name_ocr_checkpoint(checkpoint_path)
    manifest_file = Path(manifest_path).expanduser().resolve()
    validation = validate_manifest(manifest_file)
    if not validation["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(validation["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = {str(value) for value in manifest["splits"][split]}
    training_sessions = {str(value) for value in checkpoint["training"]["capture_sessions"]}
    overlap = eval_sessions & training_sessions
    if overlap:
        raise ValueError(f"held-out evaluation leaks training sessions: {sorted(overlap)}")
    training_names = {
        _validate_synthetic_name(seat.get("player_name"))
        for _, annotation in annotations
        if str(annotation["capture_session_id"]) in training_sessions
        for seat in annotation["objects"]["seats"]
    }
    frame_count = 0
    name_count = 0
    exact_names = 0
    exact_characters = 0
    total_characters = 0
    unseen_names = 0
    latencies: list[float] = []
    errors: list[dict[str, object]] = []
    digest_rows: list[dict[str, object]] = []
    prediction_confidences: list[float] = []
    registry = PlayerIdentityRegistry("synthetic-name-ocr-audit")
    final_resolutions: dict[tuple[str, int], dict[str, object]] = {}
    for annotation_path, annotation in annotations:
        if str(annotation["capture_session_id"]) not in eval_sessions:
            continue
        image_path = _image_path(annotation_path, annotation)
        layout = str(annotation["environment"]["layout_id"])
        started = time.perf_counter()
        predictions = _predict_layout_names(checkpoint, image_path, layout)
        latencies.append((time.perf_counter() - started) * 1000.0)
        frame_count += 1
        session_id = str(annotation["capture_session_id"])
        resolutions = registry.observe_batch(
            session_id,
            [
                NameObservation(
                    seat_no=int(prediction["seat_no"]),
                    raw_name=str(prediction["player_name"]),
                    confidence=float(prediction["confidence"]),
                    frame_sha256=str(prediction["frame_sha256"]),
                    observed_at_ms=int(annotation["image"]["timestamp_ms"]),
                )
                for prediction in predictions
            ],
        )
        for resolution in resolutions:
            final_resolutions[(session_id, int(resolution["seat_no"]))] = resolution
        expected = {
            int(seat["seat_no"]): _validate_synthetic_name(seat.get("player_name"))
            for seat in annotation["objects"]["seats"]
        }
        for prediction in predictions:
            seat_no = int(prediction["seat_no"])
            truth = expected[seat_no]
            observed = str(prediction["player_name"])
            prediction_confidences.append(float(prediction["confidence"]))
            name_count += 1
            exact_names += int(truth == observed)
            unseen_names += int(truth not in training_names)
            exact_characters += sum(left == right for left, right in zip(truth, observed))
            total_characters += len(truth)
            if truth != observed and len(errors) < 20:
                errors.append(
                    {
                        "sample_id": annotation["sample_id"],
                        "seat_no": seat_no,
                        "expected": truth,
                        "predicted": observed,
                    }
                )
            digest_rows.append(
                {
                    "sample_id": annotation["sample_id"],
                    "seat_no": seat_no,
                    "player_name": observed,
                }
            )
    if not frame_count or not name_count:
        raise ValueError("held-out split has no player-name examples")
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_kind": "held_out_synthetic_character_player_name_ocr",
        "promotion_eligible": False,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": [],
        "frames": frame_count,
        "names": name_count,
        "metrics": {
            "exact_name_accuracy": exact_names / name_count,
            "character_accuracy": exact_characters / total_characters,
            "unseen_whole_name_rate": unseen_names / name_count,
            "minimum_name_confidence": min(prediction_confidences),
            "final_identity_resolution_rate": sum(
                resolution.get("status") == "resolved"
                for resolution in final_resolutions.values()
            )
            / len(final_resolutions),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "errors": errors,
        "registry_snapshot_sha256": registry.snapshot()["snapshot_sha256"],
        "prediction_sha256": canonical_sha256(digest_rows),
        "limitations": [
            "Synthetic fixed-font, fixed-length, restricted-character player names only.",
            "The character head is not evidence for arbitrary fonts, Unicode, variable-length names or occlusion.",
            "Confidence is uncalibrated and the checkpoint is not promotion eligible.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate APC's player-name OCR baseline.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--seed", type=int, default=2026081503)
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
            result = train_name_ocr_baseline(args.manifest, args.checkpoint, seed=args.seed)
            result = {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "training_frames": result["training"]["frames"],
                "character_classes": result["character_model"]["class_counts"],
            }
        else:
            result = evaluate_name_ocr_baseline(
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
