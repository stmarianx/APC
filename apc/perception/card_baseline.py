from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as _np
except ImportError:  # Deferred error keeps non-vision metadata tooling importable.
    _np = None

from apc.perception.baseline import (
    BaselineCheckpoint,
    _annotation_path,
    _image_path,
    _load_json,
    _manifest_annotations,
    _percentile,
    extract_feature,
    fit_centroids,
    predict_centroids,
    predict_image,
)
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


SCHEMA_VERSION = "1.0.0"
MODEL_KIND = "learned_geometry_card_pixel_baseline"
BOARD_COUNTS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5, "showdown": 5}
RANK_FEATURE = {"size": [24, 20], "representation": "local_contrast"}
SUIT_FEATURE = {"size": [12, 20], "representation": "rgb"}
RANK_RELATIVE_CROP = (0.28, 0.30, 0.72, 0.70)
SUIT_RELATIVE_CROP = (0.49, 0.38, 0.66, 0.64)


def _sub_box(box: dict[str, object], relative: tuple[float, float, float, float]) -> dict[str, float]:
    x, y = float(box["x"]), float(box["y"])
    width, height = float(box["width"]), float(box["height"])
    left, top, right, bottom = relative
    return {
        "x": x + left * width,
        "y": y + top * height,
        "width": (right - left) * width,
        "height": (bottom - top) * height,
    }


def _feature_config(
    box: dict[str, object],
    *,
    rank: bool,
    relative: tuple[float, float, float, float] | None = None,
) -> dict[str, object]:
    # Synthetic labels render a two-character rank+suit token centered on each
    # card. Separate overlapping halves so each classifier sees its glyph plus
    # a small amount of context rather than relying on annotation text.
    relative = relative or (RANK_RELATIVE_CROP if rank else SUIT_RELATIVE_CROP)
    sub = _sub_box(box, relative)
    config = dict(RANK_FEATURE if rank else SUIT_FEATURE)
    config["crop"] = [sub["x"], sub["y"], sub["x"] + sub["width"], sub["y"] + sub["height"]]
    return config


def _mean_box(boxes: list[dict[str, object]]) -> dict[str, float]:
    if not boxes:
        raise ValueError("cannot learn geometry from zero boxes")
    return {
        key: sum(float(box[key]) for box in boxes) / len(boxes)
        for key in ("x", "y", "width", "height")
    }


def fit_exemplars(rows: list[tuple[list[float], str]]) -> dict[str, object]:
    if not rows:
        raise ValueError("training rows must not be empty")
    dimensions = {len(feature) for feature, _ in rows}
    if len(dimensions) != 1:
        raise ValueError("all feature vectors must have the same dimension")
    unique: dict[tuple[str, tuple[float, ...]], list[float]] = {}
    counts: Counter[str] = Counter()
    for feature, label in rows:
        counts[label] += 1
        unique[(label, tuple(round(value, 6) for value in feature))] = feature
    return {
        "feature_dimension": dimensions.pop(),
        "exemplars": [
            {"label": label, "feature": feature}
            for (label, _), feature in sorted(unique.items(), key=lambda item: (item[0][0], item[0][1]))
        ],
        "class_counts": dict(sorted(counts.items())),
    }


def predict_exemplars(
    feature: list[float],
    model: dict[str, object],
    *,
    allowed_labels: set[str] | None = None,
) -> tuple[str, float]:
    dimension = int(model["feature_dimension"])
    if len(feature) != dimension:
        raise ValueError(f"feature dimension {len(feature)} does not match checkpoint {dimension}")
    exemplars = [
        row
        for row in model["exemplars"]
        if allowed_labels is None or str(row["label"]) in allowed_labels
    ]
    if not exemplars:
        raise ValueError("exemplar model has no rows for the allowed labels")
    distances = sorted(
        (
            math.sqrt(
                sum((value - float(candidate)) ** 2 for value, candidate in zip(feature, row["feature"]))
                / max(1, dimension)
            ),
            str(row["label"]),
        )
        for row in exemplars
    )
    best_distance, best_label = distances[0]
    alternative = next((distance for distance, label in distances if label != best_label), best_distance)
    confidence = 1.0 if alternative == 0 and best_distance == 0 else max(
        0.0, min(1.0, (alternative - best_distance) / max(alternative, 1e-12))
    )
    return best_label, confidence


def predict_exemplars_batch(
    features: list[list[float]],
    model: dict[str, object],
    *,
    allowed_labels: set[str] | None = None,
) -> list[tuple[str, float]]:
    if not features:
        return []
    if _np is None:
        raise RuntimeError("Batched APC exemplar inference requires NumPy")
    np = _np
    dimension = int(model["feature_dimension"])
    if any(len(feature) != dimension for feature in features):
        raise ValueError("feature dimension does not match checkpoint")
    exemplars = [
        row
        for row in model["exemplars"]
        if allowed_labels is None or str(row["label"]) in allowed_labels
    ]
    if not exemplars:
        raise ValueError("exemplar model has no rows for the allowed labels")
    feature_matrix = np.asarray(features, dtype="float64")
    exemplar_matrix = np.asarray([row["feature"] for row in exemplars], dtype="float64")
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
        confidence = 1.0 if alternative == 0 and best_distance == 0 else max(
            0.0,
            min(1.0, (alternative - best_distance) / max(alternative, 1e-12)),
        )
        predictions.append((best_label, confidence))
    return predictions


def _learn_geometry(training_rows: list[tuple[Path, dict[str, Any]]]) -> dict[str, object]:
    collected: dict[str, dict[str, dict[int, list[dict[str, object]]]]] = {}
    for _, annotation in training_rows:
        layout = str(annotation["environment"]["layout_id"])
        layout_rows = collected.setdefault(layout, {"hero_cards": {}, "board_cards": {}})
        for collection in ("hero_cards", "board_cards"):
            for index, card in enumerate(annotation["objects"][collection]):
                layout_rows[collection].setdefault(index, []).append(card["box"])
    geometry: dict[str, object] = {}
    for layout, collections in sorted(collected.items()):
        geometry[layout] = {
            collection: [
                _mean_box(indexes[index])
                for index in range(max(indexes, default=-1) + 1)
            ]
            for collection, indexes in collections.items()
        }
    return geometry


def _card_rows(
    training_rows: list[tuple[Path, dict[str, Any]]],
    *,
    rank: bool,
) -> list[tuple[list[float], str]]:
    rows: list[tuple[list[float], str]] = []
    target = "rank" if rank else "suit"
    for annotation_path, annotation in training_rows:
        image_path = _image_path(annotation_path, annotation)
        for collection in ("hero_cards", "board_cards"):
            for card in annotation["objects"][collection]:
                rows.append((extract_feature(image_path, _feature_config(card["box"], rank=rank)), str(card[target])))
    return rows


def train_card_baseline(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    *,
    seed: int = 20260802,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    train_sessions = set(str(value) for value in manifest["splits"]["train"])
    rows = [(path, item) for path, item in annotations if str(item["capture_session_id"]) in train_sessions]
    material: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "model_role": "synthetic_card_pipeline_baseline_not_for_promotion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "training": {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "dataset_fingerprints": manifest["fingerprints"],
            "capture_sessions": sorted(train_sessions),
            "frames": len(rows),
        },
        "geometry": _learn_geometry(rows),
        "rank_model": fit_exemplars(_card_rows(rows, rank=True)),
        "suit_model": fit_centroids(_card_rows(rows, rank=False)),
        "rank_feature": RANK_FEATURE,
        "suit_feature": SUIT_FEATURE,
        "rank_relative_crop": list(RANK_RELATIVE_CROP),
        "suit_relative_crop": list(SUIT_RELATIVE_CROP),
    }
    material["checkpoint_sha256"] = canonical_sha256(material)
    output = Path(checkpoint_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(material, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return material


def load_card_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(path).expanduser().resolve())
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("model_kind") != MODEL_KIND:
        raise ValueError("unsupported APC card checkpoint")
    material = dict(payload)
    expected = material.pop("checkpoint_sha256", None)
    if expected != canonical_sha256(material):
        raise ValueError("card checkpoint fingerprint does not match its contents")
    return payload


def predict_cards(
    card_checkpoint: dict[str, Any],
    base_checkpoint: BaselineCheckpoint,
    image_path: str | Path,
    *,
    base_prediction: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    path = Path(image_path).expanduser().resolve()
    base = base_prediction or predict_image(base_checkpoint, path)
    layout = str(base["layout_id"]["value"])
    street = str(base["street"]["value"])
    if layout not in card_checkpoint["geometry"]:
        raise ValueError(f"card geometry has no learned layout: {layout}")
    geometry = card_checkpoint["geometry"][layout]
    rank_relative = tuple(float(value) for value in card_checkpoint["rank_relative_crop"])
    suit_relative = tuple(float(value) for value in card_checkpoint["suit_relative_crop"])
    result: dict[str, object] = {
        "layout_id": base["layout_id"],
        "street": base["street"],
        "hero_cards": [],
        "board_cards": [],
    }
    slots: list[tuple[str, dict[str, object]]] = []
    for collection, count in (("hero_cards", 2), ("board_cards", BOARD_COUNTS.get(street, 0))):
        boxes = geometry[collection]
        if len(boxes) < count:
            raise ValueError(f"learned geometry has only {len(boxes)} {collection} slots, needs {count}")
        slots.extend((collection, box) for box in boxes[:count])
    rank_predictions = predict_exemplars_batch(
        [
            extract_feature(path, _feature_config(box, rank=True, relative=rank_relative))
            for _, box in slots
        ],
        card_checkpoint["rank_model"],
    )
    for index, (collection, box) in enumerate(slots):
        rank, rank_confidence = rank_predictions[index]
        suit, suit_confidence = predict_centroids(
            extract_feature(path, _feature_config(box, rank=False, relative=suit_relative)),
            card_checkpoint["suit_model"],
        )
        result[collection].append(
            {
                "rank": rank,
                "suit": suit,
                "confidence": min(rank_confidence, suit_confidence),
                "box": box,
            }
        )
    return result


def evaluate_card_baseline(
    card_checkpoint_path: str | Path,
    base_checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation split must be validation or test")
    card_checkpoint = load_card_checkpoint(card_checkpoint_path)
    base_checkpoint = BaselineCheckpoint.load(base_checkpoint_path)
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = set(str(value) for value in manifest["splits"][split])
    training_sessions = set(str(value) for value in card_checkpoint["training"]["capture_sessions"])
    overlap = eval_sessions & training_sessions & base_checkpoint.training_sessions
    if overlap:
        raise ValueError(f"held-out evaluation leaks training sessions: {sorted(overlap)}")
    rows = [(path, item) for path, item in annotations if str(item["capture_session_id"]) in eval_sessions]
    rank_correct = suit_correct = exact_correct = card_count = state_exact = 0
    rank_confusion: Counter[tuple[str, str]] = Counter()
    suit_confusion: Counter[tuple[str, str]] = Counter()
    error_examples: list[dict[str, str]] = []
    latencies: list[float] = []
    digest_rows: list[dict[str, object]] = []
    for annotation_path, annotation in rows:
        started = time.perf_counter()
        prediction = predict_cards(card_checkpoint, base_checkpoint, _image_path(annotation_path, annotation))
        latencies.append((time.perf_counter() - started) * 1000.0)
        expected_cards = [
            card
            for collection in ("hero_cards", "board_cards")
            for card in annotation["objects"][collection]
        ]
        predicted_cards = [
            card
            for collection in ("hero_cards", "board_cards")
            for card in prediction[collection]
        ]
        sample_exact = len(expected_cards) == len(predicted_cards)
        for expected, predicted in zip(expected_cards, predicted_cards):
            rank_match = expected["rank"] == predicted["rank"]
            suit_match = expected["suit"] == predicted["suit"]
            rank_confusion[(str(expected["rank"]), str(predicted["rank"]))] += 1
            suit_confusion[(str(expected["suit"]), str(predicted["suit"]))] += 1
            if not (rank_match and suit_match) and len(error_examples) < 20:
                error_examples.append(
                    {
                        "expected": f"{expected['rank']}{expected['suit']}",
                        "predicted": f"{predicted['rank']}{predicted['suit']}",
                        "sample_id": str(annotation["sample_id"]),
                    }
                )
            rank_correct += int(rank_match)
            suit_correct += int(suit_match)
            exact_correct += int(rank_match and suit_match)
            card_count += 1
            sample_exact = sample_exact and rank_match and suit_match
        state_exact += int(sample_exact)
        digest_rows.append(
            {
                "sample_id": annotation["sample_id"],
                "hero_cards": [f"{card['rank']}{card['suit']}" for card in prediction["hero_cards"]],
                "board_cards": [f"{card['rank']}{card['suit']}" for card in prediction["board_cards"]],
            }
        )
    if not rows or not card_count:
        raise ValueError("held-out split has no visible card examples")
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "held_out_card_perception_smoke",
        "promotion_eligible": False,
        "card_checkpoint_sha256": card_checkpoint["checkpoint_sha256"],
        "base_checkpoint_sha256": base_checkpoint.payload["checkpoint_sha256"],
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": [],
        "frames": len(rows),
        "visible_cards": card_count,
        "metrics": {
            "rank_accuracy": rank_correct / card_count,
            "suit_accuracy": suit_correct / card_count,
            "rank_suit_exact_accuracy": exact_correct / card_count,
            "complete_visible_card_state_accuracy": state_exact / len(rows),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "error_slices": {
            "rank_confusion": {
                f"{expected}->{predicted}": count
                for (expected, predicted), count in sorted(rank_confusion.items())
                if expected != predicted
            },
            "suit_confusion": {
                f"{expected}->{predicted}": count
                for (expected, predicted), count in sorted(suit_confusion.items())
                if expected != predicted
            },
            "examples": error_examples,
        },
        "prediction_sha256": canonical_sha256(digest_rows),
        "limitations": [
            "Learned geometry and card glyphs cover only the synthetic renderer.",
            "The head validates pixel crop learning and end-to-end inference plumbing, not real-table generalization.",
            "Confidence margins are uncalibrated and promotion is prohibited.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate APC's learned-geometry card baseline.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--seed", type=int, default=20260802)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("card_checkpoint", type=Path)
    evaluate.add_argument("base_checkpoint", type=Path)
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            result = train_card_baseline(args.manifest, args.checkpoint, seed=args.seed)
            summary = {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "training_frames": result["training"]["frames"],
                "rank_classes": result["rank_model"]["class_counts"],
                "suit_classes": result["suit_model"]["class_counts"],
            }
        else:
            result = evaluate_card_baseline(
                args.card_checkpoint, args.base_checkpoint, args.manifest, split=args.split
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
