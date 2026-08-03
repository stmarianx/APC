from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from apc.tools.validate_dataset import canonical_sha256, validate_manifest


CHECKPOINT_SCHEMA_VERSION = "1.0.0"
MODEL_KIND = "nearest_centroid_pixel_baseline"
DEFAULT_FEATURES: dict[str, dict[str, object]] = {
    "layout_id": {
        "crop": [0.05, 0.10, 0.95, 0.82],
        "size": [64, 40],
        "representation": "bright_mask",
    },
    "theme_id": {"crop": [0.08, 0.17, 0.92, 0.80], "size": [16, 10]},
    "street": {"crop": [0.34, 0.36, 0.66, 0.58], "size": [24, 10]},
    "legal_actions": {"crop": [0.58, 0.86, 1.0, 1.0], "size": [24, 8]},
}


def _pil_image() -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "APC pixel baselines require Pillow; use the bundled workspace Python "
            "or install apc/requirements-vision.txt"
        ) from error
    return Image


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _annotation_path(manifest_path: Path, raw_path: str) -> Path:
    return (manifest_path.parent / raw_path).resolve()


def _image_path(annotation_path: Path, annotation: dict[str, Any]) -> Path:
    return (annotation_path.parent / str(annotation["image"]["path"])).resolve()


def _label(annotation: dict[str, Any], head: str) -> str:
    if head == "layout_id":
        return str(annotation["environment"]["layout_id"])
    if head == "theme_id":
        return str(annotation["environment"]["theme_id"])
    if head == "street":
        return str(annotation["state"]["street"])
    if head == "legal_actions":
        return "+".join(sorted(str(value) for value in annotation["state"]["legal_actions"]))
    raise KeyError(f"unsupported baseline head: {head}")


def extract_feature(image_path: Path, config: dict[str, object]) -> list[float]:
    Image = _pil_image()
    crop = [float(value) for value in config["crop"]]  # type: ignore[index]
    size = [int(value) for value in config["size"]]  # type: ignore[index]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        box = (
            round(crop[0] * width),
            round(crop[1] * height),
            round(crop[2] * width),
            round(crop[3] * height),
        )
        region = image.crop(box)
        representation = str(config.get("representation", "rgb"))
        resampling = getattr(Image, "Resampling", Image).BOX
        if representation == "bright_mask":
            region = region.convert("L").point(lambda value: 255 if value >= 150 else 0)
        elif representation == "dark_mask":
            region = region.convert("L").point(lambda value: 255 if value <= 210 else 0)
        elif representation == "ink_mask":
            rgb_region = region.convert("RGB")
            mask = Image.new("L", rgb_region.size)
            pixels = (
                rgb_region.get_flattened_data()
                if hasattr(rgb_region, "get_flattened_data")
                else rgb_region.getdata()
            )
            mask.putdata([
                255
                if (max(pixel) - min(pixel) >= 15 or sum(pixel) / 3 <= 205)
                else 0
                for pixel in pixels
            ])
            region = mask
        elif representation == "local_contrast":
            rgb_region = region.convert("RGB")
            pixels = list(
                rgb_region.get_flattened_data()
                if hasattr(rgb_region, "get_flattened_data")
                else rgb_region.getdata()
            )
            channel_background = []
            for channel in range(3):
                values = sorted(pixel[channel] for pixel in pixels)
                channel_background.append(values[min(len(values) - 1, round(0.9 * (len(values) - 1)))])
            distances = [
                max(abs(pixel[channel] - channel_background[channel]) for channel in range(3))
                for pixel in pixels
            ]
            maximum = max(distances, default=1)
            mask = Image.new("L", rgb_region.size)
            mask.putdata([round(255 * distance / max(1, maximum)) for distance in distances])
            region = mask
        elif representation != "rgb":
            raise ValueError(f"unsupported pixel representation: {representation}")
        reduced = region.resize((size[0], size[1]), resampling)
        flattened = (
            reduced.get_flattened_data()
            if hasattr(reduced, "get_flattened_data")
            else reduced.getdata()
        )
        if representation in {"bright_mask", "dark_mask", "ink_mask", "local_contrast"}:
            return [value / 255.0 for value in flattened]
        return [channel / 255.0 for pixel in flattened for channel in pixel]


def _mean(rows: list[list[float]]) -> list[float]:
    if not rows:
        raise ValueError("cannot calculate a centroid from zero rows")
    return [sum(column) / len(rows) for column in zip(*rows)]


def _scale(rows: list[list[float]]) -> list[float]:
    means = _mean(rows)
    if len(rows) == 1:
        return [1.0 for _ in means]
    result: list[float] = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in rows) / len(rows)
        result.append(max(math.sqrt(variance), 0.02))
    return result


def fit_centroids(rows: list[tuple[list[float], str]]) -> dict[str, object]:
    if not rows:
        raise ValueError("training rows must not be empty")
    dimensions = {len(feature) for feature, _ in rows}
    if len(dimensions) != 1:
        raise ValueError("all feature vectors must have the same dimension")
    grouped: dict[str, list[list[float]]] = {}
    for feature, label in rows:
        grouped.setdefault(label, []).append(feature)
    features = [feature for feature, _ in rows]
    scale = _scale(features)
    return {
        "feature_dimension": dimensions.pop(),
        "scale": scale,
        "centroids": {label: _mean(grouped[label]) for label in sorted(grouped)},
        "class_counts": {label: len(grouped[label]) for label in sorted(grouped)},
    }


def predict_centroids(feature: list[float], model: dict[str, object]) -> tuple[str, float]:
    dimension = int(model["feature_dimension"])
    if len(feature) != dimension:
        raise ValueError(f"feature dimension {len(feature)} does not match checkpoint {dimension}")
    scale = [float(value) for value in model["scale"]]  # type: ignore[index]
    centroids: dict[str, list[float]] = model["centroids"]  # type: ignore[assignment]
    distances = {
        label: math.sqrt(
            sum(((value - float(centroid[index])) / scale[index]) ** 2 for index, value in enumerate(feature))
            / max(1, dimension)
        )
        for label, centroid in centroids.items()
    }
    ordered = sorted(distances.items(), key=lambda item: (item[1], item[0]))
    predicted, best = ordered[0]
    if len(ordered) == 1:
        confidence = 1.0
    else:
        second = ordered[1][1]
        confidence = max(0.0, min(1.0, (second - best) / max(second, 1e-12)))
    return predicted, confidence


@dataclass(frozen=True)
class BaselineCheckpoint:
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "BaselineCheckpoint":
        checkpoint_path = Path(path).expanduser().resolve()
        payload = _load_json(checkpoint_path)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported APC baseline checkpoint schema")
        if payload.get("model_kind") != MODEL_KIND:
            raise ValueError("unsupported APC baseline model kind")
        expected = payload.get("checkpoint_sha256")
        material = dict(payload)
        material.pop("checkpoint_sha256", None)
        if expected != canonical_sha256(material):
            raise ValueError("checkpoint fingerprint does not match its contents")
        return cls(payload)

    @property
    def training_sessions(self) -> set[str]:
        return set(str(value) for value in self.payload["training"]["capture_sessions"])


def predict_image(checkpoint: BaselineCheckpoint, image_path: str | Path) -> dict[str, dict[str, object]]:
    path = Path(image_path).expanduser().resolve()
    predictions: dict[str, dict[str, object]] = {}
    for head, model in checkpoint.payload["heads"].items():
        feature = extract_feature(path, model["feature"])
        label, confidence = predict_centroids(feature, model)
        predictions[head] = {"value": label, "confidence": confidence}
    return predictions


def _manifest_annotations(manifest_path: Path) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    manifest = _load_json(manifest_path)
    rows = []
    for raw_path in manifest["annotation_files"]:
        path = _annotation_path(manifest_path, str(raw_path))
        rows.append((path, _load_json(path)))
    return manifest, rows


def train_baseline(
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
    training_rows = [(path, item) for path, item in annotations if str(item["capture_session_id"]) in train_sessions]
    if not training_rows:
        raise ValueError("manifest training split has no annotations")
    heads: dict[str, Any] = {}
    for head, feature_config in DEFAULT_FEATURES.items():
        rows = [
            (extract_feature(_image_path(path, item), feature_config), _label(item, head))
            for path, item in training_rows
        ]
        heads[head] = {"feature": feature_config, **fit_centroids(rows)}
    material: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "model_role": "pipeline_smoke_baseline_not_for_promotion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "training": {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "dataset_fingerprints": manifest["fingerprints"],
            "split_fingerprint": manifest["fingerprints"]["split_sha256"],
            "capture_sessions": sorted(train_sessions),
            "examples": len(training_rows),
        },
        "supported_outputs": sorted(heads),
        "unsupported_critical_outputs": [
            "hero_cards",
            "board_cards",
            "seat_stacks_bb",
            "pot_bb",
            "to_call_bb",
            "dealer_seat",
            "observed_action",
        ],
        "heads": heads,
    }
    material["checkpoint_sha256"] = canonical_sha256(material)
    output = Path(checkpoint_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(material, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return material


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate_baseline(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation split must be validation or test")
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    checkpoint = BaselineCheckpoint.load(checkpoint_file)
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = set(str(value) for value in manifest["splits"][split])
    overlap = checkpoint.training_sessions & eval_sessions
    if overlap:
        raise ValueError(f"held-out evaluation leaks training sessions: {sorted(overlap)}")
    rows = [(path, item) for path, item in annotations if str(item["capture_session_id"]) in eval_sessions]
    if not rows:
        raise ValueError(f"manifest {split} split has no annotations")
    correct = {head: 0 for head in checkpoint.payload["heads"]}
    confidence_rows: dict[str, list[float]] = {head: [] for head in checkpoint.payload["heads"]}
    joint = 0
    latencies: list[float] = []
    prediction_digest_rows: list[dict[str, object]] = []
    for annotation_path, annotation in rows:
        started = time.perf_counter()
        predictions = predict_image(checkpoint, _image_path(annotation_path, annotation))
        latencies.append((time.perf_counter() - started) * 1000.0)
        sample_correct = True
        digest_prediction: dict[str, str] = {}
        for head, prediction in predictions.items():
            expected = _label(annotation, head)
            predicted = str(prediction["value"])
            matched = predicted == expected
            correct[head] += int(matched)
            confidence_rows[head].append(float(prediction["confidence"]))
            sample_correct = sample_correct and matched
            digest_prediction[head] = predicted
        joint += int(sample_correct)
        prediction_digest_rows.append({"sample_id": annotation["sample_id"], "predictions": digest_prediction})
    count = len(rows)
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "held_out_perception_smoke",
        "promotion_eligible": False,
        "checkpoint_sha256": checkpoint.payload["checkpoint_sha256"],
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": [],
        "examples": count,
        "metrics": {
            "accuracy": {head: correct[head] / count for head in sorted(correct)},
            "joint_supported_state_accuracy": joint / count,
            "mean_confidence": {
                head: statistics.fmean(values) if values else 0.0
                for head, values in sorted(confidence_rows.items())
            },
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "supported_outputs": checkpoint.payload["supported_outputs"],
        "unsupported_critical_outputs": checkpoint.payload["unsupported_critical_outputs"],
        "prediction_sha256": canonical_sha256(prediction_digest_rows),
        "limitations": [
            "Synthetic renderer only; no controlled visible-table captures are represented.",
            "Nearest-centroid heads validate plumbing, not production perception quality.",
            "Critical card, numeric, dealer and event heads are intentionally absent.",
            "Confidence margins are uncalibrated and must not pass the APC confidence gate.",
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate APC's pixel smoke baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Fit pixel centroids on the manifest train split")
    train.add_argument("manifest", type=Path)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--seed", type=int, default=20260802)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate without using train-split annotations")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            result = train_baseline(args.manifest, args.checkpoint, seed=args.seed)
            summary = {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "training_examples": result["training"]["examples"],
                "supported_outputs": result["supported_outputs"],
            }
        else:
            result = evaluate_baseline(args.checkpoint, args.manifest, split=args.split)
            if args.output:
                _write_json(args.output.expanduser().resolve(), result)
            summary = result
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
