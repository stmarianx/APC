from __future__ import annotations

import argparse
import copy
import json
import math
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
    predict_centroids,
    predict_image,
)
from apc.perception.card_baseline import fit_exemplars, predict_exemplars
from apc.perception.stack_baseline import (
    DECIMAL_CELL_WIDTH,
    DIGIT_CELL_WIDTH,
    STACK_TOKEN_RE,
    _shape_geometry,
    _token_shape,
    digit_box,
)
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


SCHEMA_VERSION = "1.0.0"
MODEL_KIND = "learned_geometry_segmented_numeric_exemplar_shape_table_state_pixel_baseline"
CENTROID_SHAPE_MODEL_KIND = "learned_geometry_segmented_numeric_table_state_pixel_baseline"
LEGACY_MODEL_KIND = "learned_geometry_table_state_pixel_baseline"
HERO_FEATURE = {"size": [32, 14], "representation": "rgb"}
DEALER_FEATURE = {"size": [18, 18], "representation": "bright_mask"}
POT_FEATURE = {"size": [44, 12], "representation": "local_contrast"}
CALL_FEATURE = {"size": [60, 16], "representation": "local_contrast"}
DEALER_RELATIVE_CROP = (0.80, -0.24, 1.10, 0.36)
POT_REGION_WIDTH_PX = 180.0
CALL_REGION_WIDTH_PX = 145.0
POT_PREFIX_PX = 16.0
CALL_PREFIX_PX = 20.0
NUMBER_SUFFIX_PX = 14.0
NUMBER_VERTICAL = (0.22, 0.78)
TIGHT_GLYPH_SIZE = (12, 20)
TIGHT_GLYPH_CANVAS = (8, 12)


def _mean_box(boxes: list[dict[str, object]]) -> dict[str, float]:
    if not boxes:
        raise ValueError("cannot learn geometry from zero boxes")
    return {
        key: sum(float(box[key]) for box in boxes) / len(boxes)
        for key in ("x", "y", "width", "height")
    }


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


def number_character_box(
    container: dict[str, object],
    *,
    region_width_px: float,
    prefix_px: float,
    character_count: int,
    character_index: int,
    decimal_index: int | None,
) -> dict[str, float]:
    if not 1 <= character_count <= 5 or not 0 <= character_index < character_count:
        raise ValueError("numeric label character geometry is out of range")
    widths_px = [
        DECIMAL_CELL_WIDTH * 160 if index == decimal_index else DIGIT_CELL_WIDTH * 160
        for index in range(character_count)
    ]
    total_width_px = prefix_px + sum(widths_px) + NUMBER_SUFFIX_PX
    left_px = (region_width_px - total_width_px) / 2 + prefix_px + sum(widths_px[:character_index])
    right_px = left_px + widths_px[character_index]
    return _relative_box(
        container,
        (
            left_px / region_width_px,
            NUMBER_VERTICAL[0],
            right_px / region_width_px,
            NUMBER_VERTICAL[1],
        ),
    )


def _tight_glyph_feature(
    image_path: Path,
    box: dict[str, object],
    *,
    polarity: str,
) -> list[float]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("APC numeric OCR requires Pillow") from error
    if polarity not in {"bright", "dark"}:
        raise ValueError("glyph polarity must be bright or dark")
    with Image.open(image_path) as source:
        image = source.convert("L")
        width, height = image.size
        left = max(0, min(width - 1, round(float(box["x"]) * width)))
        top = max(0, min(height - 1, round(float(box["y"]) * height)))
        right = max(left + 1, min(width, round((float(box["x"]) + float(box["width"])) * width)))
        bottom = max(top + 1, min(height, round((float(box["y"]) + float(box["height"])) * height)))
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
        flattened = reduced.get_flattened_data() if hasattr(reduced, "get_flattened_data") else reduced.getdata()
        return [value / 255.0 for value in flattened]


def _numeric_training_rows(
    *,
    image_path: Path,
    container: dict[str, object],
    token: str,
    region_feature: dict[str, object],
    character_polarity: str,
    region_width_px: float,
    prefix_px: float,
) -> tuple[tuple[list[float], str], list[tuple[list[float], str]]]:
    if not STACK_TOKEN_RE.fullmatch(token) or not 1 <= len(token) <= 5:
        raise ValueError(f"table-state numeric token is unsupported: {token!r}")
    shape_row = (extract_feature(image_path, _config(container, region_feature)), _token_shape(token))
    decimal_index = token.find(".") if "." in token else None
    character_rows = []
    for index, character in enumerate(token):
        box = number_character_box(
            container,
            region_width_px=region_width_px,
            prefix_px=prefix_px,
            character_count=len(token),
            character_index=index,
            decimal_index=decimal_index,
        )
        character_rows.append((_tight_glyph_feature(image_path, box, polarity=character_polarity), character))
    return shape_row, character_rows


def _predict_numeric_token(
    *,
    image_path: Path,
    container: dict[str, object],
    region_feature: dict[str, object],
    character_polarity: str,
    region_width_px: float,
    prefix_px: float,
    shape_model: dict[str, object],
    character_model: dict[str, object],
) -> tuple[str, float]:
    shape_feature = extract_feature(image_path, _config(container, region_feature))
    if "exemplars" in shape_model:
        shape, shape_confidence = predict_exemplars(shape_feature, shape_model)
    else:
        shape, shape_confidence = predict_centroids(shape_feature, shape_model)
    count, decimal_index = _shape_geometry(shape)
    characters: list[str] = []
    confidences = [shape_confidence]
    for index in range(count):
        if index == decimal_index:
            characters.append(".")
            confidences.append(shape_confidence)
            continue
        box = number_character_box(
            container,
            region_width_px=region_width_px,
            prefix_px=prefix_px,
            character_count=count,
            character_index=index,
            decimal_index=decimal_index,
        )
        character, confidence = predict_exemplars(
            _tight_glyph_feature(image_path, box, polarity=character_polarity),
            character_model,
            allowed_labels=set("0123456789"),
        )
        characters.append(character)
        confidences.append(confidence)
    token = "".join(characters)
    if not STACK_TOKEN_RE.fullmatch(token):
        raise ValueError(f"table-state numeric OCR produced invalid BB token: {token!r}")
    return token, min(confidences)


def _distance_to_label(feature: list[float], model: dict[str, object], label: str) -> float:
    centroid = model["centroids"][label]
    scale = model["scale"]
    return math.sqrt(
        sum(((value - float(centroid[index])) / float(scale[index])) ** 2 for index, value in enumerate(feature))
        / max(1, len(feature))
    )


def _learn_geometry(rows: list[tuple[Path, dict[str, Any]]]) -> dict[str, object]:
    seats: dict[str, dict[int, list[dict[str, object]]]] = {}
    pots: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []
    for _, annotation in rows:
        layout = str(annotation["environment"]["layout_id"])
        layout_seats = seats.setdefault(layout, {})
        for index, seat in enumerate(annotation["objects"]["seats"]):
            layout_seats.setdefault(index, []).append(seat["box"])
        pots.append(annotation["objects"]["pot"]["box"])
        calls.extend(
            button["box"]
            for button in annotation["objects"]["action_buttons"]
            if button["action"] == "call"
        )
    return {
        "seats": {
            layout: [_mean_box(indexes[index]) for index in range(max(indexes) + 1)]
            for layout, indexes in sorted(seats.items())
        },
        "pot": _mean_box(pots),
        "call_button": _mean_box(calls),
    }


def train_table_state_baseline(
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
    geometry = _learn_geometry(rows)
    hero_rows: list[tuple[list[float], str]] = []
    dealer_rows: list[tuple[list[float], str]] = []
    pot_shape_rows: list[tuple[list[float], str]] = []
    call_shape_rows: list[tuple[list[float], str]] = []
    pot_character_rows: list[tuple[list[float], str]] = []
    call_character_rows: list[tuple[list[float], str]] = []
    for annotation_path, annotation in rows:
        image_path = _image_path(annotation_path, annotation)
        for seat in annotation["objects"]["seats"]:
            hero_rows.append((extract_feature(image_path, _config(seat["box"], HERO_FEATURE)), "yes" if seat["is_hero"] else "no"))
            dealer_box = _relative_box(seat["box"], DEALER_RELATIVE_CROP)
            dealer_rows.append((extract_feature(image_path, _config(dealer_box, DEALER_FEATURE)), "yes" if seat["has_dealer_button"] else "no"))
        pot_shape, pot_characters = _numeric_training_rows(
            image_path=image_path,
            container=annotation["objects"]["pot"]["box"],
            token=str(annotation["state"]["pot_bb"]),
            region_feature=POT_FEATURE,
            character_polarity="bright",
            region_width_px=POT_REGION_WIDTH_PX,
            prefix_px=POT_PREFIX_PX,
        )
        pot_shape_rows.append(pot_shape)
        pot_character_rows.extend(pot_characters)
        for seat in annotation["objects"]["seats"]:
            stack_token = str(seat["stack_bb"])
            stack_decimal_index = stack_token.find(".") if "." in stack_token else None
            for index, character in enumerate(stack_token):
                stack_character_box = digit_box(
                    seat["box"],
                    digit_count=len(stack_token),
                    digit_index=index,
                    decimal_index=stack_decimal_index,
                )
                pot_character_rows.append(
                    (
                        _tight_glyph_feature(image_path, stack_character_box, polarity="bright"),
                        character,
                    )
                )
        for button in annotation["objects"]["action_buttons"]:
            if button["action"] == "call":
                call_shape, call_characters = _numeric_training_rows(
                    image_path=image_path,
                    container=button["box"],
                    token=str(annotation["state"]["to_call_bb"]),
                    region_feature=CALL_FEATURE,
                    character_polarity="dark",
                    region_width_px=CALL_REGION_WIDTH_PX,
                    prefix_px=CALL_PREFIX_PX,
                )
                call_shape_rows.append(call_shape)
                call_character_rows.extend(call_characters)
    material: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "model_role": "synthetic_table_state_pipeline_baseline_not_for_promotion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "training": {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "dataset_fingerprints": manifest["fingerprints"],
            "capture_sessions": sorted(train_sessions),
            "frames": len(rows),
        },
        "geometry": geometry,
        "features": {
            "hero": HERO_FEATURE,
            "dealer": DEALER_FEATURE,
            "dealer_relative_crop": list(DEALER_RELATIVE_CROP),
            "pot": POT_FEATURE,
            "call": CALL_FEATURE,
            "tight_glyph_size": list(TIGHT_GLYPH_SIZE),
            "tight_glyph_canvas": list(TIGHT_GLYPH_CANVAS),
        },
        "hero_model": fit_centroids(hero_rows),
        "dealer_model": fit_centroids(dealer_rows),
        "pot_shape_model": fit_exemplars(pot_shape_rows),
        "call_shape_model": fit_exemplars(call_shape_rows),
        "pot_character_model": fit_exemplars(pot_character_rows),
        "call_character_model": fit_exemplars(call_character_rows),
    }
    material["checkpoint_sha256"] = canonical_sha256(material)
    output = Path(checkpoint_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(material, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return material


def load_table_state_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("model_kind") not in {
        MODEL_KIND,
        CENTROID_SHAPE_MODEL_KIND,
        LEGACY_MODEL_KIND,
    }:
        raise ValueError("unsupported APC table-state checkpoint")
    material = dict(payload)
    expected = material.pop("checkpoint_sha256", None)
    if expected != canonical_sha256(material):
        raise ValueError("table-state checkpoint fingerprint does not match its contents")
    return payload


def upgrade_table_shape_models(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    checkpoint = load_table_state_checkpoint(checkpoint_path)
    if checkpoint["model_kind"] == LEGACY_MODEL_KIND:
        raise ValueError("Legacy closed-token checkpoints cannot be shape-upgraded")
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    if checkpoint["training"]["dataset_fingerprints"] != manifest["fingerprints"]:
        raise ValueError("Shape upgrade manifest does not match checkpoint training data")
    train_sessions = set(str(value) for value in checkpoint["training"]["capture_sessions"])
    rows = [
        (path, annotation)
        for path, annotation in annotations
        if str(annotation["capture_session_id"]) in train_sessions
    ]
    pot_rows: list[tuple[list[float], str]] = []
    call_rows: list[tuple[list[float], str]] = []
    for annotation_path, annotation in rows:
        image_path = _image_path(annotation_path, annotation)
        pot_rows.append(
            (
                extract_feature(
                    image_path,
                    _config(annotation["objects"]["pot"]["box"], POT_FEATURE),
                ),
                _token_shape(str(annotation["state"]["pot_bb"])),
            )
        )
        for button in annotation["objects"]["action_buttons"]:
            if button["action"] == "call":
                call_rows.append(
                    (
                        extract_feature(image_path, _config(button["box"], CALL_FEATURE)),
                        _token_shape(str(annotation["state"]["to_call_bb"])),
                    )
                )
    material = copy.deepcopy(checkpoint)
    source_checkpoint_sha256 = material.pop("checkpoint_sha256")
    material["model_kind"] = MODEL_KIND
    material["created_at"] = datetime.now(timezone.utc).isoformat()
    material["pot_shape_model"] = fit_exemplars(pot_rows)
    material["call_shape_model"] = fit_exemplars(call_rows)
    material["training"]["shape_upgrade"] = {
        "kind": "train_split_exemplar_shape",
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "frames": len(rows),
    }
    material["checkpoint_sha256"] = canonical_sha256(material)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(material, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return material


def predict_table_state(
    checkpoint: dict[str, Any],
    base_checkpoint: BaselineCheckpoint,
    image_path: str | Path,
    *,
    base_prediction: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    path = Path(image_path).expanduser().resolve()
    base = base_prediction or predict_image(base_checkpoint, path)
    layout = str(base["layout_id"]["value"])
    seat_boxes = checkpoint["geometry"]["seats"].get(layout)
    if not seat_boxes:
        raise ValueError(f"table-state geometry has no learned layout: {layout}")
    dealer_relative = tuple(float(value) for value in checkpoint["features"]["dealer_relative_crop"])
    hero_candidates = []
    dealer_candidates = []
    for index, box in enumerate(seat_boxes, start=1):
        hero_feature = extract_feature(path, _config(box, checkpoint["features"]["hero"]))
        dealer_box = _relative_box(box, dealer_relative)
        dealer_feature = extract_feature(path, _config(dealer_box, checkpoint["features"]["dealer"]))
        hero_candidates.append((_distance_to_label(hero_feature, checkpoint["hero_model"], "yes"), index))
        dealer_candidates.append((_distance_to_label(dealer_feature, checkpoint["dealer_model"], "yes"), index))
    if checkpoint["model_kind"] == LEGACY_MODEL_KIND:
        pot_bb, pot_confidence = predict_exemplars(
            extract_feature(path, _config(checkpoint["geometry"]["pot"], checkpoint["features"]["pot"])),
            checkpoint["pot_model"],
        )
    else:
        pot_bb, pot_confidence = _predict_numeric_token(
            image_path=path,
            container=checkpoint["geometry"]["pot"],
            region_feature=checkpoint["features"]["pot"],
            character_polarity="bright",
            region_width_px=POT_REGION_WIDTH_PX,
            prefix_px=POT_PREFIX_PX,
            shape_model=checkpoint["pot_shape_model"],
            character_model=checkpoint["pot_character_model"],
        )
    legal_actions = str(base["legal_actions"]["value"]).split("+")
    if "call" in legal_actions:
        if checkpoint["model_kind"] == LEGACY_MODEL_KIND:
            to_call_bb, call_confidence = predict_exemplars(
                extract_feature(path, _config(checkpoint["geometry"]["call_button"], checkpoint["features"]["call"])),
                checkpoint["call_model"],
            )
        else:
            to_call_bb, call_confidence = _predict_numeric_token(
                image_path=path,
                container=checkpoint["geometry"]["call_button"],
                region_feature=checkpoint["features"]["call"],
                character_polarity="dark",
                region_width_px=CALL_REGION_WIDTH_PX,
                prefix_px=CALL_PREFIX_PX,
                shape_model=checkpoint["call_shape_model"],
                character_model=checkpoint["call_character_model"],
            )
    else:
        to_call_bb, call_confidence = "0", float(base["legal_actions"]["confidence"])
    return {
        "layout_id": base["layout_id"],
        "street": base["street"],
        "legal_actions": base["legal_actions"],
        "hero_seat": min(hero_candidates)[1],
        "dealer_seat": min(dealer_candidates)[1],
        "pot_bb": {"value": pot_bb, "confidence": pot_confidence},
        "to_call_bb": {"value": to_call_bb, "confidence": call_confidence},
    }


def evaluate_table_state_baseline(
    checkpoint_path: str | Path,
    base_checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "validation",
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation split must be validation or test")
    checkpoint = load_table_state_checkpoint(checkpoint_path)
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
    heads = ("hero_seat", "dealer_seat", "pot_bb", "to_call_bb")
    correct = Counter({head: 0 for head in heads})
    joint = 0
    latencies: list[float] = []
    errors: list[dict[str, object]] = []
    digest_rows = []
    for annotation_path, annotation in rows:
        started = time.perf_counter()
        prediction = predict_table_state(checkpoint, base_checkpoint, _image_path(annotation_path, annotation))
        latencies.append((time.perf_counter() - started) * 1000.0)
        expected = {
            "hero_seat": annotation["state"]["hero_seat"],
            "dealer_seat": annotation["state"]["dealer_seat"],
            "pot_bb": str(annotation["state"]["pot_bb"]),
            "to_call_bb": str(annotation["state"]["to_call_bb"]),
        }
        predicted = {
            "hero_seat": prediction["hero_seat"],
            "dealer_seat": prediction["dealer_seat"],
            "pot_bb": prediction["pot_bb"]["value"],
            "to_call_bb": prediction["to_call_bb"]["value"],
        }
        matched = {head: predicted[head] == expected[head] for head in heads}
        for head in heads:
            correct[head] += int(matched[head])
        joint += int(all(matched.values()))
        if not all(matched.values()):
            errors.append({"sample_id": annotation["sample_id"], "expected": expected, "predicted": predicted})
        digest_rows.append({"sample_id": annotation["sample_id"], "prediction": predicted})
    if not rows:
        raise ValueError("held-out split has no frames")
    count = len(rows)
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "held_out_table_state_perception_smoke",
        "promotion_eligible": False,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "base_checkpoint_sha256": base_checkpoint.payload["checkpoint_sha256"],
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": [],
        "frames": count,
        "metrics": {
            "accuracy": {head: correct[head] / count for head in heads},
            "joint_supported_state_accuracy": joint / count,
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "errors": errors[:20],
        "prediction_sha256": canonical_sha256(digest_rows),
        "limitations": [
            "Segmented numeric synthetic renderer baseline; this is not general OCR.",
            "Observed actions are handled by the separate temporal head.",
            "Geometry, confidence and value classes are not calibrated for controlled visible tables.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate APC's table-state pixel baseline.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--seed", type=int, default=118092)
    upgrade = commands.add_parser("upgrade-shapes")
    upgrade.add_argument("checkpoint", type=Path)
    upgrade.add_argument("manifest", type=Path)
    upgrade.add_argument("--output", type=Path, required=True)
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
            result = train_table_state_baseline(args.manifest, args.checkpoint, seed=args.seed)
            summary = {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "training_frames": result["training"]["frames"],
            }
        elif args.command == "upgrade-shapes":
            result = upgrade_table_shape_models(
                args.checkpoint, args.manifest, args.output
            )
            summary = {
                "checkpoint": str(args.output.expanduser().resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "source_checkpoint_sha256": result["training"]["shape_upgrade"][
                    "source_checkpoint_sha256"
                ],
                "training_frames": result["training"]["shape_upgrade"]["frames"],
            }
        else:
            result = evaluate_table_state_baseline(
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
