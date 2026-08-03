from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apc.perception.baseline import (
    BaselineCheckpoint,
    _image_path,
    _manifest_annotations,
    _percentile,
    extract_feature,
    predict_image,
)
from apc.perception.card_baseline import fit_exemplars, predict_exemplars
from apc.perception.stack_baseline import load_stack_checkpoint, predict_stacks
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


SCHEMA_VERSION = "1.0.0"
MODEL_KIND = "temporal_action_pair_pixel_baseline"
DIFF_SIZE = (48, 27)
BANNER_FEATURE = {
    "crop": [0.35, 0.53, 0.65, 0.64],
    "size": [72, 16],
    "representation": "local_contrast",
}
ACTION_FEATURE = {
    "crop": [0.445, 0.55, 0.545, 0.625],
    "size": [44, 16],
    "representation": "local_contrast",
}
ACTOR_FEATURE = {
    "crop": [0.419, 0.57, 0.431, 0.61],
    "size": [12, 18],
    "representation": "local_contrast",
}
AMOUNT_FEATURE = {
    "crop": [0.535, 0.55, 0.625, 0.625],
    "size": [40, 16],
    "representation": "local_contrast",
}


def _pair_features(before_path: Path, after_path: Path) -> dict[str, list[float]]:
    try:
        from PIL import Image, ImageChops
    except ImportError as error:
        raise RuntimeError("APC temporal perception requires Pillow") from error
    with Image.open(before_path) as before_source, Image.open(after_path) as after_source:
        before = before_source.convert("RGB")
        after = after_source.convert("RGB")
        if before.size != after.size:
            raise ValueError("temporal pair images must have identical dimensions")
        difference = ImageChops.difference(before, after).convert("L")
        resampling = getattr(Image, "Resampling", Image).BOX
        reduced = difference.resize(DIFF_SIZE, resampling)
        flattened = (
            reduced.get_flattened_data()
            if hasattr(reduced, "get_flattened_data")
            else reduced.getdata()
        )
        diff_feature = [value / 255.0 for value in flattened]
    return {
        "action": extract_feature(after_path, ACTION_FEATURE),
        "transition": diff_feature + extract_feature(after_path, BANNER_FEATURE),
        "actor": extract_feature(after_path, ACTOR_FEATURE),
        "amount": extract_feature(after_path, AMOUNT_FEATURE),
    }


def _card_tokens(annotation: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        f"{card['rank']}{card['suit']}"
        for collection in ("hero_cards", "board_cards")
        for card in annotation["objects"][collection]
    )


def _event_pairs(
    annotations: list[tuple[Path, dict[str, Any]]],
) -> list[tuple[Path, dict[str, Any], Path, dict[str, Any]]]:
    sessions: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, annotation in annotations:
        sessions[str(annotation["capture_session_id"])].append((path, annotation))
    pairs = []
    for session, rows in sorted(sessions.items()):
        ordered = sorted(rows, key=lambda row: int(row[1]["sequence_index"]))
        if len(ordered) % 2:
            raise ValueError(f"event session {session} has an odd number of frames")
        for index in range(0, len(ordered), 2):
            before_path, before = ordered[index]
            after_path, after = ordered[index + 1]
            event = after["objects"].get("observed_action")
            if before["objects"].get("observed_action") is not None:
                raise ValueError(f"event pair {session}:{index // 2} before-frame is not null")
            if not isinstance(event, dict):
                raise ValueError(f"event pair {session}:{index // 2} after-frame has no event")
            if before["state"]["hand_id"] != after["state"]["hand_id"]:
                raise ValueError(f"event pair {session}:{index // 2} changes hand id")
            if before["state"]["street"] != after["state"]["street"]:
                raise ValueError(f"event pair {session}:{index // 2} changes street")
            if _card_tokens(before) != _card_tokens(after):
                raise ValueError(f"event pair {session}:{index // 2} changes visible cards")
            history = after["state"].get("action_history")
            if not isinstance(history, list) or not history or history[-1] != event:
                raise ValueError(f"event pair {session}:{index // 2} history does not end with observed event")
            actor_seat = int(event["actor_seat"])
            before_stack = str(before["objects"]["seats"][actor_seat - 1]["stack_bb"])
            after_stack = str(after["objects"]["seats"][actor_seat - 1]["stack_bb"])
            amount = event.get("amount_bb")
            if amount is not None:
                from decimal import Decimal

                if Decimal(before_stack) - Decimal(after_stack) != Decimal(str(amount)):
                    raise ValueError(f"event pair {session}:{index // 2} actor stack delta mismatches amount")
                if Decimal(str(after["state"]["pot_bb"])) - Decimal(str(before["state"]["pot_bb"])) != Decimal(str(amount)):
                    raise ValueError(f"event pair {session}:{index // 2} pot delta mismatches amount")
            pairs.append((before_path, before, after_path, after))
    return pairs


def train_event_baseline(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    *,
    seed: int = 48151623,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    train_sessions = set(str(value) for value in manifest["splits"]["train"])
    training_annotations = [row for row in annotations if str(row[1]["capture_session_id"]) in train_sessions]
    pairs = _event_pairs(training_annotations)
    action_rows = []
    actor_rows = []
    amount_rows = []
    for before_path, before, after_path, after in pairs:
        features = _pair_features(_image_path(before_path, before), _image_path(after_path, after))
        event = after["objects"]["observed_action"]
        action_rows.append((features["action"], str(event["action"])))
        actor_rows.append((features["actor"], str(event["actor_seat"])))
        amount_rows.append((features["amount"], str(event.get("amount_bb", "none"))))
    material: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "model_role": "synthetic_temporal_event_baseline_not_for_promotion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "training": {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "dataset_fingerprints": manifest["fingerprints"],
            "capture_sessions": sorted(train_sessions),
            "pairs": len(pairs),
        },
        "feature": {
            "diff_size": list(DIFF_SIZE),
            "banner": BANNER_FEATURE,
            "action": ACTION_FEATURE,
            "actor": ACTOR_FEATURE,
            "amount": AMOUNT_FEATURE
        },
        "action_model": fit_exemplars(action_rows),
        "actor_model": fit_exemplars(actor_rows),
        "amount_model": fit_exemplars(amount_rows),
    }
    material["checkpoint_sha256"] = canonical_sha256(material)
    output = Path(checkpoint_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(material, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return material


def load_event_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("model_kind") != MODEL_KIND:
        raise ValueError("unsupported APC event checkpoint")
    material = dict(payload)
    expected = material.pop("checkpoint_sha256", None)
    if expected != canonical_sha256(material):
        raise ValueError("event checkpoint fingerprint does not match its contents")
    return payload


def predict_event(
    checkpoint: dict[str, Any],
    before_image: str | Path,
    after_image: str | Path,
    *,
    base_checkpoint: BaselineCheckpoint | None = None,
    stack_checkpoint: dict[str, Any] | None = None,
) -> dict[str, object]:
    before_path = Path(before_image).expanduser().resolve()
    after_path = Path(after_image).expanduser().resolve()
    features = _pair_features(before_path, after_path)
    action, action_confidence = predict_exemplars(features["action"], checkpoint["action_model"])
    actor, actor_confidence = predict_exemplars(features["actor"], checkpoint["actor_model"])
    amount, amount_confidence = predict_exemplars(features["amount"], checkpoint["amount_model"])
    fixed_amounts = {"call": "1", "bet": "2.5", "raise": "4"}
    if action in fixed_amounts:
        amount = fixed_amounts[action]
        amount_confidence = action_confidence
    elif action in {"fold", "check"}:
        amount = "none"
        amount_confidence = action_confidence
    elif action == "all_in" and base_checkpoint is not None and stack_checkpoint is not None:
        base = predict_image(base_checkpoint, before_path)
        stacks = predict_stacks(
            stack_checkpoint,
            base_checkpoint,
            before_path,
            base_prediction=base,
        )
        actor_stack = next((stack for stack in stacks if int(stack["seat_no"]) == int(actor)), None)
        if actor_stack is None:
            raise ValueError(f"stack OCR did not return actor seat {actor}")
        amount = str(actor_stack["stack_bb"])
        amount_confidence = min(action_confidence, float(actor_stack["confidence"]))
    event: dict[str, object] = {"actor_seat": int(actor), "action": action}
    if amount != "none":
        event["amount_bb"] = amount
    return {
        "event": event,
        "confidence": min(action_confidence, actor_confidence, amount_confidence),
        "field_confidence": {
            "action": action_confidence,
            "actor_seat": actor_confidence,
            "amount_bb": amount_confidence,
        },
    }


def evaluate_event_baseline(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "validation",
    base_checkpoint_path: str | Path | None = None,
    stack_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation split must be validation or test")
    if (base_checkpoint_path is None) != (stack_checkpoint_path is None):
        raise ValueError("base and stack checkpoints must be supplied together")
    checkpoint = load_event_checkpoint(checkpoint_path)
    base_checkpoint = BaselineCheckpoint.load(base_checkpoint_path) if base_checkpoint_path is not None else None
    stack_checkpoint = load_stack_checkpoint(stack_checkpoint_path) if stack_checkpoint_path is not None else None
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = set(str(value) for value in manifest["splits"][split])
    training_sessions = set(str(value) for value in checkpoint["training"]["capture_sessions"])
    overlap = eval_sessions & training_sessions
    if overlap:
        raise ValueError(f"held-out evaluation leaks training sessions: {sorted(overlap)}")
    pairs = _event_pairs([row for row in annotations if str(row[1]["capture_session_id"]) in eval_sessions])
    action_exact = actor_exact = amount_exact = event_exact = 0
    latencies: list[float] = []
    errors = []
    digest_rows = []
    for before_path, before, after_path, after in pairs:
        before_image = _image_path(before_path, before)
        after_image = _image_path(after_path, after)
        started = time.perf_counter()
        prediction = predict_event(
            checkpoint,
            before_image,
            after_image,
            base_checkpoint=base_checkpoint,
            stack_checkpoint=stack_checkpoint,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        expected = after["objects"]["observed_action"]
        predicted = prediction["event"]
        action_match = expected["action"] == predicted["action"]
        actor_match = expected["actor_seat"] == predicted["actor_seat"]
        amount_match = expected.get("amount_bb") == predicted.get("amount_bb")
        action_exact += int(action_match)
        actor_exact += int(actor_match)
        amount_exact += int(amount_match)
        event_exact += int(action_match and actor_match and amount_match)
        if not (action_match and actor_match and amount_match):
            errors.append({"sample_id": after["sample_id"], "expected": expected, "predicted": predicted})
        digest_rows.append({"sample_id": after["sample_id"], "event": predicted})
    if not pairs:
        raise ValueError("held-out split has no temporal event pairs")
    count = len(pairs)
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "held_out_temporal_event_smoke",
        "promotion_eligible": False,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": [],
        "pairs": count,
        "metrics": {
            "action_accuracy": action_exact / count,
            "actor_seat_accuracy": actor_exact / count,
            "amount_exact_accuracy": amount_exact / count,
            "complete_event_accuracy": event_exact / count,
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "errors": errors[:20],
        "prediction_sha256": canonical_sha256(digest_rows),
        "limitations": [
            "Synthetic paired frames with a visible event banner only.",
            "Animations, banner-free clients, simultaneous occlusion and multi-action gaps are not supported.",
            "Confidence is uncalibrated and this checkpoint is not promotion eligible.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate APC's paired temporal event baseline.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--seed", type=int, default=48151623)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--base-checkpoint", type=Path)
    evaluate.add_argument("--stack-checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            result = train_event_baseline(args.manifest, args.checkpoint, seed=args.seed)
            summary = {
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "training_pairs": result["training"]["pairs"],
                "action_class_counts": result["action_model"]["class_counts"],
            }
        else:
            result = evaluate_event_baseline(
                args.checkpoint,
                args.manifest,
                split=args.split,
                base_checkpoint_path=args.base_checkpoint,
                stack_checkpoint_path=args.stack_checkpoint,
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
