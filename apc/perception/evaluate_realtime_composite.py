from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from apc.deadline import (
    TIER_REQUIREMENTS_MS,
    decision_window_from_observation,
    plan_deadline_decision,
)
from apc.perception.baseline import (
    BaselineCheckpoint,
    _image_path,
    _manifest_annotations,
    _percentile,
)
from apc.perception.card_baseline import load_card_checkpoint
from apc.perception.composite import infer_visible_state
from apc.perception.name_ocr_baseline import load_name_ocr_checkpoint
from apc.perception.stack_baseline import load_stack_checkpoint
from apc.perception.table_state_baseline import load_table_state_checkpoint
from apc.perception.turn_clock_baseline import load_turn_clock_checkpoint
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


SCHEMA_VERSION = "1.0.0"


def _card_token(card: dict[str, object]) -> str:
    return f"{card['rank']}{card['suit']}"


def supported_state_checks(
    annotation: dict[str, Any],
    visible_state: dict[str, Any],
) -> dict[str, bool]:
    objects = annotation["objects"]
    state = annotation["state"]
    expected_names = [
        (int(seat["seat_no"]), str(seat["player_name"]))
        for seat in objects["seats"]
    ]
    observed_names = [
        (int(row["seat_no"]), str(row["player_name"]))
        for row in visible_state.get("recognized_player_names", [])
    ]
    expected_stacks = [
        (int(seat["seat_no"]), Decimal(str(seat["stack_bb"])))
        for seat in objects["seats"]
    ]
    observed_stacks = [
        (int(row["seat_no"]), Decimal(str(row["stack_bb"])))
        for row in visible_state.get("seat_stacks_bb", [])
    ]
    clock = objects.get("turn_clock")
    return {
        "layout_id": visible_state.get("layout_id") == annotation["environment"]["layout_id"],
        "theme_id": visible_state.get("theme_id") == annotation["environment"]["theme_id"],
        "street": visible_state.get("street") == state["street"],
        "legal_actions": set(visible_state.get("legal_actions", []))
        == set(state["legal_actions"]),
        "hero_seat": visible_state.get("hero_seat") == state["hero_seat"],
        "dealer_seat": visible_state.get("dealer_seat") == state["dealer_seat"],
        "hero_cards": visible_state.get("hero_cards")
        == [_card_token(card) for card in objects["hero_cards"]],
        "board_cards": visible_state.get("board_cards")
        == [_card_token(card) for card in objects["board_cards"]],
        "pot_bb": Decimal(str(visible_state.get("pot_bb"))) == Decimal(str(state["pot_bb"])),
        "to_call_bb": Decimal(str(visible_state.get("to_call_bb")))
        == Decimal(str(state["to_call_bb"])),
        "seat_stacks_bb": observed_stacks == expected_stacks,
        "player_names": observed_names == expected_names,
        "decision_time_remaining_ms": isinstance(clock, dict)
        and visible_state.get("decision_time_remaining_ms") == clock["remaining_ms"],
    }


def evaluate_realtime_composite(
    manifest_path: str | Path,
    *,
    base_checkpoint_path: str | Path,
    card_checkpoint_path: str | Path,
    table_state_checkpoint_path: str | Path,
    stack_checkpoint_path: str | Path,
    turn_clock_checkpoint_path: str | Path,
    name_ocr_checkpoint_path: str | Path,
    split: str = "test",
    safety_margin_ms: int = 750,
    actuation_reserve_ms: int = 250,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("real-time evaluation split must be validation or test")
    manifest_file = Path(manifest_path).expanduser().resolve()
    validation = validate_manifest(manifest_file)
    if not validation["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(validation["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    sessions = {str(value) for value in manifest["splits"][split]}
    base = BaselineCheckpoint.load(base_checkpoint_path)
    card = load_card_checkpoint(card_checkpoint_path)
    table = load_table_state_checkpoint(table_state_checkpoint_path)
    stack = load_stack_checkpoint(stack_checkpoint_path)
    clock = load_turn_clock_checkpoint(turn_clock_checkpoint_path)
    names = load_name_ocr_checkpoint(name_ocr_checkpoint_path)
    checkpoint_dataset_ids = {
        str(base.payload["training"]["dataset_id"]),
        str(card["training"]["dataset_id"]),
        str(table["training"]["dataset_id"]),
        str(stack["training"]["dataset_id"]),
        str(clock["training"]["dataset_id"]),
        str(names["training"]["dataset_id"]),
    }
    if str(manifest["dataset_id"]) in checkpoint_dataset_ids:
        raise ValueError("real-time audit dataset was used by at least one checkpoint")

    available_tiers = tuple(sorted(TIER_REQUIREMENTS_MS))
    field_passes: Counter[str] = Counter()
    deadline_statuses: Counter[str] = Counter()
    strategy_tiers: Counter[str] = Counter()
    latencies: list[float] = []
    head_latencies: dict[str, list[float]] = {}
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for annotation_path, annotation in annotations:
        if str(annotation["capture_session_id"]) not in sessions:
            continue
        result = infer_visible_state(
            _image_path(annotation_path, annotation),
            base_checkpoint=base,
            card_checkpoint=card,
            table_state_checkpoint=table,
            stack_checkpoint=stack,
            turn_clock_checkpoint=clock,
            name_ocr_checkpoint=names,
        )
        latency_ms = float(result["latency_ms"])
        latencies.append(latency_ms)
        for head, head_latency in result["head_latency_ms"].items():
            head_latencies.setdefault(str(head), []).append(float(head_latency))
        visible = result["visible_state"]
        checks = supported_state_checks(annotation, visible)
        for field, passed in checks.items():
            field_passes[field] += int(passed)
        state_sha = canonical_sha256(visible)
        observed_at_ms = 1_000_000
        window = decision_window_from_observation(
            visible,
            state_revision=int(annotation["sequence_index"]),
            state_sha256=state_sha,
            observed_at_ms=observed_at_ms,
            safety_margin_ms=safety_margin_ms,
            actuation_reserve_ms=actuation_reserve_ms,
        )
        plan = plan_deadline_decision(
            window,
            now_ms=observed_at_ms + math.ceil(latency_ms),
            available_tiers=available_tiers,
        )
        deadline_statuses[str(plan["status"])] += 1
        if plan["strategy_tier"] is not None:
            strategy_tiers[str(plan["strategy_tier"])] += 1
        row = {
            "sample_id": annotation["sample_id"],
            "checks": checks,
            "joint_supported_state": all(checks.values()),
            "latency_ms": latency_ms,
            "clock_remaining_ms": visible["decision_time_remaining_ms"],
            "deadline_plan": plan,
            "composite_status": result["status"],
            "recommendation_is_none": result["recommendation"] is None,
        }
        rows.append(row)
        if (not row["joint_supported_state"] or plan["status"] != "compute") and len(failures) < 20:
            failures.append(row)
    if not rows:
        raise ValueError("real-time split has no frames")
    frame_count = len(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_kind": "held_out_synthetic_realtime_composite_deadline_audit",
        "promotion_eligible": False,
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "checkpoint_dataset_ids": sorted(checkpoint_dataset_ids),
        "audit_dataset_excluded_from_checkpoint_training": True,
        "split": split,
        "capture_sessions": sorted(sessions),
        "frames": frame_count,
        "metrics": {
            "field_accuracy": {
                field: field_passes[field] / frame_count for field in sorted(field_passes)
            },
            "joint_supported_state_accuracy": sum(
                bool(row["joint_supported_state"]) for row in rows
            )
            / frame_count,
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
            "head_latency_ms": {
                head: {
                    "p50": _percentile(values, 0.50),
                    "p95": _percentile(values, 0.95),
                    "max": max(values),
                }
                for head, values in sorted(head_latencies.items())
            },
            "perception_p95_target_ms": 250,
            "perception_p95_target_passed": _percentile(latencies, 0.95) <= 250,
            "deadline_status_counts": dict(sorted(deadline_statuses.items())),
            "strategy_tier_counts": dict(sorted(strategy_tiers.items())),
            "unsafe_or_expired_frames": sum(
                str(row["deadline_plan"]["status"]) != "compute" for row in rows
            ),
            "forced_recommendation_abstention_rate": sum(
                bool(row["recommendation_is_none"]) for row in rows
            )
            / frame_count,
        },
        "failures": failures,
        "semantic_prediction_sha256": canonical_sha256(
            [
                {
                    "sample_id": row["sample_id"],
                    "checks": row["checks"],
                    "joint_supported_state": row["joint_supported_state"],
                    "clock_remaining_ms": row["clock_remaining_ms"],
                    "composite_status": row["composite_status"],
                    "recommendation_is_none": row["recommendation_is_none"],
                }
                for row in rows
            ]
        ),
        "prediction_sha256": canonical_sha256(rows),
        "limitations": [
            "All frames and checkpoints are synthetic; this report cannot open a controlled-visible gate.",
            "The audit measures perception plus deadline routing, not solver computation or actuation.",
            "Names use the restricted synthetic OCR contract and all confidence values remain uncalibrated.",
            "Composite recommendation abstention is expected because single frames lack complete temporal history and resolved identity evidence.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit APC's integrated visible-state deadline path.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--card-checkpoint", type=Path, required=True)
    parser.add_argument("--table-state-checkpoint", type=Path, required=True)
    parser.add_argument("--stack-checkpoint", type=Path, required=True)
    parser.add_argument("--turn-clock-checkpoint", type=Path, required=True)
    parser.add_argument("--name-ocr-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--safety-margin-ms", type=int, default=750)
    parser.add_argument("--actuation-reserve-ms", type=int, default=250)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_realtime_composite(
            args.manifest,
            base_checkpoint_path=args.base_checkpoint,
            card_checkpoint_path=args.card_checkpoint,
            table_state_checkpoint_path=args.table_state_checkpoint,
            stack_checkpoint_path=args.stack_checkpoint,
            turn_clock_checkpoint_path=args.turn_clock_checkpoint,
            name_ocr_checkpoint_path=args.name_ocr_checkpoint,
            split=args.split,
            safety_margin_ms=args.safety_margin_ms,
            actuation_reserve_ms=args.actuation_reserve_ms,
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
