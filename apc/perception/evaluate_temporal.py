from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from apc.perception.baseline import (
    BaselineCheckpoint,
    _image_path,
    _manifest_annotations,
    _percentile,
)
from apc.perception.card_baseline import load_card_checkpoint
from apc.perception.event_baseline import _event_pairs, load_event_checkpoint
from apc.perception.stack_baseline import load_stack_checkpoint
from apc.perception.table_state_baseline import load_table_state_checkpoint
from apc.perception.temporal_composite import (
    effective_stacks_by_opponent,
    heads_up_effective_stack,
    infer_temporal_state,
)
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


def _cards(annotation: dict[str, Any], collection: str) -> list[str]:
    return [f"{card['rank']}{card['suit']}" for card in annotation["objects"][collection]]


def _expected_after(annotation: dict[str, Any]) -> dict[str, object]:
    seats = [
        {"seat_no": int(seat["seat_no"]), "stack_bb": str(seat["stack_bb"])}
        for seat in annotation["objects"]["seats"]
    ]
    visible: dict[str, object] = {
        "layout_id": annotation["environment"]["layout_id"],
        "theme_id": annotation["environment"]["theme_id"],
        "street": annotation["state"]["street"],
        "legal_actions": sorted(annotation["state"]["legal_actions"]),
        "hero_seat": annotation["state"]["hero_seat"],
        "dealer_seat": annotation["state"]["dealer_seat"],
        "hero_cards": _cards(annotation, "hero_cards"),
        "board_cards": _cards(annotation, "board_cards"),
        "pot_bb": str(annotation["state"]["pot_bb"]),
        "to_call_bb": str(annotation["state"]["to_call_bb"]),
        "seat_stacks_bb": seats,
        "observed_action": annotation["objects"]["observed_action"],
    }
    visible["effective_stack_bb"] = heads_up_effective_stack(visible)
    history = annotation["state"].get("action_history")
    if not isinstance(history, list):
        history = [annotation["objects"]["observed_action"]]
    folded = {
        int(row["actor_seat"])
        for row in history
        if isinstance(row, dict)
        and row.get("action") == "fold"
        and isinstance(row.get("actor_seat"), int)
    }
    visible["effective_stacks_by_opponent_bb"] = effective_stacks_by_opponent(
        visible,
        folded_seats=folded,
    )
    return visible


def _stack_values(rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list):
        return []
    return [
        {"seat_no": row.get("seat_no"), "stack_bb": row.get("stack_bb")}
        for row in rows
        if isinstance(row, dict)
    ]


def _checkpoint_sessions(checkpoint: dict[str, Any]) -> set[str]:
    training = checkpoint.get("training")
    if not isinstance(training, dict):
        return set()
    sessions = training.get("capture_sessions", [])
    return {str(value) for value in sessions}


def evaluate_temporal_composite(
    *,
    manifest_path: str | Path,
    base_checkpoint_path: str | Path,
    card_checkpoint_path: str | Path,
    table_state_checkpoint_path: str | Path,
    stack_checkpoint_path: str | Path,
    event_checkpoint_path: str | Path,
    split: str = "validation",
) -> dict[str, object]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation split must be validation or test")
    manifest_file = Path(manifest_path).expanduser().resolve()
    report = validate_manifest(manifest_file)
    if not report["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = {str(value) for value in manifest["splits"][split]}
    selected = [row for row in annotations if str(row[1]["capture_session_id"]) in eval_sessions]
    pairs = _event_pairs(selected)
    if not pairs:
        raise ValueError("held-out split has no temporal pairs")

    base = BaselineCheckpoint.load(base_checkpoint_path)
    card = load_card_checkpoint(card_checkpoint_path)
    table = load_table_state_checkpoint(table_state_checkpoint_path)
    stack = load_stack_checkpoint(stack_checkpoint_path)
    event = load_event_checkpoint(event_checkpoint_path)
    training_sessions = {
        "base": set(base.training_sessions),
        "card": _checkpoint_sessions(card),
        "table_state": _checkpoint_sessions(table),
        "stack": _checkpoint_sessions(stack),
        "event": _checkpoint_sessions(event),
    }
    overlap = {
        name: sorted(eval_sessions & sessions)
        for name, sessions in training_sessions.items()
        if eval_sessions & sessions
    }
    if overlap:
        raise ValueError(f"held-out evaluation leaks checkpoint training sessions: {overlap}")

    exact_counts = {
        "layout_theme_street": 0,
        "cards": 0,
        "hero_dealer": 0,
        "pot_and_call": 0,
        "all_stacks": 0,
        "observed_action": 0,
        "effective_stack_policy": 0,
        "opponent_effective_stacks": 0,
        "transition_accepted": 0,
        "abstention_contract": 0,
        "complete_supported_transition": 0,
    }
    latencies: list[float] = []
    errors: list[dict[str, object]] = []
    digest_rows: list[dict[str, object]] = []
    for before_annotation_path, before_annotation, after_annotation_path, after_annotation in pairs:
        before_image = _image_path(before_annotation_path, before_annotation)
        after_image = _image_path(after_annotation_path, after_annotation)
        started = time.perf_counter()
        prediction = infer_temporal_state(
            before_image,
            after_image,
            base_checkpoint=base,
            card_checkpoint=card,
            table_state_checkpoint=table,
            stack_checkpoint=stack,
            event_checkpoint=event,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        visible = prediction["visible_state"]
        expected = _expected_after(after_annotation)
        matched = {
            "layout_theme_street": all(visible[key] == expected[key] for key in ("layout_id", "theme_id", "street")),
            "cards": visible["hero_cards"] == expected["hero_cards"] and visible["board_cards"] == expected["board_cards"],
            "hero_dealer": visible["hero_seat"] == expected["hero_seat"] and visible["dealer_seat"] == expected["dealer_seat"],
            "pot_and_call": visible["pot_bb"] == expected["pot_bb"] and visible["to_call_bb"] == expected["to_call_bb"],
            "all_stacks": _stack_values(visible["seat_stacks_bb"]) == expected["seat_stacks_bb"],
            "observed_action": visible["observed_action"] == expected["observed_action"],
            "effective_stack_policy": visible["effective_stack_bb"] == expected["effective_stack_bb"],
            "opponent_effective_stacks": visible["effective_stacks_by_opponent_bb"] == expected["effective_stacks_by_opponent_bb"],
            "transition_accepted": prediction["transition_audit"]["status"] == "accepted",
            "abstention_contract": prediction["recommendation"] is None and prediction["status"] == "abstain_incomplete_state" and "player_identities" in prediction["missing_critical_fields"],
        }
        matched["complete_supported_transition"] = all(matched.values())
        for name, value in matched.items():
            exact_counts[name] += int(value)
        sample = f"{before_annotation['sample_id']}->{after_annotation['sample_id']}"
        if not matched["complete_supported_transition"]:
            errors.append(
                {
                    "pair": sample,
                    "matched": matched,
                    "expected": expected,
                    "observed": {
                        key: visible.get(key)
                        for key in expected
                    },
                    "transition_violations": prediction["transition_audit"]["violations"],
                }
            )
        digest_rows.append(
            {
                "pair": sample,
                "matched": matched,
                "observed_action": visible["observed_action"],
                "transition_status": prediction["transition_audit"]["status"],
            }
        )
    count = len(pairs)
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "held_out_temporal_composite_smoke",
        "promotion_eligible": False,
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": {},
        "checkpoint_provenance": {
            "base_sha256": base.payload["checkpoint_sha256"],
            "card_sha256": card["checkpoint_sha256"],
            "table_state_sha256": table["checkpoint_sha256"],
            "stack_sha256": stack["checkpoint_sha256"],
            "event_sha256": event["checkpoint_sha256"],
        },
        "pairs": count,
        "metrics": {
            "accuracy": {name: value / count for name, value in exact_counts.items()},
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "errors": errors[:20],
        "prediction_sha256": canonical_sha256(digest_rows),
        "limitations": [
            "Synthetic adjacent-frame closed-vocabulary evaluation only.",
            "The composite intentionally abstains because player identities and complete hand history are unresolved.",
            "Multiway scalar effective stack is intentionally null while per-opponent values remain explicit.",
            "Confidence is uncalibrated and latency includes redundant reference implementation passes."
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate APC's paired-frame composite on a held-out manifest split.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--card-checkpoint", type=Path, required=True)
    parser.add_argument("--table-state-checkpoint", type=Path, required=True)
    parser.add_argument("--stack-checkpoint", type=Path, required=True)
    parser.add_argument("--event-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_temporal_composite(
            manifest_path=args.manifest,
            base_checkpoint_path=args.base_checkpoint,
            card_checkpoint_path=args.card_checkpoint,
            table_state_checkpoint_path=args.table_state_checkpoint,
            stack_checkpoint_path=args.stack_checkpoint,
            event_checkpoint_path=args.event_checkpoint,
            split=args.split,
        )
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
