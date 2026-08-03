from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from apc.perception.baseline import BaselineCheckpoint, _image_path, _manifest_annotations, _percentile
from apc.perception.boundary_baseline import load_boundary_checkpoint, predict_boundary
from apc.perception.card_baseline import load_card_checkpoint
from apc.perception.event_baseline import load_event_checkpoint
from apc.perception.hand_tracker import TemporalHandTracker
from apc.perception.stack_baseline import load_stack_checkpoint
from apc.perception.table_state_baseline import load_table_state_checkpoint
from apc.perception.temporal_composite import effective_stacks_by_opponent, infer_temporal_state
from apc.tools.validate_dataset import canonical_sha256, validate_manifest
from apc.tools.validate_sequences import audit_sequence_manifest


def _card_tokens(annotation: dict[str, Any], collection: str) -> list[str]:
    return [f"{card['rank']}{card['suit']}" for card in annotation["objects"][collection]]


def _expected_stacks(annotation: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {"seat_no": int(seat["seat_no"]), "stack_bb": str(seat["stack_bb"])}
        for seat in annotation["objects"]["seats"]
    ]


def _observed_stacks(state: dict[str, object]) -> list[dict[str, object]]:
    rows = state.get("seat_stacks_bb")
    if not isinstance(rows, list):
        return []
    return [
        {"seat_no": row.get("seat_no"), "stack_bb": row.get("stack_bb")}
        for row in rows
        if isinstance(row, dict)
    ]


def _expected_opponent_stacks(annotation: dict[str, Any]) -> list[dict[str, object]]:
    state = {
        "hero_seat": int(annotation["state"]["hero_seat"]),
        "seat_stacks_bb": _expected_stacks(annotation),
    }
    folded = {
        int(row["actor_seat"])
        for row in annotation["state"]["action_history"]
        if row.get("action") == "fold"
    }
    return effective_stacks_by_opponent(state, folded_seats=folded)


def evaluate_hand_tracker(
    *,
    manifest_path: str | Path,
    base_checkpoint_path: str | Path,
    card_checkpoint_path: str | Path,
    table_state_checkpoint_path: str | Path,
    stack_checkpoint_path: str | Path,
    event_checkpoint_path: str | Path,
    boundary_checkpoint_path: str | Path,
    split: str = "validation",
    boundary_threshold: float = 0.20,
) -> dict[str, object]:
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation split must be validation or test")
    manifest_file = Path(manifest_path).expanduser().resolve()
    frame_report = validate_manifest(manifest_file)
    sequence_report = audit_sequence_manifest(manifest_file)
    if not frame_report["valid"] or not sequence_report["valid"]:
        raise ValueError("complete-hand dataset validation failed")
    manifest, annotations = _manifest_annotations(manifest_file)
    eval_sessions = {str(value) for value in manifest["splits"][split]}
    sessions: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for row in annotations:
        if str(row[1]["capture_session_id"]) in eval_sessions:
            sessions[str(row[1]["capture_session_id"])].append(row)

    base = BaselineCheckpoint.load(base_checkpoint_path)
    card = load_card_checkpoint(card_checkpoint_path)
    table = load_table_state_checkpoint(table_state_checkpoint_path)
    stack = load_stack_checkpoint(stack_checkpoint_path)
    event = load_event_checkpoint(event_checkpoint_path)
    boundary = load_boundary_checkpoint(boundary_checkpoint_path)
    training_sessions = {
        "base": set(base.training_sessions),
        "card": {str(value) for value in card["training"]["capture_sessions"]},
        "table": {str(value) for value in table["training"]["capture_sessions"]},
        "stack": {str(value) for value in stack["training"]["capture_sessions"]},
        "event": {str(value) for value in event["training"]["capture_sessions"]},
        "boundary": {str(value) for value in boundary["training"]["capture_sessions"]},
    }
    overlap = {
        name: sorted(eval_sessions & rows)
        for name, rows in training_sessions.items()
        if eval_sessions & rows
    }
    if overlap:
        raise ValueError(f"held-out hand tracker evaluation leaks training sessions: {overlap}")

    tracker = TemporalHandTracker(minimum_boundary_confidence=boundary_threshold)
    metrics = {
        "pair_state_accepted": 0,
        "history_exact": 0,
        "visible_state_exact": 0,
        "opponent_effective_stacks_exact": 0,
        "revision_exact": 0,
        "hand_id_stable": 0,
        "abstention_contract": 0,
        "complete_tracked_pair": 0,
    }
    boundary_confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    pairs = 0
    boundary_transitions = 0
    completed_hands = 0
    exact_completed_hands = 0
    latencies: list[float] = []
    errors: list[dict[str, object]] = []
    digest_rows: list[dict[str, object]] = []
    for session, raw_rows in sorted(sessions.items()):
        rows = sorted(raw_rows, key=lambda row: int(row[1]["sequence_index"]))
        if len(rows) % 2:
            raise ValueError(f"session {session} has an odd number of frames")
        expected_to_generated: dict[str, str] = {}
        generated_ids: set[str] = set()
        previous_after_image: Path | None = None
        for pair_index in range(0, len(rows), 2):
            before_path, before_annotation = rows[pair_index]
            after_path, after_annotation = rows[pair_index + 1]
            before_image = _image_path(before_path, before_annotation)
            after_image = _image_path(after_path, after_annotation)
            expected_start = bool(before_annotation["state"].get("hand_start"))
            if previous_after_image is None:
                predicted_start = True
                boundary_confidence = 1.0
                boundary_source = "capture_session_start"
            else:
                boundary_prediction = predict_boundary(boundary, previous_after_image, before_image)
                predicted_start = bool(boundary_prediction["hand_start"])
                boundary_confidence = float(boundary_prediction["confidence"])
                boundary_source = "temporal_boundary_model"
                key = "tp" if expected_start and predicted_start else "fn" if expected_start else "fp" if predicted_start else "tn"
                boundary_confusion[key] += 1
                boundary_transitions += 1
            prior_history = [] if predicted_start else tracker.prior_history(session)
            started = time.perf_counter()
            temporal = infer_temporal_state(
                before_image,
                after_image,
                base_checkpoint=base,
                card_checkpoint=card,
                table_state_checkpoint=table,
                stack_checkpoint=stack,
                event_checkpoint=event,
                prior_action_history=prior_history,
                history_complete=True,
            )
            tracked = tracker.submit(
                session,
                temporal,
                hand_start=predicted_start,
                boundary_confidence=boundary_confidence,
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            pairs += 1
            state = tracked.get("state")
            expected_hand_id = str(after_annotation["state"]["hand_id"])
            if isinstance(state, dict) and isinstance(tracked.get("hand_id"), str):
                generated = str(tracked["hand_id"])
                if expected_hand_id not in expected_to_generated:
                    expected_to_generated[expected_hand_id] = generated
                    generated_ids.add(generated)
                hand_stable = expected_to_generated[expected_hand_id] == generated and len(generated_ids) == len(expected_to_generated)
                visible_exact = (
                    state.get("hero_cards") == _card_tokens(after_annotation, "hero_cards")
                    and state.get("board_cards") == _card_tokens(after_annotation, "board_cards")
                    and state.get("pot_bb") == str(after_annotation["state"]["pot_bb"])
                    and state.get("to_call_bb") == str(after_annotation["state"]["to_call_bb"])
                    and _observed_stacks(state) == _expected_stacks(after_annotation)
                    and state.get("observed_action") == after_annotation["objects"]["observed_action"]
                )
                opponent_effective_exact = (
                    state.get("effective_stacks_by_opponent_bb")
                    == _expected_opponent_stacks(after_annotation)
                )
                history_exact = state.get("action_history") == after_annotation["state"]["action_history"]
                revision_exact = state.get("revision") == len(after_annotation["state"]["action_history"]) - 1
            else:
                hand_stable = visible_exact = history_exact = revision_exact = False
                opponent_effective_exact = False
            matched = {
                "pair_state_accepted": tracked.get("status") == "state_tracked_incomplete_identity",
                "history_exact": history_exact,
                "visible_state_exact": visible_exact,
                "opponent_effective_stacks_exact": opponent_effective_exact,
                "revision_exact": revision_exact,
                "hand_id_stable": hand_stable,
                "abstention_contract": tracked.get("recommendation") is None and "player_identities" in tracked.get("missing_critical_fields", []),
            }
            matched["complete_tracked_pair"] = all(matched.values())
            for name, value in matched.items():
                metrics[name] += int(value)
            if after_annotation["state"]["street"] == "river":
                completed_hands += 1
                exact_completed_hands += int(matched["complete_tracked_pair"] and isinstance(state, dict) and len(state["action_history"]) == 4)
            pair_id = f"{before_annotation['sample_id']}->{after_annotation['sample_id']}"
            if not matched["complete_tracked_pair"]:
                errors.append(
                    {
                        "pair": pair_id,
                        "boundary_source": boundary_source,
                        "expected_hand_start": expected_start,
                        "predicted_hand_start": predicted_start,
                        "boundary_confidence": boundary_confidence,
                        "tracked_status": tracked.get("status"),
                        "matched": matched,
                    }
                )
            digest_rows.append(
                {
                    "pair": pair_id,
                    "predicted_hand_start": predicted_start,
                    "boundary_confidence": boundary_confidence,
                    "tracked_status": tracked.get("status"),
                    "generated_hand_id": tracked.get("hand_id"),
                    "matched": matched,
                }
            )
            previous_after_image = after_image
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "held_out_complete_hand_tracker_smoke",
        "promotion_eligible": False,
        "dataset_id": manifest["dataset_id"],
        "dataset_fingerprints": manifest["fingerprints"],
        "sequence_audit_sha256": sequence_report["sequence_audit_sha256"],
        "split": split,
        "capture_sessions": sorted(eval_sessions),
        "training_session_overlap": {},
        "boundary_threshold": boundary_threshold,
        "checkpoint_provenance": {
            "base_sha256": base.payload["checkpoint_sha256"],
            "card_sha256": card["checkpoint_sha256"],
            "table_state_sha256": table["checkpoint_sha256"],
            "stack_sha256": stack["checkpoint_sha256"],
            "event_sha256": event["checkpoint_sha256"],
            "boundary_sha256": boundary["checkpoint_sha256"],
        },
        "pairs": pairs,
        "completed_hands": completed_hands,
        "metrics": {
            "accuracy": {name: value / pairs for name, value in metrics.items()},
            "completed_hand_exact_accuracy": exact_completed_hands / completed_hands,
            "boundary": {
                "transitions": boundary_transitions,
                "confusion": boundary_confusion,
                "accuracy": (boundary_confusion["tp"] + boundary_confusion["tn"]) / boundary_transitions,
            },
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            },
        },
        "errors": errors[:20],
        "prediction_sha256": canonical_sha256(digest_rows),
        "limitations": [
            "Synthetic two-hand held-out session only; this is not promotion evidence.",
            "The synthetic boundary threshold is selected on development validation and is not calibrated.",
            "Player identities remain unresolved by perception, so the tracker always abstains from coaching.",
            "Latency includes redundant image decoding and feature extraction in a reference implementation.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate APC's complete-hand temporal tracker.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--card-checkpoint", type=Path, required=True)
    parser.add_argument("--table-state-checkpoint", type=Path, required=True)
    parser.add_argument("--stack-checkpoint", type=Path, required=True)
    parser.add_argument("--event-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--boundary-threshold", type=float, default=0.20)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_hand_tracker(
            manifest_path=args.manifest,
            base_checkpoint_path=args.base_checkpoint,
            card_checkpoint_path=args.card_checkpoint,
            table_state_checkpoint_path=args.table_state_checkpoint,
            stack_checkpoint_path=args.stack_checkpoint,
            event_checkpoint_path=args.event_checkpoint,
            boundary_checkpoint_path=args.boundary_checkpoint,
            split=args.split,
            boundary_threshold=args.boundary_threshold,
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
