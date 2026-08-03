from __future__ import annotations

import argparse
import hashlib
import json
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from apc.perception.baseline import BaselineCheckpoint
from apc.perception.card_baseline import load_card_checkpoint
from apc.perception.composite import infer_visible_state
from apc.perception.event_baseline import load_event_checkpoint, predict_event
from apc.perception.stack_baseline import load_stack_checkpoint
from apc.perception.table_state_baseline import load_table_state_checkpoint


D = Decimal
CHIP_ACTIONS = frozenset({"call", "bet", "raise", "all_in"})
NON_CHIP_ACTIONS = frozenset({"fold", "check"})


def _temporal_perception_abstention(
    reason: str,
    before_image: str | Path,
    after_image: str | Path,
    before: dict[str, object],
    after: dict[str, object],
    *,
    started: float,
    detail: object,
) -> dict[str, object]:
    before_path = Path(before_image).expanduser().resolve()
    after_path = Path(after_image).expanduser().resolve()
    missing = {
        str(value)
        for result in (before, after)
        for value in result.get("missing_critical_fields", [])
    }
    missing.add("temporally_consistent_state")
    checkpoint_provenance = after.get("checkpoint_provenance")
    if not isinstance(checkpoint_provenance, dict):
        checkpoint_provenance = before.get("checkpoint_provenance", {})
    confidence_rows = [
        float(result.get("minimum_supported_confidence", 0.0))
        for result in (before, after)
        if isinstance(result.get("minimum_supported_confidence"), (int, float))
    ]
    return {
        "schema_version": "1.0.0",
        "model_name": "APC",
        "units": "BB",
        "status": f"abstain_{reason}",
        "recommendation": None,
        "frames": {
            "before": {
                "image_path": str(before_path),
                "image_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
            },
            "after": {
                "image_path": str(after_path),
                "image_sha256": hashlib.sha256(after_path.read_bytes()).hexdigest(),
            },
        },
        "checkpoint_provenance": dict(checkpoint_provenance),
        "previous_visible_state": before.get("visible_state"),
        "visible_state": after.get("visible_state"),
        "event_confidence": {},
        "minimum_supported_confidence": min(confidence_rows, default=0.0),
        "transition_audit": {
            "status": "rejected",
            "checks": [],
            "violations": [
                {
                    "code": reason,
                    "passed": False,
                    "expected": "all prerequisite perception heads accepted",
                    "observed": detail,
                }
            ],
            "deltas": {"pot_bb": None, "changed_seats": []},
        },
        "missing_critical_fields": sorted(missing),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "limitations": [
            "A prerequisite perception head abstained, so no event or temporal state was inferred.",
            "No recommendation can be emitted from a rejected temporal prerequisite.",
        ],
    }


def _decimal(value: object, name: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a BB number")
    try:
        result = D(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a BB number") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be a non-negative finite BB number")
    return result


def _stack_map(state: dict[str, object]) -> dict[int, Decimal]:
    rows = state.get("seat_stacks_bb")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("visible state must contain at least two seat stacks")
    result: dict[int, Decimal] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("seat stack rows must be objects")
        seat = row.get("seat_no")
        if isinstance(seat, bool) or not isinstance(seat, int) or seat < 1:
            raise ValueError("seat_no must be a positive integer")
        if seat in result:
            raise ValueError(f"duplicate stack for seat {seat}")
        result[seat] = _decimal(row.get("stack_bb"), f"seat {seat} stack_bb")
    return result


def heads_up_effective_stack(state: dict[str, object]) -> str | None:
    """Return the unambiguous current effective stack only for a two-seat state."""
    stacks = _stack_map(state)
    if len(stacks) != 2:
        return None
    value = min(stacks.values())
    return format(value, "f")


def effective_stacks_by_opponent(
    state: dict[str, object],
    *,
    folded_seats: Iterable[int] = (),
) -> list[dict[str, object]]:
    """Return explicit Hero-versus-opponent effective stacks for active visible seats."""
    stacks = _stack_map(state)
    hero = state.get("hero_seat")
    if isinstance(hero, bool) or not isinstance(hero, int) or hero not in stacks:
        raise ValueError("visible state hero_seat must identify a stacked seat")
    folded = {int(seat) for seat in folded_seats}
    return [
        {
            "opponent_seat": seat,
            "effective_stack_bb": format(min(stacks[hero], stack), "f"),
        }
        for seat, stack in sorted(stacks.items())
        if seat != hero and seat not in folded
    ]


def validate_visible_transition(
    before: dict[str, object],
    after: dict[str, object],
    event: dict[str, object],
) -> dict[str, object]:
    """Audit one adjacent visible-state transition without inventing hidden state."""
    checks: list[dict[str, object]] = []

    def check(code: str, passed: bool, expected: object, observed: object) -> None:
        checks.append(
            {
                "code": code,
                "passed": bool(passed),
                "expected": expected,
                "observed": observed,
            }
        )

    for field in (
        "layout_id",
        "theme_id",
        "street",
        "hero_seat",
        "dealer_seat",
        "hero_cards",
        "board_cards",
    ):
        check(
            f"{field}_immutable",
            before.get(field) == after.get(field),
            before.get(field),
            after.get(field),
        )

    before_stacks = _stack_map(before)
    after_stacks = _stack_map(after)
    check(
        "seat_set_immutable",
        set(before_stacks) == set(after_stacks),
        sorted(before_stacks),
        sorted(after_stacks),
    )
    actor = event.get("actor_seat")
    if isinstance(actor, bool) or not isinstance(actor, int):
        raise ValueError("event.actor_seat must be an integer")
    check("actor_seat_visible", actor in before_stacks and actor in after_stacks, True, actor)

    action = event.get("action")
    if action not in CHIP_ACTIONS | NON_CHIP_ACTIONS:
        raise ValueError(f"unsupported observed action: {action}")
    before_pot = _decimal(before.get("pot_bb"), "before.pot_bb")
    after_pot = _decimal(after.get("pot_bb"), "after.pot_bb")
    pot_delta = after_pot - before_pot
    check("pot_nondecreasing", pot_delta >= 0, ">= 0", format(pot_delta, "f"))

    changed_seats = sorted(
        seat
        for seat in set(before_stacks) & set(after_stacks)
        if before_stacks[seat] != after_stacks[seat]
    )
    if action in NON_CHIP_ACTIONS:
        check("non_chip_action_has_no_stack_delta", not changed_seats, [], changed_seats)
        check("non_chip_action_has_no_pot_delta", pot_delta == 0, "0", format(pot_delta, "f"))
    else:
        amount = _decimal(event.get("amount_bb"), "event.amount_bb")
        actor_delta = before_stacks.get(actor, D("0")) - after_stacks.get(actor, D("0"))
        check("only_actor_stack_changes", changed_seats == [actor], [actor], changed_seats)
        check("actor_stack_delta_matches_amount", actor_delta == amount, format(amount, "f"), format(actor_delta, "f"))
        check("pot_delta_matches_amount", pot_delta == amount, format(amount, "f"), format(pot_delta, "f"))
        if action == "all_in":
            check("all_in_exhausts_actor_stack", after_stacks.get(actor) == 0, "0", format(after_stacks.get(actor, D("-1")), "f"))

    violations = [row for row in checks if not row["passed"]]
    return {
        "status": "accepted" if not violations else "rejected",
        "checks": checks,
        "violations": violations,
        "deltas": {
            "pot_bb": format(pot_delta, "f"),
            "changed_seats": changed_seats,
        },
    }


def _history_rows(prior: Iterable[dict[str, object]], event: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(prior):
        if not isinstance(row, dict):
            raise ValueError(f"prior action history row {index} must be an object")
        rows.append(dict(row))
    if not rows or rows[-1] != event:
        rows.append(dict(event))
    return rows


def infer_temporal_state(
    before_image: str | Path,
    after_image: str | Path,
    *,
    base_checkpoint: BaselineCheckpoint,
    card_checkpoint: dict[str, Any],
    table_state_checkpoint: dict[str, Any],
    stack_checkpoint: dict[str, Any],
    event_checkpoint: dict[str, Any],
    prior_action_history: Iterable[dict[str, object]] = (),
    history_complete: bool = False,
) -> dict[str, object]:
    started = time.perf_counter()
    before = infer_visible_state(
        before_image,
        base_checkpoint=base_checkpoint,
        card_checkpoint=card_checkpoint,
        table_state_checkpoint=table_state_checkpoint,
        stack_checkpoint=stack_checkpoint,
    )
    after = infer_visible_state(
        after_image,
        base_checkpoint=base_checkpoint,
        card_checkpoint=card_checkpoint,
        table_state_checkpoint=table_state_checkpoint,
        stack_checkpoint=stack_checkpoint,
    )
    accepted_frame_status = "abstain_incomplete_state"
    if before.get("status") != accepted_frame_status or after.get("status") != accepted_frame_status:
        return _temporal_perception_abstention(
            "frame_perception_rejected",
            before_image,
            after_image,
            before,
            after,
            started=started,
            detail={"before": before.get("status"), "after": after.get("status")},
        )
    try:
        event_result = predict_event(
            event_checkpoint,
            before_image,
            after_image,
            base_checkpoint=base_checkpoint,
            stack_checkpoint=stack_checkpoint,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        return _temporal_perception_abstention(
            "event_perception_failed",
            before_image,
            after_image,
            before,
            after,
            started=started,
            detail=str(error),
        )
    event = dict(event_result["event"])
    before_state = dict(before["visible_state"])
    after_state = dict(after["visible_state"])
    transition = validate_visible_transition(before_state, after_state, event)
    history = _history_rows(prior_action_history, event)
    effective_stack = heads_up_effective_stack(after_state)
    folded_seats = {
        int(row["actor_seat"])
        for row in history
        if row.get("action") == "fold" and isinstance(row.get("actor_seat"), int)
    }
    opponent_effective_stacks = effective_stacks_by_opponent(
        after_state,
        folded_seats=folded_seats,
    )

    missing = {"player_identities"}
    if effective_stack is None:
        missing.add("effective_stack_bb")
    if not history_complete:
        missing.add("action_history")
    if transition["status"] != "accepted":
        missing.add("temporally_consistent_state")

    after_state["effective_stack_bb"] = effective_stack
    after_state["effective_stacks_by_opponent_bb"] = opponent_effective_stacks
    after_state["observed_action"] = event
    after_state["action_history"] = history
    after_state["history_complete"] = history_complete
    after_state["seat_aliases"] = [
        {
            "seat_no": row["seat_no"],
            "temporary_alias": f"Seat{row['seat_no']}",
            "identity_status": "unresolved",
        }
        for row in after_state["seat_stacks_bb"]
    ]

    frame_confidence = min(
        float(before["minimum_supported_confidence"]),
        float(after["minimum_supported_confidence"]),
        float(event_result["confidence"]),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    status = "abstain_transition_rejected" if transition["status"] != "accepted" else "abstain_incomplete_state"
    return {
        "schema_version": "1.0.0",
        "model_name": "APC",
        "units": "BB",
        "status": status,
        "recommendation": None,
        "frames": {
            "before": {
                "image_path": str(Path(before_image).expanduser().resolve()),
                "image_sha256": hashlib.sha256(Path(before_image).expanduser().resolve().read_bytes()).hexdigest(),
            },
            "after": {
                "image_path": str(Path(after_image).expanduser().resolve()),
                "image_sha256": hashlib.sha256(Path(after_image).expanduser().resolve().read_bytes()).hexdigest(),
            },
        },
        "checkpoint_provenance": {
            **dict(after["checkpoint_provenance"]),
            "event_sha256": event_checkpoint["checkpoint_sha256"],
        },
        "previous_visible_state": before_state,
        "visible_state": after_state,
        "event_confidence": event_result["field_confidence"],
        "minimum_supported_confidence": frame_confidence,
        "transition_audit": transition,
        "missing_critical_fields": sorted(missing),
        "latency_ms": elapsed_ms,
        "limitations": [
            "Synthetic closed-vocabulary checkpoints only.",
            "A pair-local action history is not treated as complete unless history_complete is explicitly supplied with prior evidence.",
            "A scalar effective stack is derived only for heads-up; multiway states preserve one Hero-versus-opponent value per active visible opponent.",
            "Seat aliases are temporary and cannot be used as persistent player identities.",
            "Confidence values are uncalibrated and no recommendation is emitted.",
        ],
    }


def _read_history(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("history file must contain a JSON array")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run APC's paired-frame temporal perception smoke pipeline.")
    parser.add_argument("before_image", type=Path)
    parser.add_argument("after_image", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--card-checkpoint", type=Path, required=True)
    parser.add_argument("--table-state-checkpoint", type=Path, required=True)
    parser.add_argument("--stack-checkpoint", type=Path, required=True)
    parser.add_argument("--event-checkpoint", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--history-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = infer_temporal_state(
            args.before_image,
            args.after_image,
            base_checkpoint=BaselineCheckpoint.load(args.base_checkpoint),
            card_checkpoint=load_card_checkpoint(args.card_checkpoint),
            table_state_checkpoint=load_table_state_checkpoint(args.table_state_checkpoint),
            stack_checkpoint=load_stack_checkpoint(args.stack_checkpoint),
            event_checkpoint=load_event_checkpoint(args.event_checkpoint),
            prior_action_history=_read_history(args.history),
            history_complete=args.history_complete,
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
