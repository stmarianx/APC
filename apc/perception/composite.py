from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from apc.perception.baseline import BaselineCheckpoint, predict_image
from apc.perception.card_baseline import load_card_checkpoint, predict_cards
from apc.perception.table_state_baseline import (
    load_table_state_checkpoint,
    predict_table_state,
)
from apc.perception.stack_baseline import load_stack_checkpoint, predict_stacks
from apc.perception.name_ocr_baseline import (
    load_name_ocr_checkpoint,
    predict_player_names,
)
from apc.perception.turn_clock_baseline import (
    load_turn_clock_checkpoint,
    predict_turn_clock,
)
from apc.visual_identity_signature import extract_frame_signatures


MISSING_CRITICAL_FIELDS = (
    "effective_stack_bb",
    "player_identities",
    "action_history",
    "observed_action",
)


def _safe_perception_head(
    name: str,
    operation: Callable[[], Any],
) -> tuple[Any | None, dict[str, str] | None]:
    try:
        return operation(), None
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        return None, {
            "field": name,
            "reason": "perception_head_failed",
            "detail": str(error),
        }


def _visual_signatures_or_abstain(
    image_path: str | Path,
    stacks: list[dict[str, object]],
) -> tuple[list[object], str | None]:
    try:
        return (
            extract_frame_signatures(
                image_path,
                [
                    {"seat_no": stack["seat_no"], "box": stack["seat_box"]}
                    for stack in stacks
                ],
            ),
            None,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        return [], str(error)


def _card_token(card: dict[str, object]) -> str:
    return f"{card['rank']}{card['suit']}"


def _visible_card_integrity(
    hero_cards: list[str],
    board_cards: list[str],
    street: str,
) -> dict[str, object]:
    expected_board = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}.get(street)
    all_cards = hero_cards + board_cards
    checks = {
        "hero_card_count": len(hero_cards) == 2,
        "board_card_count": expected_board is not None and len(board_cards) == expected_board,
        "cards_unique": len(set(all_cards)) == len(all_cards),
    }
    return {
        "status": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "hero_cards": list(hero_cards),
        "board_cards": list(board_cards),
        "street": street,
    }


def infer_visible_state(
    image_path: str | Path,
    *,
    base_checkpoint: BaselineCheckpoint,
    card_checkpoint: dict[str, Any],
    table_state_checkpoint: dict[str, Any],
    stack_checkpoint: dict[str, Any],
    turn_clock_checkpoint: dict[str, Any] | None = None,
    name_ocr_checkpoint: dict[str, Any] | None = None,
) -> dict[str, object]:
    path = Path(image_path).expanduser().resolve()
    started = time.perf_counter()
    base, base_error = _safe_perception_head(
        "base_perception",
        lambda: predict_image(base_checkpoint, path),
    )
    cards, card_error = _safe_perception_head(
        "card_perception",
        lambda: predict_cards(
            card_checkpoint,
            base_checkpoint,
            path,
            base_prediction=base,
        ),
    ) if base is not None else (None, {
        "field": "card_perception",
        "reason": "dependency_unavailable",
        "detail": "base_perception failed",
    })
    table, table_error = _safe_perception_head(
        "table_state_perception",
        lambda: predict_table_state(
            table_state_checkpoint,
            base_checkpoint,
            path,
            base_prediction=base,
        ),
    ) if base is not None else (None, {
        "field": "table_state_perception",
        "reason": "dependency_unavailable",
        "detail": "base_perception failed",
    })
    stacks, stack_error = _safe_perception_head(
        "stack_perception",
        lambda: predict_stacks(
            stack_checkpoint,
            base_checkpoint,
            path,
            base_prediction=base,
        ),
    ) if base is not None else (None, {
        "field": "stack_perception",
        "reason": "dependency_unavailable",
        "detail": "base_perception failed",
    })
    clock, clock_error = (
        _safe_perception_head(
            "turn_clock_perception",
            lambda: predict_turn_clock(turn_clock_checkpoint, path),
        )
        if turn_clock_checkpoint is not None
        else (None, None)
    )
    names, names_error = (
        _safe_perception_head(
            "player_name_perception",
            lambda: predict_player_names(
                name_ocr_checkpoint,
                path,
                base_checkpoint=base_checkpoint,
                base_prediction=base,
            ),
        )
        if name_ocr_checkpoint is not None and base is not None
        else (
            (None, None)
            if name_ocr_checkpoint is None
            else (
                None,
                {
                    "field": "player_name_perception",
                    "reason": "dependency_unavailable",
                    "detail": "base_perception failed",
                },
            )
        )
    )
    base = base or {}
    cards = cards or {"hero_cards": [], "board_cards": []}
    table = table or {
        "hero_seat": None,
        "dealer_seat": None,
        "pot_bb": {"value": None, "confidence": 0.0},
        "to_call_bb": {"value": None, "confidence": 0.0},
    }
    stacks = stacks or []
    signature_rows, signature_error = (
        _visual_signatures_or_abstain(path, stacks)
        if stacks
        else ([], "stack geometry unavailable")
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    hero_cards = [_card_token(card) for card in cards["hero_cards"]]
    board_cards = [_card_token(card) for card in cards["board_cards"]]
    card_integrity = _visible_card_integrity(
        hero_cards,
        board_cards,
        str(base.get("street", {}).get("value")),
    )
    head_errors = [
        error
        for error in (
            base_error,
            card_error,
            table_error,
            stack_error,
            clock_error,
            names_error,
        )
        if error is not None
    ]
    perception_abstentions = list(head_errors)
    if signature_error is not None:
        perception_abstentions.append(
            {
                "field": "visual_identity_signatures",
                "reason": "name_band_signature_unavailable",
                "detail": signature_error,
            }
        )
    if card_integrity["status"] != "accepted":
        perception_abstentions.append(
            {
                "field": "visible_cards",
                "reason": "card_integrity_rejected",
                "detail": card_integrity["checks"],
            }
        )
    missing_critical_fields = list(MISSING_CRITICAL_FIELDS)
    missing_critical_fields.extend(str(error["field"]) for error in head_errors)
    if card_integrity["status"] != "accepted":
        missing_critical_fields.append("visible_card_integrity")
    field_confidence = {
        "layout_id": float(base.get("layout_id", {}).get("confidence", 0.0)),
        "theme_id": float(base.get("theme_id", {}).get("confidence", 0.0)),
        "street": float(base.get("street", {}).get("confidence", 0.0)),
        "legal_actions": float(base.get("legal_actions", {}).get("confidence", 0.0)),
        "hero_cards": min((float(card["confidence"]) for card in cards["hero_cards"]), default=0.0),
        "board_cards": min((float(card["confidence"]) for card in cards["board_cards"]), default=1.0),
        "pot_bb": float(table["pot_bb"]["confidence"]),
        "to_call_bb": float(table["to_call_bb"]["confidence"]),
        "seat_stacks_bb": min((float(stack["confidence"]) for stack in stacks), default=0.0),
        "visual_identity_signatures": min(
            (float(row.quality_score) for row in signature_rows),
            default=0.0,
        ),
    }
    if clock is not None:
        field_confidence["decision_time_remaining_ms"] = float(clock["confidence"])
    if names is not None:
        field_confidence["recognized_player_names"] = min(
            (float(row["confidence"]) for row in names["player_names"]),
            default=0.0,
        )
    visible_state = {
        "layout_id": base.get("layout_id", {}).get("value"),
        "theme_id": base.get("theme_id", {}).get("value"),
        "street": base.get("street", {}).get("value"),
        "legal_actions": (
            str(base["legal_actions"]["value"]).split("+")
            if base.get("legal_actions", {}).get("value") is not None
            else []
        ),
        "hero_seat": table["hero_seat"],
        "dealer_seat": table["dealer_seat"],
        "hero_cards": hero_cards,
        "board_cards": board_cards,
        "pot_bb": table["pot_bb"]["value"],
        "to_call_bb": table["to_call_bb"]["value"],
        "seat_stacks_bb": [
            {
                "seat_no": stack["seat_no"],
                "stack_bb": stack["stack_bb"],
                "seat_box": stack["seat_box"],
            }
            for stack in stacks
        ],
        "visual_identity_signatures": [
            {
                "seat_no": row.seat_no,
                "visual_token": row.visual_token,
                "signature_sha256": row.signature_sha256,
                "quality_score": row.quality_score,
                "frame_sha256": row.frame_sha256,
            }
            for row in signature_rows
        ],
    }
    if clock is not None:
        visible_state.update(
            {
                "hero_to_act": bool(visible_state["legal_actions"]),
                "decision_time_remaining_ms": clock["remaining_ms"],
                "decision_deadline_source": "visible_timer",
                "turn_clock_box": clock["clock_box"],
            }
        )
    if names is not None:
        visible_state["recognized_player_names"] = names["player_names"]
    checkpoint_provenance = {
        "base_sha256": base_checkpoint.payload["checkpoint_sha256"],
        "card_sha256": card_checkpoint["checkpoint_sha256"],
        "table_state_sha256": table_state_checkpoint["checkpoint_sha256"],
        "stack_sha256": stack_checkpoint["checkpoint_sha256"],
    }
    if turn_clock_checkpoint is not None:
        checkpoint_provenance["turn_clock_sha256"] = turn_clock_checkpoint[
            "checkpoint_sha256"
        ]
    if name_ocr_checkpoint is not None:
        checkpoint_provenance["name_ocr_sha256"] = name_ocr_checkpoint[
            "checkpoint_sha256"
        ]
    return {
        "schema_version": "1.0.0",
        "model_name": "APC",
        "units": "BB",
        "status": (
            "abstain_perception_head_failure"
            if head_errors
            else "abstain_invalid_visible_state"
            if card_integrity["status"] != "accepted"
            else "abstain_incomplete_state"
        ),
        "recommendation": None,
        "frame": {
            "image_path": str(path),
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "checkpoint_provenance": checkpoint_provenance,
        "visible_state": visible_state,
        "field_confidence": field_confidence,
        "card_integrity_audit": card_integrity,
        "minimum_supported_confidence": min(field_confidence.values()),
        "missing_critical_fields": missing_critical_fields,
        "perception_abstentions": perception_abstentions,
        "latency_ms": elapsed_ms,
        "limitations": [
            "Synthetic closed-vocabulary checkpoints only.",
            "Recommendation is forced to abstain until effective-stack context, player identities and action history are observed.",
            "Confidence values are uncalibrated and cannot open the production confidence gate.",
            "Optional visual identity extraction abstains instead of terminating the remaining perception heads.",
            "Card, numeric and stack heads fail independently and force an audited composite abstention.",
            "The optional synthetic name head is restricted to its declared fixed-length character contract.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run APC's composite visible-state smoke pipeline.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--card-checkpoint", type=Path, required=True)
    parser.add_argument("--table-state-checkpoint", type=Path, required=True)
    parser.add_argument("--stack-checkpoint", type=Path, required=True)
    parser.add_argument("--turn-clock-checkpoint", type=Path)
    parser.add_argument("--name-ocr-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = infer_visible_state(
            args.image,
            base_checkpoint=BaselineCheckpoint.load(args.base_checkpoint),
            card_checkpoint=load_card_checkpoint(args.card_checkpoint),
            table_state_checkpoint=load_table_state_checkpoint(args.table_state_checkpoint),
            stack_checkpoint=load_stack_checkpoint(args.stack_checkpoint),
            turn_clock_checkpoint=(
                load_turn_clock_checkpoint(args.turn_clock_checkpoint)
                if args.turn_clock_checkpoint
                else None
            ),
            name_ocr_checkpoint=(
                load_name_ocr_checkpoint(args.name_ocr_checkpoint)
                if args.name_ocr_checkpoint
                else None
            ),
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
