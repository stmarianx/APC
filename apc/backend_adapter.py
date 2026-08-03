from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any


POSITION_LABELS = {
    2: ("BTN", "BB"),
    3: ("BTN", "SB", "BB"),
    4: ("BTN", "SB", "BB", "UTG"),
    5: ("BTN", "SB", "BB", "UTG", "CO"),
    6: ("BTN", "SB", "BB", "UTG", "HJ", "CO"),
    7: ("BTN", "SB", "BB", "UTG", "MP", "HJ", "CO"),
    8: ("BTN", "SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO"),
    9: ("BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"),
}


def _bb(value: object, name: str) -> str:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a BB number") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be a non-negative finite BB number")
    return format(result, "f")


def seat_positions(seat_numbers: list[int], dealer_seat: int) -> dict[int, str]:
    ordered = sorted(seat_numbers)
    if len(ordered) != len(set(ordered)) or dealer_seat not in ordered:
        raise ValueError("occupied seat numbers must be unique and include the dealer")
    labels = POSITION_LABELS.get(len(ordered))
    if labels is None:
        raise ValueError("solver position mapping supports two to nine occupied seats")
    dealer_index = ordered.index(dealer_seat)
    clockwise = ordered[dealer_index:] + ordered[:dealer_index]
    return {seat: labels[index] for index, seat in enumerate(clockwise)}


@dataclass(frozen=True)
class BackendAdapterConfig:
    provider_id: str = "apc_visible_table"
    provider_version: str = "0.1.0"
    calibration_profile_id: str = "apc-platform-neutral-v1"
    game: str = "holdem_no_limit"
    rake_model: str = "training_no_rake"
    utility_model: str = "chip_ev"
    raise_amount_semantics: str = "to_amount_only"
    multiway_effective_stack_policy: str = "abstain"
    calibration_regions: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "table": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
        }
    )

    def __post_init__(self) -> None:
        for value, name in (
            (self.provider_id, "provider_id"),
            (self.provider_version, "provider_version"),
            (self.calibration_profile_id, "calibration_profile_id"),
            (self.game, "game"),
            (self.rake_model, "rake_model"),
            (self.utility_model, "utility_model"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.raise_amount_semantics not in {"to_amount_only", "amount_is_to"}:
            raise ValueError("raise_amount_semantics must be to_amount_only or amount_is_to")
        if self.multiway_effective_stack_policy not in {
            "abstain",
            "minimum_active_opponent",
        }:
            raise ValueError(
                "multiway_effective_stack_policy must be abstain or minimum_active_opponent"
            )
        if not self.calibration_regions:
            raise ValueError("calibration_regions cannot be empty")


def _canonical_history(
    history: list[dict[str, object]],
    positions: dict[int, str],
    *,
    raise_amount_semantics: str,
) -> tuple[list[str], set[int], list[str]]:
    tokens: list[str] = []
    folded: set[int] = set()
    issues: list[str] = []
    for index, event in enumerate(history):
        if not isinstance(event, dict):
            issues.append(f"action_history[{index}] is not structured")
            continue
        actor = event.get("actor_seat")
        if isinstance(actor, bool) or not isinstance(actor, int) or actor not in positions:
            issues.append(f"action_history[{index}] actor seat is unresolved")
            continue
        action = str(event.get("action"))
        prefix = positions[actor]
        if action in {"fold", "check", "call"}:
            tokens.append(f"{prefix} {action}")
            if action == "fold":
                folded.add(actor)
        elif action == "bet":
            if event.get("amount_bb") is None:
                issues.append(f"action_history[{index}] bet amount is missing")
            else:
                tokens.append(f"{prefix} bet:{_bb(event['amount_bb'], 'bet amount')}")
        elif action == "raise":
            to_amount = event.get("to_amount_bb")
            if to_amount is None and raise_amount_semantics == "amount_is_to":
                to_amount = event.get("amount_bb")
            if to_amount is None:
                issues.append(f"action_history[{index}] raise-to amount is ambiguous")
            else:
                tokens.append(f"{prefix} raise_to:{_bb(to_amount, 'raise-to amount')}")
        elif action == "all_in":
            if event.get("amount_bb") is None:
                issues.append(f"action_history[{index}] all-in amount is missing")
            else:
                tokens.append(f"{prefix} all_in:{_bb(event['amount_bb'], 'all-in amount')}")
        else:
            issues.append(f"action_history[{index}] has unsupported action {action!r}")
    return tokens, folded, issues


def build_backend_observation(
    tracked_result: dict[str, object],
    *,
    config: BackendAdapterConfig | None = None,
) -> dict[str, object]:
    settings = config or BackendAdapterConfig()
    state = tracked_result.get("state")
    if not isinstance(state, dict):
        return {"status": "abstain_missing_tracked_state", "payload": None, "missing": ["state"]}
    missing = set(str(value) for value in tracked_result.get("missing_critical_fields", []))
    if state.get("history_complete") is not True:
        missing.add("complete_action_history")
    identity_gate = tracked_result.get("identity_gate")
    if not isinstance(identity_gate, dict) or identity_gate.get("status") != "passed":
        missing.add("player_identities")
    stacks = state.get("seat_stacks_bb")
    if not isinstance(stacks, list):
        missing.add("seat_stacks_bb")
        stacks = []
    seat_numbers = [
        int(row["seat_no"])
        for row in stacks
        if isinstance(row, dict) and isinstance(row.get("seat_no"), int)
    ]
    hero_seat, dealer_seat = state.get("hero_seat"), state.get("dealer_seat")
    if isinstance(hero_seat, bool) or not isinstance(hero_seat, int):
        missing.add("hero_seat")
    if isinstance(dealer_seat, bool) or not isinstance(dealer_seat, int):
        missing.add("dealer_seat")
    positions: dict[int, str] = {}
    if not ({"hero_seat", "dealer_seat"} & missing) and seat_numbers:
        try:
            positions = seat_positions(seat_numbers, int(dealer_seat))
        except ValueError:
            missing.add("hero_position")
    history = state.get("action_history")
    if not isinstance(history, list):
        missing.add("action_history")
        history = []
    canonical_history, folded, history_issues = _canonical_history(
        history,
        positions,
        raise_amount_semantics=settings.raise_amount_semantics,
    ) if positions else ([], set(), ["position map unavailable"])
    if history_issues:
        missing.add("canonical_action_history")
    active_players = len(seat_numbers) - len(folded)
    if active_players < 2:
        missing.add("active_players")
    if isinstance(hero_seat, int) and hero_seat in folded:
        missing.add("hero_active")

    active_seats = set(seat_numbers) - folded
    active_opponents = active_seats - ({int(hero_seat)} if isinstance(hero_seat, int) else set())
    raw_effective_rows = state.get("effective_stacks_by_opponent_bb")
    effective_by_opponent: dict[int, str] = {}
    effective_stack_issues: list[str] = []
    if isinstance(raw_effective_rows, list):
        for index, row in enumerate(raw_effective_rows):
            if not isinstance(row, dict):
                effective_stack_issues.append(f"effective_stacks_by_opponent_bb[{index}] is not structured")
                continue
            seat = row.get("opponent_seat")
            if isinstance(seat, bool) or not isinstance(seat, int) or seat == hero_seat or seat in effective_by_opponent:
                effective_stack_issues.append(f"effective_stacks_by_opponent_bb[{index}] opponent seat is invalid")
                continue
            try:
                effective_by_opponent[seat] = _bb(
                    row.get("effective_stack_bb"),
                    f"opponent seat {seat} effective stack",
                )
            except ValueError as error:
                effective_stack_issues.append(str(error))

    effective_stack = state.get("effective_stack_bb")
    if effective_stack is None and len(active_opponents) == 1:
        opponent = next(iter(active_opponents))
        effective_stack = effective_by_opponent.get(opponent)
    elif (
        effective_stack is None
        and len(active_opponents) > 1
        and settings.multiway_effective_stack_policy == "minimum_active_opponent"
    ):
        if set(effective_by_opponent) >= active_opponents:
            effective_stack = min(
                (Decimal(effective_by_opponent[seat]) for seat in active_opponents)
            )
        else:
            effective_stack_issues.append("active opponent effective-stack coverage is incomplete")
    if effective_stack is None:
        missing.add("effective_stack_bb")
    else:
        missing.discard("effective_stack_bb")
    if effective_stack_issues:
        missing.add("effective_stacks_by_opponent_bb")
    perception = tracked_result.get("perception_evidence")
    if not isinstance(perception, dict):
        missing.add("perception_evidence")
        perception = {}
    confidence = perception.get("minimum_supported_confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        missing.add("perception_confidence")
    if missing:
        return {
            "status": "abstain_incomplete_backend_state",
            "payload": None,
            "missing": sorted(missing),
            "audit": {
                "history_issues": history_issues,
                "active_players": active_players,
                "seat_positions": positions,
                "active_opponents": sorted(active_opponents),
                "effective_stack_issues": effective_stack_issues,
            },
        }

    frames = perception.get("frames")
    after_frame = frames.get("after") if isinstance(frames, dict) else None
    if not isinstance(after_frame, dict) or not isinstance(after_frame.get("image_sha256"), str):
        return {"status": "abstain_incomplete_backend_state", "payload": None, "missing": ["after_frame_evidence"]}
    image_sha = str(after_frame["image_sha256"])
    image_path = after_frame.get("image_path")
    observed_confidence = float(confidence)
    explicit_confidence = 1.0
    values = {
        "table_id": str(tracked_result.get("track_id")),
        "hand_id": str(state["hand_id"]),
        "game": settings.game,
        "players": active_players,
        "hero_position": positions[int(hero_seat)],
        "effective_stack_bb": _bb(effective_stack, "effective_stack_bb"),
        "pot_bb": _bb(state.get("pot_bb"), "pot_bb"),
        "to_call_bb": _bb(state.get("to_call_bb"), "to_call_bb"),
        "board": list(state.get("board_cards", [])),
        "hero_cards": list(state.get("hero_cards", [])),
        "action_history": canonical_history,
        "legal_actions": list(state.get("legal_actions", [])),
        "rake_model": settings.rake_model,
        "utility_model": settings.utility_model,
    }
    explicit_fields = {"table_id", "game", "rake_model", "utility_model"}
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "provider": settings.provider_id,
        "provider_version": settings.provider_version,
        "frame": {
            "frame_id": image_sha[:24],
            "image_sha256": image_sha,
        },
        "calibration": {
            "profile_id": settings.calibration_profile_id,
            "regions": copy.deepcopy(settings.calibration_regions),
        },
        "fields": {
            name: {
                "value": value,
                "confidence": explicit_confidence if name in explicit_fields else observed_confidence,
                "region": "table",
            }
            for name, value in values.items()
        },
        "apc_evidence": {
            "units": "BB",
            "identity_gate": copy.deepcopy(identity_gate),
            "player_identities": copy.deepcopy(state.get("player_identities")),
            "seat_positions": positions,
            "folded_seats": sorted(folded),
            "checkpoint_provenance": copy.deepcopy(perception.get("checkpoint_provenance")),
            "history_semantics": {
                "raise_amount_semantics": settings.raise_amount_semantics,
                "structured_source": copy.deepcopy(history),
            },
            "effective_stack_semantics": {
                "multiway_policy": settings.multiway_effective_stack_policy,
                "by_opponent_bb": effective_by_opponent,
                "active_opponents": sorted(active_opponents),
                "selected_scalar_bb": _bb(effective_stack, "effective_stack_bb"),
            },
        },
    }
    if isinstance(image_path, str) and image_path:
        payload["frame"]["image_path"] = image_path
    return {
        "status": "observation_ready_uncalibrated",
        "payload": payload,
        "missing": [],
        "audit": {
            "active_players": active_players,
            "seat_positions": positions,
            "canonical_action_history": canonical_history,
            "effective_stack_policy": settings.multiway_effective_stack_policy,
            "effective_stacks_by_opponent_bb": effective_by_opponent,
            "minimum_supported_confidence": observed_confidence,
            "recommendation_allowed": False,
        },
    }
