from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Protocol


D = Decimal
_ACTION_RE = re.compile(
    r"^(?P<actor>.+) (?P<action>fold|check|call|bet:(?P<bet>[0-9]+(?:\.[0-9]+)?)|raise_to:(?P<raise>[0-9]+(?:\.[0-9]+)?)|all_in:(?P<all_in>[0-9]+(?:\.[0-9]+)?))$"
)


class TransitionState(Protocol):
    table_id: str
    hand_id: str
    revision: int
    game: str
    players: int
    hero_position: str
    effective_stack_bb: Decimal
    pot_bb: Decimal
    to_call_bb: Decimal
    board: tuple[object, ...]
    hero_cards: tuple[object, ...]
    action_history: tuple[str, ...]
    rake_model: str
    utility_model: str


class StateTransitionError(ValueError):
    def __init__(self, message: str, audit: dict[str, object]) -> None:
        super().__init__(message)
        self.audit = audit


def _value(value: object) -> object:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return value


def _action_is_normalized(action: str) -> bool:
    match = _ACTION_RE.fullmatch(action)
    if match is None:
        return False
    size = match.group("bet") or match.group("raise") or match.group("all_in")
    if size is None:
        return True
    try:
        return D(size).is_finite() and D(size) > 0
    except InvalidOperation:
        return False


def validate_state_transition(
    previous: TransitionState | None,
    current: TransitionState,
    *,
    expected_table_id: str,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    def check(
        code: str,
        passed: bool,
        message: str,
        *,
        prior: object = None,
        observed: object = None,
    ) -> None:
        row = {
            "code": code,
            "passed": passed,
            "message": message,
            "previous": _value(prior),
            "observed": _value(observed),
        }
        checks.append(row)
        if not passed:
            violations.append(row)

    check(
        "table_identity",
        current.table_id == expected_table_id,
        "table_id does not match the live session",
        prior=expected_table_id,
        observed=current.table_id,
    )
    normalized_actions = all(
        _action_is_normalized(action) for action in current.action_history
    )
    check(
        "normalized_action_tokens",
        normalized_actions,
        "action_history contains an unrecognized normalized action",
        observed=current.action_history,
    )

    if previous is None:
        audit = {
            "status": "accepted" if not violations else "rejected",
            "kind": "initial_state",
            "from_revision": None,
            "to_revision": current.revision,
            "hand_id": current.hand_id,
            "checks": checks,
            "violations": violations,
            "warnings": warnings,
            "deltas": {
                "board_cards_added": len(current.board),
                "actions_added": len(current.action_history),
                "pot_change_bb": None,
                "effective_stack_change_bb": None,
            },
        }
        if violations:
            raise StateTransitionError(str(violations[0]["message"]), audit)
        return audit

    check(
        "revision_forward",
        current.revision > previous.revision,
        f"State revision must advance beyond {previous.revision}",
        prior=previous.revision,
        observed=current.revision,
    )
    if current.revision > previous.revision + 1:
        warnings.append(
            {
                "code": "revision_gap",
                "message": "One or more stable states were not observed.",
                "missing_revisions": current.revision - previous.revision - 1,
            }
        )

    new_hand = current.hand_id != previous.hand_id
    if not new_hand:
        board_prefix = (
            len(current.board) >= len(previous.board)
            and current.board[: len(previous.board)] == previous.board
        )
        check(
            "board_prefix",
            board_prefix,
            "board cannot roll back or change within the same hand",
            prior=previous.board,
            observed=current.board,
        )
        history_prefix = (
            len(current.action_history) >= len(previous.action_history)
            and current.action_history[: len(previous.action_history)]
            == previous.action_history
        )
        check(
            "action_history_prefix",
            history_prefix,
            "action_history cannot roll back or change within the same hand",
            prior=previous.action_history,
            observed=current.action_history,
        )
        check(
            "hero_cards_immutable",
            set(current.hero_cards) == set(previous.hero_cards),
            "hero_cards cannot change within the same hand",
            prior=previous.hero_cards,
            observed=current.hero_cards,
        )
        for code, field in (
            ("game_immutable", "game"),
            ("hero_position_immutable", "hero_position"),
            ("rake_model_immutable", "rake_model"),
            ("utility_model_immutable", "utility_model"),
        ):
            prior = getattr(previous, field)
            observed = getattr(current, field)
            check(
                code,
                observed == prior,
                f"{field} cannot change within the same hand",
                prior=prior,
                observed=observed,
            )
        check(
            "player_count_nonincreasing",
            current.players <= previous.players,
            "players cannot increase within the same hand",
            prior=previous.players,
            observed=current.players,
        )
        check(
            "pot_nondecreasing",
            current.pot_bb >= previous.pot_bb,
            "pot_bb cannot decrease within the same hand",
            prior=previous.pot_bb,
            observed=current.pot_bb,
        )
        if previous.players == 2 and current.players == 2:
            check(
                "heads_up_effective_stack_nonincreasing",
                current.effective_stack_bb <= previous.effective_stack_bb,
                "effective_stack_bb cannot increase in a heads-up hand",
                prior=previous.effective_stack_bb,
                observed=current.effective_stack_bb,
            )
        no_public_progress = (
            current.board == previous.board
            and current.action_history == previous.action_history
        )
        if no_public_progress:
            check(
                "unchanged_state_pot",
                current.pot_bb == previous.pot_bb,
                "pot_bb changed without a board or action-history transition",
                prior=previous.pot_bb,
                observed=current.pot_bb,
            )
            check(
                "unchanged_state_stack",
                current.effective_stack_bb == previous.effective_stack_bb,
                "effective_stack_bb changed without a board or action-history transition",
                prior=previous.effective_stack_bb,
                observed=current.effective_stack_bb,
            )
            check(
                "unchanged_state_call_price",
                current.to_call_bb == previous.to_call_bb,
                "to_call_bb changed without a board or action-history transition",
                prior=previous.to_call_bb,
                observed=current.to_call_bb,
            )

    audit = {
        "status": "accepted" if not violations else "rejected",
        "kind": "new_hand" if new_hand else "same_hand_progression",
        "from_revision": previous.revision,
        "to_revision": current.revision,
        "hand_id": current.hand_id,
        "checks": checks,
        "violations": violations,
        "warnings": warnings,
        "deltas": {
            "board_cards_added": len(current.board)
            if new_hand
            else len(current.board) - len(previous.board),
            "actions_added": len(current.action_history)
            if new_hand
            else len(current.action_history) - len(previous.action_history),
            "pot_change_bb": None
            if new_hand
            else format(current.pot_bb - previous.pot_bb, "f"),
            "effective_stack_change_bb": None
            if new_hand
            else format(
                current.effective_stack_bb - previous.effective_stack_bb, "f"
            ),
        },
    }
    if violations:
        raise StateTransitionError(str(violations[0]["message"]), audit)
    return audit
