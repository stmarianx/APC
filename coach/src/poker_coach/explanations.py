from __future__ import annotations

from decimal import Decimal

from .decision_math import minimum_defense_frequency, stack_to_pot_ratio
from .models import ActionKind, HandAction
from .replay import DecisionSnapshot
from .solutions import SolvedSpot


def action_id(action: HandAction) -> str:
    if action.kind == ActionKind.BET:
        return f"bet:{format(action.amount, 'f')}"
    if action.kind == ActionKind.RAISE:
        assert action.to_amount is not None
        return f"raise_to:{format(action.to_amount, 'f')}"
    return action.kind.value


def explain_decision(
    snapshot: DecisionSnapshot,
    action: HandAction,
    *,
    equity: Decimal | None = None,
    solution: SolvedSpot | None = None,
) -> dict[str, object]:
    if snapshot.action_index != action.index or snapshot.actor != action.player:
        raise ValueError("Snapshot and action do not describe the same decision")
    result: dict[str, object] = {
        "action_index": action.index,
        "street": action.street.name.lower(),
        "actor": snapshot.actor,
        "observed_action": action_id(action),
        "pot_before_action": format(snapshot.pot, "f"),
        "actor_stack": format(snapshot.actor_stack, "f"),
        "effective_stack": format(snapshot.effective_stack, "f"),
        "to_call": format(snapshot.to_call, "f"),
        "board": [str(card) for card in snapshot.board],
        "assumptions": [
            "Pot and stack values are reconstructed from the saved action history.",
            "MDF and polar formulas are idealized heads-up river-style diagnostics, not universal prescriptions.",
        ],
    }
    if snapshot.pot > 0:
        result["spr"] = format(stack_to_pot_ratio(snapshot.effective_stack, snapshot.pot), "f")
    if snapshot.to_call > 0:
        required = snapshot.to_call / (snapshot.pot + snapshot.to_call)
        result["call_break_even_equity"] = format(required, "f")
        if equity is not None:
            if not Decimal("0") <= equity <= Decimal("1"):
                raise ValueError("Equity must be between zero and one")
            call_ev = equity * (snapshot.pot + snapshot.to_call) - snapshot.to_call
            result["input_equity"] = format(equity, "f")
            result["call_ev"] = format(call_ev, "f")

    risk: Decimal | None = None
    if action.kind == ActionKind.BET:
        risk = action.amount
    elif action.kind == ActionKind.RAISE:
        assert action.to_amount is not None
        actor_state = next(player for player in snapshot.players if player.player == snapshot.actor)
        risk = action.to_amount - actor_state.committed_street
    if risk is not None and risk > 0 and snapshot.pot > 0:
        result["aggressive_risk"] = format(risk, "f")
        result["pure_bluff_break_even_folds"] = format(risk / (snapshot.pot + risk), "f")
        result["idealized_mdf"] = format(minimum_defense_frequency(snapshot.pot, risk), "f")

    if solution is not None:
        chosen = solution.action(action_id(action))
        result["solution"] = {
            "spot_fingerprint": solution.key.fingerprint,
            "source": solution.source,
            "source_version": solution.source_version,
            "frequency": format(chosen.frequency, "f"),
            "action_ev": format(chosen.ev, "f"),
            "best_ev": format(solution.best_ev, "f"),
            "ev_loss": format(solution.ev_loss(chosen.action), "f"),
            "strategy": [
                {"action": candidate.action, "frequency": format(candidate.frequency, "f"), "ev": format(candidate.ev, "f")}
                for candidate in solution.actions
            ],
        }
    return result

