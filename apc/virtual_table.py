from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation

from apc.recommendation import action_command_from_solver_id


def _decimal(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a decimal number") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _fingerprint(value: object) -> str:
    material = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class VirtualDecisionTable:
    """One solver-backed, internal-only virtual-chip decision episode.

    This is intentionally not a full poker engine. It provides the exact state,
    legality and BB feedback contract needed to evaluate policies before full-hand
    self-play exists, and it has no coordinates, input hooks or external actuation.
    """

    def __init__(self, spot: object, *, to_call_bb: object = "0") -> None:
        self.spot = spot
        self.to_call_bb = _decimal(to_call_bb, "to_call_bb")
        self._terminal = False
        action_ids = tuple(action.action for action in spot.actions)
        if "call" in action_ids and self.to_call_bb <= 0:
            raise ValueError("a solver call action requires explicit positive to_call_bb")
        self._actions = {action.action: action for action in spot.actions}
        self._best_ev = max(action.ev for action in spot.actions)
        self._episode_id = _fingerprint(
            {
                "kind": "controlled_virtual_decision_v1",
                "node": spot.node_id,
                "state": spot.key.fingerprint,
                "to_call_bb": format(self.to_call_bb, "f"),
            }
        )

    def observation(self) -> dict[str, object]:
        key = self.spot.key
        return {
            "schema_version": "1.0.0",
            "episode_id": self._episode_id,
            "environment": "controlled_virtual_chips",
            "scope": "single_solver_decision",
            "terminal": self._terminal,
            "units": "BB",
            "state": {
                "game": key.game,
                "players": key.players,
                "hero_position": key.hero_position,
                "effective_stack_bb": format(key.effective_stack_bb, "f"),
                "pot_bb": format(key.pot_bb, "f"),
                "to_call_bb": format(self.to_call_bb, "f"),
                "board": [str(card) for card in key.board],
                "hero_cards": [str(card) for card in key.hero_cards],
                "action_history": list(key.action_history),
                "rake_model": key.rake_model,
                "utility_model": key.utility_model,
                "legal_actions": list(self._actions),
            },
            "provider": {
                "internal_virtual_chips": True,
                "external_actuation": False,
                "screen_or_input_control": False,
                "reward_oracle_hidden_from_observation": True,
            },
            "provenance": {
                "solver_node_id": self.spot.node_id,
                "solver_fingerprint": key.fingerprint,
                "solver_source": self.spot.source,
                "solver_source_version": self.spot.source_version,
            },
        }

    def step(self, action_id: str) -> dict[str, object]:
        if self._terminal:
            raise ValueError("virtual decision episode is already terminal")
        if action_id not in self._actions:
            raise ValueError(f"action is not legal in this virtual decision: {action_id}")
        key = self.spot.key
        command = action_command_from_solver_id(
            action_id,
            pot_bb=key.pot_bb,
            to_call_bb=self.to_call_bb,
            effective_stack_bb=key.effective_stack_bb,
        ).payload()
        committed = command.get("to_amount_bb") or command.get("amount_bb")
        if committed is not None and Decimal(str(committed)) > key.effective_stack_bb:
            raise ValueError("virtual action size exceeds the effective stack")
        action = self._actions[action_id]
        regret = self._best_ev - action.ev
        self._terminal = True
        feedback = {
            "schema_version": "1.0.0",
            "episode_id": self._episode_id,
            "status": "terminal",
            "units": "BB",
            "selected_action_id": action_id,
            "command": command,
            "reward_bb": format(action.ev, "f"),
            "best_available_ev_bb": format(self._best_ev, "f"),
            "regret_bb": format(regret, "f"),
            "completed_episode_feedback": True,
            "full_hand_completed": False,
            "external_actuation": False,
            "solver_provenance": {
                "node_id": self.spot.node_id,
                "fingerprint": key.fingerprint,
                "source": self.spot.source,
                "source_version": self.spot.source_version,
            },
        }
        feedback["feedback_fingerprint"] = _fingerprint(feedback)
        return feedback
