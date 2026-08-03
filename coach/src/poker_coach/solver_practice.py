from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from .range_strategy import public_node_fingerprint
from .solutions import SolvedSpot


@dataclass
class SolverPracticeSession:
    session_id: str
    spot: SolvedSpot
    answered: bool = False


def _action_label(action: str) -> str:
    if ":" not in action:
        return action.title()
    kind, size = action.split(":", 1)
    return f"{kind.replace('_', ' ').title()} {size}"


class SolverPracticeService:
    def __init__(self) -> None:
        self._sessions: dict[str, SolverPracticeSession] = {}
        self._lock = threading.RLock()

    def create(
        self,
        solutions: Iterable[SolvedSpot],
        *,
        public_fingerprint: str | None = None,
    ) -> dict[str, object]:
        rows = [
            spot
            for spot in solutions
            if spot.key.hero_cards
            and (public_fingerprint is None or public_node_fingerprint(spot) == public_fingerprint)
        ]
        if not rows:
            raise ValueError("No imported private-hand solution matches this practice node")
        spot = sorted(rows, key=lambda row: (row.node_id, row.key.fingerprint))[0]
        session = SolverPracticeSession(uuid.uuid4().hex, spot)
        with self._lock:
            self._sessions[session.session_id] = session
        return self._challenge(session)

    def submit(self, session_id: str, action_id: str) -> dict[str, object]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown solver practice session: {session_id}")
            if session.answered:
                raise ValueError("Solver practice session is already answered")
            try:
                chosen = session.spot.action(action_id)
            except KeyError as error:
                raise ValueError(f"Illegal solved action: {action_id}") from error
            session.answered = True
        loss = session.spot.best_ev - chosen.ev
        return {
            "session_id": session_id,
            "chosen_action": action_id,
            "chosen_frequency": format(chosen.frequency, "f"),
            "chosen_ev_bb": format(chosen.ev, "f"),
            "best_ev_bb": format(session.spot.best_ev, "f"),
            "ev_loss_bb": format(loss, "f"),
            "grade": "excellent" if loss <= Decimal("0.02") else "review" if loss <= Decimal("0.15") else "major_leak",
            "strategy": [
                {
                    "action": action.action,
                    "label": _action_label(action.action),
                    "frequency": format(action.frequency, "f"),
                    "ev_bb": format(action.ev, "f"),
                }
                for action in session.spot.actions
            ],
            "source": session.spot.source,
            "source_version": session.spot.source_version,
            "node_id": session.spot.node_id,
        }

    @staticmethod
    def _challenge(session: SolverPracticeSession) -> dict[str, object]:
        spot = session.spot
        key = spot.key
        return {
            "session_id": session.session_id,
            "node_id": spot.node_id,
            "public_fingerprint": public_node_fingerprint(spot),
            "hero_position": key.hero_position,
            "hero_cards": [str(card) for card in key.hero_cards],
            "board": [str(card) for card in key.board],
            "pot_bb": format(key.pot_bb, "f"),
            "effective_stack_bb": format(key.effective_stack_bb, "f"),
            "action_history": list(key.action_history),
            "actions": [
                {"action_id": action.action, "label": _action_label(action.action)}
                for action in spot.actions
            ],
            "strategy_hidden": True,
            "source": spot.source,
            "source_version": spot.source_version,
        }
