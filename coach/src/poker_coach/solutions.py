from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from .models import Card
from .isomorphism import canonicalize_suit_state


@dataclass(frozen=True)
class SolutionKey:
    game: str
    players: int
    hero_position: str
    effective_stack_bb: Decimal
    pot_bb: Decimal
    board: tuple[Card, ...]
    action_history: tuple[str, ...]
    rake_model: str
    utility_model: str = "chip_ev"
    allowed_sizes: tuple[Decimal, ...] = ()
    hero_cards: tuple[Card, ...] = ()

    def __post_init__(self) -> None:
        if self.players < 2:
            raise ValueError("A solved spot requires at least two players")
        if self.effective_stack_bb < 0 or self.pot_bb <= 0:
            raise ValueError("Invalid normalized stack or pot")
        if len(self.board) not in (0, 3, 4, 5):
            raise ValueError("Invalid board length")
        if len(self.hero_cards) not in (0, 2):
            raise ValueError("A solution key requires zero or two hero cards")
        if len(set(self.board + self.hero_cards)) != len(self.board + self.hero_cards):
            raise ValueError("Solution key contains conflicting cards")

    def canonical(self) -> dict[str, object]:
        return {
            "game": self.game,
            "players": self.players,
            "hero_position": self.hero_position,
            "effective_stack_bb": format(self.effective_stack_bb, "f"),
            "pot_bb": format(self.pot_bb, "f"),
            "board": [str(card) for card in self.board],
            "action_history": list(self.action_history),
            "rake_model": self.rake_model,
            "utility_model": self.utility_model,
            "allowed_sizes": [format(size, "f") for size in self.allowed_sizes],
            "hero_cards": [str(card) for card in self.hero_cards],
        }

    def canonical_isomorphic(self) -> dict[str, object]:
        """Fingerprint payload with suit names removed but suit relations retained."""

        board, hero_cards = canonicalize_suit_state(self.board, self.hero_cards)
        payload = self.canonical()
        payload["board"] = [str(card) for card in board]
        payload["hero_cards"] = [str(card) for card in hero_cards]
        payload["card_normalization"] = "suit_isomorphism_v1"
        return payload

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_isomorphic(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActionSolution:
    action: str
    frequency: Decimal
    ev: Decimal

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("Solved action id cannot be empty")
        if not Decimal("0") <= self.frequency <= Decimal("1"):
            raise ValueError("Action frequency must be between zero and one")


@dataclass(frozen=True)
class SolvedSpot:
    key: SolutionKey
    actions: tuple[ActionSolution, ...]
    source: str
    source_version: str = ""
    node_id: str = ""

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("Solved spot requires actions")
        if len({action.action for action in self.actions}) != len(self.actions):
            raise ValueError("Solved action ids must be unique")
        total = sum((action.frequency for action in self.actions), Decimal("0"))
        if abs(total - Decimal("1")) > Decimal("0.000001"):
            raise ValueError("Solved frequencies must sum to one")
        if not self.source:
            raise ValueError("Solved spot requires provenance")
        if any(character in self.node_id for character in "\r\n\t"):
            raise ValueError("Solved node id cannot contain control whitespace")

    @property
    def best_ev(self) -> Decimal:
        return max(action.ev for action in self.actions)

    def action(self, action_id: str) -> ActionSolution:
        for action in self.actions:
            if action.action == action_id:
                return action
        raise KeyError(action_id)

    def ev_loss(self, action_id: str) -> Decimal:
        return self.best_ev - self.action(action_id).ev


class SolutionStore(Protocol):
    def get(self, key: SolutionKey) -> SolvedSpot | None: ...
    def put(self, spot: SolvedSpot) -> None: ...


class InMemorySolutionStore:
    def __init__(self) -> None:
        self._spots: dict[str, SolvedSpot] = {}

    def get(self, key: SolutionKey) -> SolvedSpot | None:
        return self._spots.get(key.fingerprint)

    def put(self, spot: SolvedSpot) -> None:
        self._spots[spot.key.fingerprint] = spot

    def all(self) -> tuple[SolvedSpot, ...]:
        return tuple(self._spots[fingerprint] for fingerprint in sorted(self._spots))
