from __future__ import annotations

import json
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from importlib.resources import files
from typing import Iterable

from .decision_math import (
    bluff_break_even_fold_frequency,
    call_break_even_equity,
    minimum_defense_frequency,
    stack_to_pot_ratio,
)
from .models import Card


D = Decimal


def _number(value: Decimal) -> float:
    return float(value)


@dataclass(frozen=True)
class TrainingAction:
    action_id: str
    label: str
    frequency: Decimal
    ev_bb: Decimal
    amount_bb: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.action_id or not self.label:
            raise ValueError("Training actions require an id and label")
        if not D("0") <= self.frequency <= D("1"):
            raise ValueError("Action frequency must be between zero and one")
        if self.amount_bb is not None and self.amount_bb < 0:
            raise ValueError("Action amount cannot be negative")


@dataclass(frozen=True)
class StrategyProvenance:
    tier: str
    source: str
    version: str
    solver_verified: bool
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.tier not in {"educational_baseline", "solver_verified"}:
            raise ValueError("Unknown strategy provenance tier")
        if self.solver_verified != (self.tier == "solver_verified"):
            raise ValueError("Solver verification flag conflicts with provenance tier")
        if not self.source or not self.version:
            raise ValueError("Strategy provenance requires source and version")


@dataclass(frozen=True)
class TrainingScenario:
    scenario_id: str
    title: str
    category: str
    difficulty: str
    street: str
    hero_position: str
    villain_position: str
    hero_cards: tuple[Card, Card]
    board: tuple[Card, ...]
    pot_bb: Decimal
    to_call_bb: Decimal
    effective_stack_bb: Decimal
    action_history: tuple[str, ...]
    actions: tuple[TrainingAction, ...]
    concepts: tuple[str, ...]
    explanation: str
    provenance: StrategyProvenance

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.title:
            raise ValueError("Training scenario requires an id and title")
        if self.street not in {"preflop", "flop", "turn", "river"}:
            raise ValueError("Unknown training street")
        if len(self.board) not in {0, 3, 4, 5}:
            raise ValueError("Invalid board length")
        if len(set(self.hero_cards + self.board)) != len(self.hero_cards + self.board):
            raise ValueError("Scenario contains duplicate cards")
        if self.pot_bb <= 0 or self.to_call_bb < 0 or self.effective_stack_bb < 0:
            raise ValueError("Invalid pot, call, or stack value")
        if not self.actions:
            raise ValueError("Training scenario requires actions")
        if len({action.action_id for action in self.actions}) != len(self.actions):
            raise ValueError("Training action ids must be unique")
        frequency = sum((action.frequency for action in self.actions), D("0"))
        if abs(frequency - D("1")) > D("0.000001"):
            raise ValueError("Training action frequencies must sum to one")

    @property
    def best_ev(self) -> Decimal:
        return max(action.ev_bb for action in self.actions)

    def action(self, action_id: str) -> TrainingAction:
        for action in self.actions:
            if action.action_id == action_id:
                return action
        raise KeyError(action_id)

    def mathematical_context(self) -> dict[str, float]:
        result = {"spr": _number(stack_to_pot_ratio(self.effective_stack_bb, self.pot_bb))}
        if self.to_call_bb > 0:
            pot_before_bet = max(D("0"), self.pot_bb - self.to_call_bb)
            result["break_even_call_equity"] = _number(
                call_break_even_equity(pot_before_bet, self.to_call_bb)
            )
            result["minimum_defense_frequency"] = _number(
                minimum_defense_frequency(pot_before_bet, self.to_call_bb)
            )
        bet_sizes = [action.amount_bb for action in self.actions if action.amount_bb]
        if self.to_call_bb == 0 and bet_sizes:
            representative = max(bet_sizes)
            result["largest_bet_break_even_fold_frequency"] = _number(
                bluff_break_even_fold_frequency(self.pot_bb, representative)
            )
        return result

    def to_dict(self, *, reveal_strategy: bool = False) -> dict[str, object]:
        action_rows: list[dict[str, object]] = []
        for action in self.actions:
            row: dict[str, object] = {
                "action_id": action.action_id,
                "label": action.label,
                "amount_bb": None if action.amount_bb is None else _number(action.amount_bb),
            }
            if reveal_strategy:
                row["frequency"] = _number(action.frequency)
                row["ev_bb"] = _number(action.ev_bb)
            action_rows.append(row)
        result: dict[str, object] = {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "category": self.category,
            "difficulty": self.difficulty,
            "street": self.street,
            "hero_position": self.hero_position,
            "villain_position": self.villain_position,
            "hero_cards": [str(card) for card in self.hero_cards],
            "board": [str(card) for card in self.board],
            "pot_bb": _number(self.pot_bb),
            "to_call_bb": _number(self.to_call_bb),
            "effective_stack_bb": _number(self.effective_stack_bb),
            "action_history": list(self.action_history),
            "actions": action_rows,
            "concepts": list(self.concepts),
            "provenance": {
                "tier": self.provenance.tier,
                "source": self.provenance.source,
                "version": self.provenance.version,
                "solver_verified": self.provenance.solver_verified,
                "source_refs": list(self.provenance.source_refs),
            },
        }
        if reveal_strategy:
            result["explanation"] = self.explanation
            result["math"] = self.mathematical_context()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TrainingScenario":
        provenance_data = dict(data["provenance"])
        return cls(
            scenario_id=str(data["scenario_id"]),
            title=str(data["title"]),
            category=str(data["category"]),
            difficulty=str(data["difficulty"]),
            street=str(data["street"]),
            hero_position=str(data["hero_position"]),
            villain_position=str(data["villain_position"]),
            hero_cards=tuple(Card.parse(str(card)) for card in data["hero_cards"]),  # type: ignore[arg-type]
            board=tuple(Card.parse(str(card)) for card in data["board"]),
            pot_bb=D(str(data["pot_bb"])),
            to_call_bb=D(str(data["to_call_bb"])),
            effective_stack_bb=D(str(data["effective_stack_bb"])),
            action_history=tuple(str(item) for item in data["action_history"]),
            actions=tuple(
                TrainingAction(
                    action_id=str(action["action_id"]),
                    label=str(action["label"]),
                    frequency=D(str(action["frequency"])),
                    ev_bb=D(str(action["ev_bb"])),
                    amount_bb=None if action.get("amount_bb") is None else D(str(action["amount_bb"])),
                )
                for action in data["actions"]
            ),
            concepts=tuple(str(item) for item in data["concepts"]),
            explanation=str(data["explanation"]),
            provenance=StrategyProvenance(
                tier=str(provenance_data["tier"]),
                source=str(provenance_data["source"]),
                version=str(provenance_data["version"]),
                solver_verified=bool(provenance_data["solver_verified"]),
                source_refs=tuple(str(item) for item in provenance_data.get("source_refs", [])),
            ),
        )


class ScenarioLibrary:
    def __init__(self, scenarios: Iterable[TrainingScenario]) -> None:
        self._scenarios = {scenario.scenario_id: scenario for scenario in scenarios}
        if not self._scenarios:
            raise ValueError("Scenario library cannot be empty")

    @classmethod
    def bundled(cls) -> "ScenarioLibrary":
        resource = files("poker_coach").joinpath("data/training_scenarios.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
        return cls(TrainingScenario.from_dict(row) for row in payload["scenarios"])

    def get(self, scenario_id: str) -> TrainingScenario:
        try:
            return self._scenarios[scenario_id]
        except KeyError as error:
            raise KeyError(f"Unknown scenario: {scenario_id}") from error

    def all(self) -> tuple[TrainingScenario, ...]:
        return tuple(self._scenarios.values())


@dataclass
class TrainingSession:
    session_id: str
    scenario_ids: tuple[str, ...]
    created_at: str
    cursor: int = 0
    decisions: list[dict[str, object]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.cursor >= len(self.scenario_ids)


class TrainingService:
    def __init__(self, library: ScenarioLibrary | None = None) -> None:
        self.library = library or ScenarioLibrary.bundled()
        self._sessions: dict[str, TrainingSession] = {}
        self._lock = threading.RLock()

    def list_scenarios(self) -> list[dict[str, object]]:
        return [scenario.to_dict() for scenario in self.library.all()]

    def create_session(self, *, count: int | None = None, seed: int | None = None) -> dict[str, object]:
        scenarios = list(self.library.all())
        if count is None:
            count = len(scenarios)
        if count < 1 or count > len(scenarios):
            raise ValueError(f"count must be between 1 and {len(scenarios)}")
        random.Random(seed).shuffle(scenarios)
        session = TrainingSession(
            session_id=uuid.uuid4().hex,
            scenario_ids=tuple(scenario.scenario_id for scenario in scenarios[:count]),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return self.session_state(session.session_id)

    def session_state(self, session_id: str) -> dict[str, object]:
        with self._lock:
            session = self._session(session_id)
            result: dict[str, object] = {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "complete": session.complete,
                "progress": {
                    "answered": session.cursor,
                    "total": len(session.scenario_ids),
                    "score": self._score(session),
                },
                "decisions": list(session.decisions),
            }
            if not session.complete:
                result["scenario"] = self.library.get(session.scenario_ids[session.cursor]).to_dict()
            return result

    def submit_decision(self, session_id: str, action_id: str) -> dict[str, object]:
        with self._lock:
            session = self._session(session_id)
            if session.complete:
                raise ValueError("Training session is already complete")
            scenario = self.library.get(session.scenario_ids[session.cursor])
            try:
                chosen = scenario.action(action_id)
            except KeyError as error:
                raise ValueError(f"Illegal action for scenario: {action_id}") from error
            loss = scenario.best_ev - chosen.ev_bb
            grade = self._grade(loss)
            preferred = [
                action.action_id
                for action in scenario.actions
                if scenario.best_ev - action.ev_bb <= D("0.05")
            ]
            feedback = {
                "scenario": scenario.to_dict(reveal_strategy=True),
                "chosen_action_id": chosen.action_id,
                "chosen_frequency": _number(chosen.frequency),
                "chosen_ev_bb": _number(chosen.ev_bb),
                "best_ev_bb": _number(scenario.best_ev),
                "ev_loss_bb": _number(loss),
                "grade": grade,
                "preferred_actions": preferred,
                "message": self._message(grade, chosen, scenario),
                "answered_at": datetime.now(timezone.utc).isoformat(),
            }
            session.decisions.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "action_id": chosen.action_id,
                    "ev_loss_bb": _number(loss),
                    "grade": grade,
                }
            )
            session.cursor += 1
            return {
                "feedback": feedback,
                "progress": {
                    "answered": session.cursor,
                    "total": len(session.scenario_ids),
                    "score": self._score(session),
                },
                "complete": session.complete,
            }

    def _session(self, session_id: str) -> TrainingSession:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise KeyError(f"Unknown session: {session_id}") from error

    @staticmethod
    def _grade(loss: Decimal) -> str:
        if loss <= D("0.01"):
            return "excellent"
        if loss <= D("0.05"):
            return "good"
        if loss <= D("0.20"):
            return "review"
        return "major_leak"

    @staticmethod
    def _message(grade: str, chosen: TrainingAction, scenario: TrainingScenario) -> str:
        if grade == "excellent":
            return f"{chosen.label} is at the top of this reference strategy."
        if grade == "good":
            return f"{chosen.label} is close in EV and belongs in the strategic mix."
        if grade == "review":
            return f"{chosen.label} gives up noticeable EV in this model; review the pot geometry and ranges."
        return f"{chosen.label} is a major deviation in this model; rebuild the decision from range and equity assumptions."

    @staticmethod
    def _score(session: TrainingSession) -> int:
        if not session.decisions:
            return 0
        points = {"excellent": 100, "good": 80, "review": 50, "major_leak": 0}
        return round(sum(points[str(row["grade"])] for row in session.decisions) / len(session.decisions))
