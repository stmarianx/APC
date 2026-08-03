from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Iterable

from .isomorphism import suit_isomorphic
from .models import Card
from .solutions import SolvedSpot
from .board_texture import analyze_board_texture
from .state_transition import validate_state_transition
from .strategy_selection import DEFAULT_LATENCY_BUDGET_MS, StrategySelectionService


D = Decimal
SCHEMA_VERSION = "1.0.0"


def _decimal(value: object, field_name: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a decimal number")
    try:
        result = D(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field_name} must be a decimal number") from error
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def _required_text(payload: dict[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _string_array(payload: dict[str, object], field_name: str) -> tuple[str, ...]:
    value = payload.get(field_name, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field_name} must be an array of non-empty strings")
    rows = tuple(item.strip() for item in value)
    if len(set(rows)) != len(rows) and field_name == "legal_actions":
        raise ValueError("legal_actions must be unique")
    return rows


def _cards(payload: dict[str, object], field_name: str) -> tuple[Card, ...]:
    value = payload.get(field_name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a card-string array")
    try:
        return tuple(Card.parse(item) for item in value)
    except ValueError as error:
        raise ValueError(f"{field_name}: {error}") from error


def _action_label(action: str) -> str:
    if ":" not in action:
        return action.title()
    kind, size = action.split(":", 1)
    return f"{kind.replace('_', ' ').title()} {size}"


@dataclass(frozen=True)
class LiveTableState:
    table_id: str
    hand_id: str
    revision: int
    game: str
    players: int
    hero_position: str
    effective_stack_bb: Decimal
    pot_bb: Decimal
    to_call_bb: Decimal
    board: tuple[Card, ...]
    hero_cards: tuple[Card, ...]
    action_history: tuple[str, ...]
    legal_actions: tuple[str, ...]
    rake_model: str
    utility_model: str
    source: str

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if self.players < 2:
            raise ValueError("players must be at least two")
        if self.effective_stack_bb < 0:
            raise ValueError("effective_stack_bb cannot be negative")
        if self.pot_bb <= 0:
            raise ValueError("pot_bb must be positive")
        if self.to_call_bb < 0:
            raise ValueError("to_call_bb cannot be negative")
        if self.to_call_bb > self.effective_stack_bb:
            raise ValueError("to_call_bb cannot exceed the effective stack")
        if len(self.board) not in (0, 3, 4, 5):
            raise ValueError("board must contain zero, three, four, or five cards")
        if len(self.hero_cards) != 2:
            raise ValueError("hero_cards must contain exactly two cards")
        if len(set(self.board + self.hero_cards)) != len(self.board + self.hero_cards):
            raise ValueError("table state contains conflicting cards")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LiveTableState":
        schema = str(payload.get("schema_version", SCHEMA_VERSION))
        if schema != SCHEMA_VERSION:
            raise ValueError(f"Unsupported live-state schema: {schema}")
        revision = payload.get("revision")
        players = payload.get("players")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise ValueError("revision must be a non-negative integer")
        if isinstance(players, bool) or not isinstance(players, int):
            raise ValueError("players must be an integer")
        return cls(
            table_id=_required_text(payload, "table_id"),
            hand_id=_required_text(payload, "hand_id"),
            revision=revision,
            game=str(payload.get("game", "holdem_no_limit")).strip(),
            players=players,
            hero_position=_required_text(payload, "hero_position"),
            effective_stack_bb=_decimal(
                payload.get("effective_stack_bb"), "effective_stack_bb"
            ),
            pot_bb=_decimal(payload.get("pot_bb"), "pot_bb"),
            to_call_bb=_decimal(payload.get("to_call_bb", 0), "to_call_bb"),
            board=_cards(payload, "board"),
            hero_cards=_cards(payload, "hero_cards"),
            action_history=_string_array(payload, "action_history"),
            legal_actions=_string_array(payload, "legal_actions"),
            rake_model=str(payload.get("rake_model", "training_no_rake")).strip(),
            utility_model=str(payload.get("utility_model", "chip_ev")).strip(),
            source=str(payload.get("source", "normalized_feed")).strip(),
        )

    @property
    def street(self) -> str:
        return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(self.board)]

    @property
    def state_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "table_id": self.table_id,
            "hand_id": self.hand_id,
            "revision": self.revision,
            "street": self.street,
            "game": self.game,
            "players": self.players,
            "hero_position": self.hero_position,
            "effective_stack_bb": format(self.effective_stack_bb, "f"),
            "pot_bb": format(self.pot_bb, "f"),
            "to_call_bb": format(self.to_call_bb, "f"),
            "board": [str(card) for card in self.board],
            "hero_cards": [str(card) for card in self.hero_cards],
            "action_history": list(self.action_history),
            "legal_actions": list(self.legal_actions),
            "rake_model": self.rake_model,
            "utility_model": self.utility_model,
            "source": self.source,
        }


@dataclass(frozen=True)
class LiveSolutionMatch:
    solution: SolvedSpot
    confidence: str
    score: Decimal
    stack_error_bb: Decimal
    pot_error_bb: Decimal
    history_exact: bool
    card_match: str


@dataclass
class LiveTableSession:
    session_id: str
    table_id: str
    state: LiveTableState | None = None
    match: LiveSolutionMatch | None = None
    response: dict[str, object] | None = None
    decided_revisions: set[int] = field(default_factory=set)


class LiveTableService:
    def __init__(
        self,
        strategy_selector: StrategySelectionService | None = None,
        *,
        strategy_latency_budget_ms: int = DEFAULT_LATENCY_BUDGET_MS,
    ) -> None:
        if strategy_latency_budget_ms <= 0:
            raise ValueError("strategy_latency_budget_ms must be positive")
        self._sessions: dict[str, LiveTableSession] = {}
        self._lock = threading.RLock()
        self.strategy_selector = strategy_selector or StrategySelectionService()
        self.strategy_latency_budget_ms = strategy_latency_budget_ms

    def create_session(self, table_id: str) -> dict[str, object]:
        normalized = table_id.strip()
        if not normalized:
            raise ValueError("table_id is required")
        session = LiveTableSession(uuid.uuid4().hex, normalized)
        with self._lock:
            self._sessions[session.session_id] = session
        return {
            "session_id": session.session_id,
            "table_id": session.table_id,
            "last_revision": None,
            "status": "awaiting_state",
            "schema_version": SCHEMA_VERSION,
        }

    def current(self, session_id: str) -> dict[str, object]:
        with self._lock:
            session = self._session(session_id)
            if session.response is None:
                return {
                    "session_id": session.session_id,
                    "table_id": session.table_id,
                    "last_revision": None,
                    "status": "awaiting_state",
                    "schema_version": SCHEMA_VERSION,
                }
            return dict(session.response)

    def update_state(
        self,
        session_id: str,
        payload: dict[str, object],
        solutions: Iterable[SolvedSpot],
    ) -> dict[str, object]:
        state = LiveTableState.from_dict(payload)
        solution_rows = tuple(solutions)
        with self._lock:
            session = self._session(session_id)
            transition = self._validate_progression(session, state)
            match = self._best_match(state, solution_rows)
            response = self._response(session_id, state, match, transition)
            session.state = state
            session.match = match
            session.response = response
            return dict(response)

    def record_decision(
        self, session_id: str, revision: int, action_id: str
    ) -> dict[str, object]:
        with self._lock:
            session = self._session(session_id)
            if session.state is None or session.match is None:
                raise ValueError("Current table state has no matched solver solution")
            if revision != session.state.revision:
                raise ValueError(
                    f"Decision revision {revision} is stale; current revision is {session.state.revision}"
                )
            if revision in session.decided_revisions:
                raise ValueError("A decision is already recorded for this revision")
            if session.state.legal_actions and action_id not in session.state.legal_actions:
                raise ValueError(f"Action is not legal in the current table state: {action_id}")
            try:
                chosen = session.match.solution.action(action_id)
            except KeyError as error:
                raise ValueError(f"Action is not covered by the matched solution: {action_id}") from error
            session.decided_revisions.add(revision)
            loss = session.match.solution.best_ev - chosen.ev
            return {
                "session_id": session_id,
                "revision": revision,
                "state_id": session.state.state_id,
                "action_id": action_id,
                "chosen_frequency": format(chosen.frequency, "f"),
                "chosen_ev_bb": format(chosen.ev, "f"),
                "best_ev_bb": format(session.match.solution.best_ev, "f"),
                "ev_loss_bb": format(loss, "f"),
                "grade": "excellent"
                if loss <= D("0.02")
                else "review"
                if loss <= D("0.15")
                else "major_leak",
            }

    def _session(self, session_id: str) -> LiveTableSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown live table session: {session_id}")
        return session

    @staticmethod
    def _validate_progression(
        session: LiveTableSession, state: LiveTableState
    ) -> dict[str, object]:
        return validate_state_transition(
            session.state, state, expected_table_id=session.table_id
        )

    @staticmethod
    def _best_match(
        state: LiveTableState, solutions: tuple[SolvedSpot, ...]
    ) -> LiveSolutionMatch | None:
        candidates: list[LiveSolutionMatch] = []
        for spot in solutions:
            key = spot.key
            if (
                key.game != state.game
                or key.players != state.players
                or key.hero_position != state.hero_position
                or key.rake_model != state.rake_model
                or key.utility_model != state.utility_model
            ):
                continue
            if not suit_isomorphic(
                key.board, key.hero_cards, state.board, state.hero_cards
            ):
                continue
            stack_error = abs(key.effective_stack_bb - state.effective_stack_bb)
            pot_error = abs(key.pot_bb - state.pot_bb)
            if stack_error > D("2") or pot_error > D("1"):
                continue
            history_exact = key.action_history == state.action_history
            history_suffix = bool(key.action_history) and state.action_history[
                -len(key.action_history) :
            ] == key.action_history
            score = D("1")
            score -= min(D("0.20"), stack_error * D("0.10"))
            score -= min(D("0.20"), pot_error * D("0.20"))
            if not history_exact:
                score -= D("0.10") if history_suffix else D("0.35")
            if score < D("0.55"):
                continue
            raw_exact = key.board == state.board and set(key.hero_cards) == set(
                state.hero_cards
            )
            confidence = (
                "exact"
                if stack_error <= D("0.01")
                and pot_error <= D("0.01")
                and history_exact
                else "close"
                if score >= D("0.80")
                else "approximate"
            )
            candidates.append(
                LiveSolutionMatch(
                    solution=spot,
                    confidence=confidence,
                    score=score,
                    stack_error_bb=stack_error,
                    pot_error_bb=pot_error,
                    history_exact=history_exact,
                    card_match="exact" if raw_exact else "suit_isomorphic",
                )
            )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: (row.score, row.solution.key.fingerprint),
        )

    def _response(
        self,
        session_id: str,
        state: LiveTableState,
        match: LiveSolutionMatch | None,
        transition: dict[str, object],
    ) -> dict[str, object]:
        pot_odds = (
            D("0")
            if state.to_call_bb == 0
            else state.to_call_bb / (state.pot_bb + state.to_call_bb)
        )
        response: dict[str, object] = {
            "session_id": session_id,
            "table_id": state.table_id,
            "last_revision": state.revision,
            "state_id": state.state_id,
            "status": "matched" if match is not None else "unmatched",
            "state": state.to_dict(),
            "math": {
                "spr": format(state.effective_stack_bb / state.pot_bb, "f"),
                "to_call_bb": format(state.to_call_bb, "f"),
                "call_break_even_equity": format(pot_odds, "f"),
                "pot_after_call_bb": format(
                    state.pot_bb + state.to_call_bb, "f"
                ),
            },
            "texture": analyze_board_texture(
                state.board, state.hero_cards
            ).to_dict(),
            "match": None,
            "warnings": [],
            "transition": transition,
            "strategy_route": None,
        }
        if match is None:
            response["warnings"] = [
                "No imported solution covers this exact card and configuration state."
            ]
            return response
        spot = match.solution
        legal = set(state.legal_actions)
        covered = [
            action
            for action in spot.actions
            if not legal or action.action in legal
        ]
        omitted = [
            action.action
            for action in spot.actions
            if legal and action.action not in legal
        ]
        warnings: list[str] = []
        if match.confidence != "exact":
            warnings.append(
                f"Using a {match.confidence} match; review stack, pot and action-history differences."
            )
        if omitted:
            warnings.append(
                "Solver actions outside the supplied legal-action set were omitted."
            )
        if not covered:
            warnings.append(
                "The matched solution has no action in the supplied legal-action set."
            )
        dominant = max(covered, key=lambda action: action.frequency) if covered else None
        max_ev = max(covered, key=lambda action: action.ev) if covered else None
        response["match"] = {
            "confidence": match.confidence,
            "score": format(match.score, "f"),
            "stack_error_bb": format(match.stack_error_bb, "f"),
            "pot_error_bb": format(match.pot_error_bb, "f"),
            "history_exact": match.history_exact,
            "card_match": match.card_match,
            "fingerprint": spot.key.fingerprint,
            "node_id": spot.node_id,
            "source": spot.source,
            "source_version": spot.source_version,
            "best_ev_bb": format(spot.best_ev, "f"),
            "dominant_action": None if dominant is None else dominant.action,
            "max_ev_action": None if max_ev is None else max_ev.action,
            "actions": [
                {
                    "action": action.action,
                    "label": _action_label(action.action),
                    "frequency": format(action.frequency, "f"),
                    "ev_bb": format(action.ev, "f"),
                }
                for action in covered
            ],
            "coverage": {
                "solved_actions": len(spot.actions),
                "legal_solved_actions": len(covered),
                "omitted_actions": omitted,
            },
        }
        response["strategy_route"] = self.strategy_selector.select(
            spot,
            state_id=state.state_id,
            revision=state.revision,
            legal_actions=state.legal_actions,
            latency_budget_ms=self.strategy_latency_budget_ms,
        )
        response["warnings"] = warnings
        return response
