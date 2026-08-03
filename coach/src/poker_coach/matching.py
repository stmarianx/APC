from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable

from .features import position_map
from .models import ActionKind, Card, HandAction, HandHistory, Street
from .replay import DecisionSnapshot, HandReplayer
from .solutions import SolvedSpot
from .isomorphism import suit_isomorphic


D = Decimal


def _compact(value: Decimal) -> str:
    if value == value.to_integral():
        return str(value.to_integral())
    return format(value.normalize(), "f")


def _game_id(hand: HandHistory) -> str:
    if "hold" in hand.game.lower() and hand.limit == "no_limit":
        return "holdem_no_limit"
    return f"{hand.game.lower().replace(' ', '_')}_{hand.limit}"


def _hero_cards(hand: HandHistory) -> tuple[Card, ...]:
    if hand.hero is None:
        return ()
    for holding in hand.hole_cards:
        if holding.player == hand.hero:
            return holding.cards
    return ()


def _normalized_history(
    hand: HandHistory,
    before_index: int,
    positions: dict[str, str],
    big_blind: Decimal,
) -> tuple[str, ...]:
    rows: list[str] = []
    for action in hand.actions:
        if action.index >= before_index:
            break
        if action.kind not in {
            ActionKind.FOLD,
            ActionKind.CHECK,
            ActionKind.CALL,
            ActionKind.BET,
            ActionKind.RAISE,
        } or action.player is None:
            continue
        actor = positions[action.player]
        if action.kind == ActionKind.BET:
            rows.append(f"{actor} bet:{_compact(action.amount / big_blind)}")
        elif action.kind == ActionKind.RAISE:
            assert action.to_amount is not None
            rows.append(f"{actor} raise_to:{_compact(action.to_amount / big_blind)}")
        else:
            rows.append(f"{actor} {action.kind.value}")
    return tuple(rows)


@dataclass(frozen=True)
class DecisionContext:
    hand_id: str
    action_index: int
    street: Street
    hero: str
    game: str
    players: int
    hero_position: str
    effective_stack_bb: Decimal
    pot_bb: Decimal
    board: tuple[Card, ...]
    hero_cards: tuple[Card, ...]
    action_history: tuple[str, ...]
    snapshot: DecisionSnapshot
    action: HandAction


@dataclass(frozen=True)
class DecisionSolutionMatch:
    context: DecisionContext
    solution: SolvedSpot
    confidence: str
    score: Decimal
    stack_error_bb: Decimal
    pot_error_bb: Decimal
    history_exact: bool
    card_match: str
    matched_action: str | None
    action_size_error: Decimal | None

    @property
    def ev_loss_bb(self) -> Decimal | None:
        if self.matched_action is None:
            return None
        return self.solution.ev_loss(self.matched_action)

    def to_dict(self) -> dict[str, object]:
        observed = self.context.action.kind.value
        return {
            "hand_id": self.context.hand_id,
            "action_index": self.context.action_index,
            "street": self.context.street.name.lower(),
            "hero_position": self.context.hero_position,
            "hero_cards": [str(card) for card in self.context.hero_cards],
            "board": [str(card) for card in self.context.board],
            "pot_bb": format(self.context.pot_bb, "f"),
            "effective_stack_bb": format(self.context.effective_stack_bb, "f"),
            "observed_action_kind": observed,
            "matched_action": self.matched_action,
            "action_size_error": None if self.action_size_error is None else format(self.action_size_error, "f"),
            "match": {
                "confidence": self.confidence,
                "score": format(self.score, "f"),
                "stack_error_bb": format(self.stack_error_bb, "f"),
                "pot_error_bb": format(self.pot_error_bb, "f"),
                "history_exact": self.history_exact,
                "card_match": self.card_match,
            },
            "solution": {
                "fingerprint": self.solution.key.fingerprint,
                "source": self.solution.source,
                "source_version": self.solution.source_version,
                "best_ev_bb": format(self.solution.best_ev, "f"),
                "ev_loss_bb": None if self.ev_loss_bb is None else format(self.ev_loss_bb, "f"),
                "actions": [
                    {
                        "action": action.action,
                        "frequency": format(action.frequency, "f"),
                        "ev_bb": format(action.ev, "f"),
                    }
                    for action in self.solution.actions
                ],
            },
        }


class DecisionSolutionMatcher:
    def contexts(self, hand: HandHistory) -> tuple[DecisionContext, ...]:
        if hand.hero is None or hand.big_blind is None or hand.big_blind <= 0:
            return ()
        positions = position_map(hand)
        holding = _hero_cards(hand)
        replay = HandReplayer().replay(hand)
        actions = {action.index: action for action in hand.actions}
        rows: list[DecisionContext] = []
        for snapshot in replay.decisions:
            if snapshot.actor != hand.hero:
                continue
            active_players = sum(1 for player in snapshot.players if not player.folded)
            rows.append(
                DecisionContext(
                    hand_id=hand.hand_id,
                    action_index=snapshot.action_index,
                    street=snapshot.street,
                    hero=hand.hero,
                    game=_game_id(hand),
                    players=active_players,
                    hero_position=positions[hand.hero],
                    effective_stack_bb=snapshot.effective_stack / hand.big_blind,
                    pot_bb=snapshot.pot / hand.big_blind,
                    board=snapshot.board,
                    hero_cards=holding,
                    action_history=_normalized_history(
                        hand, snapshot.action_index, positions, hand.big_blind
                    ),
                    snapshot=snapshot,
                    action=actions[snapshot.action_index],
                )
            )
        return tuple(rows)

    def match_hand(
        self,
        hand: HandHistory,
        solutions: Iterable[SolvedSpot],
    ) -> tuple[DecisionSolutionMatch, ...]:
        solution_rows = tuple(solutions)
        matches: list[DecisionSolutionMatch] = []
        for context in self.contexts(hand):
            candidates: list[DecisionSolutionMatch] = []
            for solution in solution_rows:
                candidate = self._candidate(context, solution)
                if candidate is not None:
                    candidates.append(candidate)
            if candidates:
                matches.append(
                    max(candidates, key=lambda match: (match.score, match.solution.key.fingerprint))
                )
        return tuple(matches)

    def _candidate(
        self,
        context: DecisionContext,
        solution: SolvedSpot,
    ) -> DecisionSolutionMatch | None:
        key = solution.key
        if (
            key.game != context.game
            or key.players != context.players
            or key.hero_position != context.hero_position
        ):
            return None
        if not suit_isomorphic(
            key.board,
            key.hero_cards,
            context.board,
            context.hero_cards,
        ):
            return None
        raw_cards_exact = key.board == context.board and set(key.hero_cards) == set(
            context.hero_cards
        )
        stack_error = abs(key.effective_stack_bb - context.effective_stack_bb)
        pot_error = abs(key.pot_bb - context.pot_bb)
        if stack_error > D("2") or pot_error > D("1"):
            return None
        history_exact = key.action_history == context.action_history
        history_suffix = bool(key.action_history) and context.action_history[-len(key.action_history) :] == key.action_history
        score = D("1")
        score -= min(D("0.20"), stack_error * D("0.10"))
        score -= min(D("0.20"), pot_error * D("0.20"))
        if not history_exact:
            score -= D("0.10") if history_suffix else D("0.35")
        if score < D("0.55"):
            return None
        if stack_error <= D("0.01") and pot_error <= D("0.01") and history_exact:
            confidence = "exact"
        elif score >= D("0.80"):
            confidence = "close"
        else:
            confidence = "approximate"
        matched_action, action_error = self._match_action(context, solution)
        return DecisionSolutionMatch(
            context=context,
            solution=solution,
            confidence=confidence,
            score=score,
            stack_error_bb=stack_error,
            pot_error_bb=pot_error,
            history_exact=history_exact,
            card_match="exact" if raw_cards_exact else "suit_isomorphic",
            matched_action=matched_action,
            action_size_error=action_error,
        )

    @staticmethod
    def _match_action(
        context: DecisionContext,
        solution: SolvedSpot,
    ) -> tuple[str | None, Decimal | None]:
        action = context.action
        if action.kind in {ActionKind.FOLD, ActionKind.CHECK, ActionKind.CALL}:
            action_id = action.kind.value
            try:
                solution.action(action_id)
                return action_id, D("0")
            except KeyError:
                return None, None
        prefix: str
        observed_size: Decimal
        tolerance: Decimal
        if action.kind == ActionKind.BET and context.snapshot.pot > 0:
            prefix = "bet:"
            observed_size = action.amount / context.snapshot.pot
            tolerance = D("0.08")
        elif action.kind == ActionKind.RAISE and action.to_amount is not None:
            prefix = "raise_to:"
            # Context values are normalized to BB; recover the blind from pot ratios.
            blind = context.snapshot.pot / context.pot_bb if context.pot_bb > 0 else D("1")
            observed_size = action.to_amount / blind
            tolerance = D("0.5")
        else:
            return None, None
        candidates: list[tuple[Decimal, str]] = []
        for solved_action in solution.actions:
            if not solved_action.action.startswith(prefix):
                continue
            try:
                solved_size = D(solved_action.action.split(":", 1)[1])
            except (InvalidOperation, IndexError):
                continue
            candidates.append((abs(solved_size - observed_size), solved_action.action))
        if not candidates:
            return None, None
        error, action_id = min(candidates, key=lambda row: (row[0], row[1]))
        return (action_id, error) if error <= tolerance else (None, error)


def _drill_title(match: DecisionSolutionMatch) -> str:
    street = match.context.street.name.title()
    if match.context.action.kind in {ActionKind.BET, ActionKind.RAISE}:
        return f"{street} sizing and range construction"
    if match.context.action.kind == ActionKind.CALL:
        return f"{street} bluff-catching and pot odds"
    return f"{street} range decision"


def build_drill_queue(matches: Iterable[DecisionSolutionMatch]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    confidence_weight = {"exact": D("1"), "close": D("0.8"), "approximate": D("0.5")}
    for match in matches:
        loss = match.ev_loss_bb
        if loss is None or loss <= 0:
            continue
        identity = f"{match.context.hand_id}:{match.context.action_index}:{match.solution.key.fingerprint}"
        strategy = sorted(
            match.solution.actions,
            key=lambda action: (action.ev, action.frequency, action.action),
            reverse=True,
        )
        rows.append(
            {
                "drill_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
                "title": _drill_title(match),
                "priority": format(loss * confidence_weight[match.confidence], "f"),
                "ev_loss_bb": format(loss, "f"),
                "confidence": match.confidence,
                "hand_id": match.context.hand_id,
                "action_index": match.context.action_index,
                "street": match.context.street.name.lower(),
                "hero_position": match.context.hero_position,
                "hero_cards": [str(card) for card in match.context.hero_cards],
                "board": [str(card) for card in match.context.board],
                "pot_bb": format(match.context.pot_bb, "f"),
                "effective_stack_bb": format(match.context.effective_stack_bb, "f"),
                "observed_action": match.matched_action,
                "best_actions": [
                    {
                        "action": action.action,
                        "frequency": format(action.frequency, "f"),
                        "ev_bb": format(action.ev, "f"),
                    }
                    for action in strategy
                ],
                "solution_fingerprint": match.solution.key.fingerprint,
                "source": match.solution.source,
                "source_version": match.solution.source_version,
            }
        )
    return sorted(rows, key=lambda row: (D(str(row["priority"])), str(row["drill_id"])), reverse=True)


def analyze_with_solutions(
    hands: Iterable[HandHistory],
    solutions: Iterable[SolvedSpot],
) -> dict[str, object]:
    hand_rows = tuple(hands)
    solution_rows = tuple(solutions)
    matcher = DecisionSolutionMatcher()
    matches = tuple(
        match
        for hand in hand_rows
        for match in matcher.match_hand(hand, solution_rows)
    )
    hero_decisions = sum(len(matcher.contexts(hand)) for hand in hand_rows)
    measured = [match for match in matches if match.ev_loss_bb is not None]
    total_loss = sum((match.ev_loss_bb or D("0") for match in measured), D("0"))
    by_street: dict[str, dict[str, object]] = {}
    for match in measured:
        street = match.context.street.name.lower()
        row = by_street.setdefault(street, {"decisions": 0, "ev_loss_bb": D("0")})
        row["decisions"] = int(row["decisions"]) + 1
        row["ev_loss_bb"] = D(str(row["ev_loss_bb"])) + (match.ev_loss_bb or D("0"))
    worst = max(measured, key=lambda match: match.ev_loss_bb or D("0"), default=None)
    leak_summary = {
        "measured_decisions": len(measured),
        "total_ev_loss_bb": format(total_loss, "f"),
        "average_ev_loss_bb": format(total_loss / len(measured), "f") if measured else None,
        "worst_decision": None if worst is None else {
            "hand_id": worst.context.hand_id,
            "action_index": worst.context.action_index,
            "street": worst.context.street.name.lower(),
            "ev_loss_bb": format(worst.ev_loss_bb or D("0"), "f"),
            "observed_action": worst.matched_action,
        },
        "by_street": {
            street: {"decisions": row["decisions"], "ev_loss_bb": format(D(str(row["ev_loss_bb"])), "f")}
            for street, row in sorted(by_street.items())
        },
        "confidence_counts": {
            confidence: sum(1 for match in matches if match.confidence == confidence)
            for confidence in ("exact", "close", "approximate")
        },
        "coverage": format(D(len(matches)) / D(hero_decisions), "f") if hero_decisions else "0",
    }
    return {
        "schema_version": "0.1.0",
        "hands": len(hand_rows),
        "solutions_available": len(solution_rows),
        "hero_decisions": hero_decisions,
        "matched_decisions": len(matches),
        "unmatched_decisions": hero_decisions - len(matches),
        "matches": [match.to_dict() for match in matches],
        "drills": build_drill_queue(matches),
        "leak_summary": leak_summary,
    }
