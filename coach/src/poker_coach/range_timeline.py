from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable

from .features import position_map
from .isomorphism import suit_isomorphic
from .matching import _game_id, _normalized_history
from .models import ActionKind, Card, HandAction, HandHistory, Street
from .range_inference import combo_label, condition_solution_range
from .range_strategy import public_node_fingerprint
from .replay import DecisionSnapshot, HandReplayer
from .solutions import SolvedSpot


D = Decimal


@dataclass(frozen=True)
class OpponentDecisionContext:
    hand_id: str
    action_index: int
    street: Street
    actor: str
    actor_position: str
    game: str
    players: int
    effective_stack_bb: Decimal
    pot_bb: Decimal
    board: tuple[Card, ...]
    action_history: tuple[str, ...]
    snapshot: DecisionSnapshot
    action: HandAction


@dataclass(frozen=True)
class PublicRangeMatch:
    context: OpponentDecisionContext
    spots: tuple[SolvedSpot, ...]
    public_fingerprint: str
    confidence: str
    score: Decimal
    stack_error_bb: Decimal
    pot_error_bb: Decimal
    history_exact: bool
    card_match: str
    observed_action: str | None
    action_size_error: Decimal | None


def opponent_decision_contexts(hand: HandHistory) -> tuple[OpponentDecisionContext, ...]:
    """Reconstruct public decision states for every non-Hero action."""

    if hand.hero is None or hand.big_blind is None or hand.big_blind <= 0:
        return ()
    positions = position_map(hand)
    replay = HandReplayer().replay(hand)
    actions = {action.index: action for action in hand.actions}
    rows: list[OpponentDecisionContext] = []
    for snapshot in replay.decisions:
        if snapshot.actor == hand.hero:
            continue
        active_players = sum(1 for player in snapshot.players if not player.folded)
        rows.append(
            OpponentDecisionContext(
                hand_id=hand.hand_id,
                action_index=snapshot.action_index,
                street=snapshot.street,
                actor=snapshot.actor,
                actor_position=positions[snapshot.actor],
                game=_game_id(hand),
                players=active_players,
                effective_stack_bb=snapshot.effective_stack / hand.big_blind,
                pot_bb=snapshot.pot / hand.big_blind,
                board=snapshot.board,
                action_history=_normalized_history(
                    hand, snapshot.action_index, positions, hand.big_blind
                ),
                snapshot=snapshot,
                action=actions[snapshot.action_index],
            )
        )
    return tuple(rows)


def _group_private_nodes(
    solutions: Iterable[SolvedSpot],
) -> dict[str, tuple[SolvedSpot, ...]]:
    grouped: dict[str, list[SolvedSpot]] = defaultdict(list)
    for spot in solutions:
        if len(spot.key.hero_cards) == 2:
            grouped[public_node_fingerprint(spot)].append(spot)
    return {
        fingerprint: tuple(
            sorted(spots, key=lambda spot: (combo_label(spot), spot.key.fingerprint))
        )
        for fingerprint, spots in grouped.items()
    }


def _observed_action(
    context: OpponentDecisionContext,
    spots: tuple[SolvedSpot, ...],
) -> tuple[str | None, Decimal | None]:
    action_ids = sorted({action.action for spot in spots for action in spot.actions})
    action = context.action
    if action.kind in {ActionKind.FOLD, ActionKind.CHECK, ActionKind.CALL}:
        action_id = action.kind.value
        return (action_id, D("0")) if action_id in action_ids else (None, None)
    if action.kind == ActionKind.BET and context.snapshot.pot > 0:
        prefix = "bet:"
        observed_size = action.amount / context.snapshot.pot
        tolerance = D("0.08")
    elif action.kind == ActionKind.RAISE and action.to_amount is not None:
        prefix = "raise_to:"
        blind = (
            context.snapshot.pot / context.pot_bb
            if context.pot_bb > 0
            else D("1")
        )
        observed_size = action.to_amount / blind
        tolerance = D("0.5")
    else:
        return None, None
    candidates: list[tuple[Decimal, str]] = []
    for action_id in action_ids:
        if not action_id.startswith(prefix):
            continue
        try:
            size = D(action_id.split(":", 1)[1])
        except (InvalidOperation, IndexError):
            continue
        candidates.append((abs(size - observed_size), action_id))
    if not candidates:
        return None, None
    error, action_id = min(candidates, key=lambda row: (row[0], row[1]))
    return (action_id, error) if error <= tolerance else (None, error)


def _candidate(
    context: OpponentDecisionContext,
    fingerprint: str,
    spots: tuple[SolvedSpot, ...],
) -> PublicRangeMatch | None:
    representative = spots[0]
    key = representative.key
    if (
        key.game != context.game
        or key.players != context.players
        or key.hero_position != context.actor_position
    ):
        return None
    if not suit_isomorphic(key.board, (), context.board, ()):
        return None
    stack_error = abs(key.effective_stack_bb - context.effective_stack_bb)
    pot_error = abs(key.pot_bb - context.pot_bb)
    if stack_error > D("2") or pot_error > D("1"):
        return None
    history_exact = key.action_history == context.action_history
    history_suffix = bool(key.action_history) and context.action_history[
        -len(key.action_history) :
    ] == key.action_history
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
    observed, size_error = _observed_action(context, spots)
    return PublicRangeMatch(
        context=context,
        spots=spots,
        public_fingerprint=fingerprint,
        confidence=confidence,
        score=score,
        stack_error_bb=stack_error,
        pot_error_bb=pot_error,
        history_exact=history_exact,
        card_match="exact" if key.board == context.board else "suit_isomorphic",
        observed_action=observed,
        action_size_error=size_error,
    )


def _fmt(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _match_dict(match: PublicRangeMatch) -> dict[str, object]:
    representative = match.spots[0]
    return {
        "public_fingerprint": match.public_fingerprint,
        "confidence": match.confidence,
        "score": _fmt(match.score),
        "stack_error_bb": _fmt(match.stack_error_bb),
        "pot_error_bb": _fmt(match.pot_error_bb),
        "history_exact": match.history_exact,
        "card_match": match.card_match,
        "action_size_error": _fmt(match.action_size_error),
        "source": representative.source,
        "source_version": representative.source_version,
        "private_nodes": len(match.spots),
    }


def _carry_prior(
    previous: dict[str, object] | None,
    spots: tuple[SolvedSpot, ...],
) -> tuple[dict[str, object] | None, dict[str, object]]:
    representative = spots[0]
    labels = {combo_label(spot) for spot in spots}
    if previous is None:
        return None, {
            "mode": "uniform_reset",
            "reason": "first_compatible_action",
        }
    if (
        previous["source"] != representative.source
        or previous["source_version"] != representative.source_version
    ):
        return None, {
            "mode": "uniform_reset",
            "reason": "solver_provenance_changed",
        }
    previous_weights = previous["weights"]
    assert isinstance(previous_weights, dict)
    if set(previous_weights) != labels:
        return None, {
            "mode": "uniform_reset",
            "reason": "exact_combo_coverage_changed",
        }
    return dict(previous_weights), {
        "mode": "posterior_carried",
        "reason": "identical_exact_combo_coverage_and_provenance",
        "from_action_index": previous["action_index"],
        "unmatched_actions_skipped": previous.get("unmatched_gap", 0),
    }


def build_opponent_range_timelines(
    hands: Iterable[HandHistory],
    solutions: Iterable[SolvedSpot],
) -> dict[str, object]:
    """Condition imported exact-combo ranges on saved opponent actions."""

    hand_rows = tuple(hands)
    groups = _group_private_nodes(solutions)
    timelines: list[dict[str, object]] = []
    opponent_decisions = 0
    public_matches = 0
    conditioned_actions = 0
    conditioning_failures = 0

    for hand in hand_rows:
        contexts = opponent_decision_contexts(hand)
        opponent_decisions += len(contexts)
        by_actor: dict[str, list[OpponentDecisionContext]] = defaultdict(list)
        for context in contexts:
            by_actor[context.actor].append(context)
        positions = position_map(hand)
        for actor, actor_contexts in sorted(by_actor.items()):
            events: list[dict[str, object]] = []
            previous: dict[str, object] | None = None
            for context in sorted(actor_contexts, key=lambda row: row.action_index):
                candidates = [
                    candidate
                    for fingerprint, spots in groups.items()
                    if (candidate := _candidate(context, fingerprint, spots))
                    is not None
                ]
                if not candidates:
                    if previous is not None:
                        previous["unmatched_gap"] = int(
                            previous.get("unmatched_gap", 0)
                        ) + 1
                    continue
                match = max(
                    candidates,
                    key=lambda row: (row.score, row.public_fingerprint),
                )
                public_matches += 1
                prior_weights, transition = _carry_prior(previous, match.spots)
                event: dict[str, object] = {
                    "hand_id": hand.hand_id,
                    "opponent": actor,
                    "position": context.actor_position,
                    "action_index": context.action_index,
                    "street": context.street.name.lower(),
                    "board": [str(card) for card in context.board],
                    "pot_bb": _fmt(context.pot_bb),
                    "effective_stack_bb": _fmt(context.effective_stack_bb),
                    "observed_raw": context.action.raw,
                    "observed_action": match.observed_action,
                    "prior_transition": transition,
                    "match": _match_dict(match),
                }
                if match.observed_action is None:
                    event["status"] = "action_not_covered"
                    event["conditioning_error"] = (
                        "The public state matched, but the observed action or size is not "
                        "covered by the imported abstraction."
                    )
                    previous = None
                    conditioning_failures += 1
                else:
                    try:
                        posterior = condition_solution_range(
                            match.spots,
                            match.observed_action,
                            prior_weights=prior_weights,
                        )
                    except ValueError as error:
                        event["status"] = "conditioning_failed"
                        event["conditioning_error"] = str(error)
                        previous = None
                        conditioning_failures += 1
                    else:
                        event["status"] = "conditioned"
                        event["posterior"] = posterior
                        weights = {
                            str(row["combo"]): row["posterior"]
                            for row in posterior["combos"]
                        }
                        previous = {
                            "source": match.spots[0].source,
                            "source_version": match.spots[0].source_version,
                            "action_index": context.action_index,
                            "weights": weights,
                            "unmatched_gap": 0,
                        }
                        conditioned_actions += 1
                events.append(event)
            if events:
                timelines.append(
                    {
                        "hand_id": hand.hand_id,
                        "opponent": actor,
                        "position": positions[actor],
                        "events": events,
                    }
                )

    return {
        "schema_version": "1.0.0",
        "hands": len(hand_rows),
        "solver_public_nodes": len(groups),
        "opponent_decisions": opponent_decisions,
        "public_state_matches": public_matches,
        "conditioned_actions": conditioned_actions,
        "conditioning_failures": conditioning_failures,
        "unmatched_actions": opponent_decisions - public_matches,
        "coverage": format(
            D(public_matches) / D(opponent_decisions)
            if opponent_decisions
            else D("0"),
            "f",
        ),
        "timelines": timelines,
        "caveat": (
            "Posteriors cover only imported exact private combos. Matching tolerances and "
            "solver frequencies do not turn missing combos into population reads. Actions "
            "without compatible public nodes are skipped and do not update the posterior."
        ),
    }
