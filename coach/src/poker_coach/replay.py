from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .models import ActionKind, Card, HandAction, HandHistory, Street


CHIP_ACTIONS = {
    ActionKind.POST_ANTE,
    ActionKind.POST_SMALL_BLIND,
    ActionKind.POST_BIG_BLIND,
    ActionKind.POST_DEAD_BLIND,
    ActionKind.CALL,
    ActionKind.BET,
    ActionKind.RAISE,
}
DECISION_ACTIONS = {
    ActionKind.FOLD,
    ActionKind.CHECK,
    ActionKind.CALL,
    ActionKind.BET,
    ActionKind.RAISE,
}


@dataclass(frozen=True)
class PlayerLedger:
    player: str
    stack: Decimal
    committed_total: Decimal
    committed_street: Decimal
    folded: bool
    awarded: Decimal


@dataclass(frozen=True)
class DecisionSnapshot:
    action_index: int
    street: Street
    actor: str
    observed_action: ActionKind
    pot: Decimal
    current_bet: Decimal
    to_call: Decimal
    call_amount: Decimal
    actor_stack: Decimal
    effective_stack: Decimal
    opponent_effective_stacks: tuple[tuple[str, Decimal], ...]
    board: tuple[Card, ...]
    players: tuple[PlayerLedger, ...]


@dataclass(frozen=True)
class ReplayResult:
    decisions: tuple[DecisionSnapshot, ...]
    players: tuple[PlayerLedger, ...]
    committed_pot: Decimal
    awarded_total: Decimal
    rake: Decimal
    reconciliation_error: Decimal


@dataclass
class _MutableLedger:
    stack: Decimal
    committed_total: Decimal = Decimal("0")
    committed_street: Decimal = Decimal("0")
    folded: bool = False
    awarded: Decimal = Decimal("0")


def _board_for(hand: HandHistory, street: Street) -> tuple[Card, ...]:
    if street == Street.FLOP:
        return hand.board[:3]
    if street == Street.TURN:
        return hand.board[:4]
    if street >= Street.RIVER:
        return hand.board[:5]
    return ()


class HandReplayer:
    def replay(self, hand: HandHistory) -> ReplayResult:
        ledger = {player.name: _MutableLedger(player.starting_stack) for player in hand.players}
        decisions: list[DecisionSnapshot] = []
        current_street = Street.PREFLOP

        for action in hand.actions:
            if action.street != current_street:
                if action.street in (Street.FLOP, Street.TURN, Street.RIVER, Street.SHOWDOWN, Street.SUMMARY):
                    for state in ledger.values():
                        state.committed_street = Decimal("0")
                current_street = action.street

            if action.player is not None and action.player not in ledger:
                raise ValueError(f"Action references unknown player: {action.player}")

            if action.kind in DECISION_ACTIONS and action.player is not None:
                decisions.append(self._snapshot(hand, action, ledger))

            if action.kind in CHIP_ACTIONS:
                assert action.player is not None
                state = ledger[action.player]
                if action.kind == ActionKind.RAISE:
                    assert action.to_amount is not None
                    added = action.to_amount - state.committed_street
                else:
                    added = action.amount
                if added < 0:
                    raise ValueError(f"Negative contribution implied by action: {action.raw}")
                if added > state.stack:
                    raise ValueError(f"Action exceeds remaining stack: {action.raw}")
                state.stack -= added
                state.committed_total += added
                state.committed_street += added
            elif action.kind == ActionKind.RETURN:
                assert action.player is not None
                state = ledger[action.player]
                if action.amount > state.committed_total or action.amount > state.committed_street:
                    raise ValueError(f"Returned amount exceeds contribution: {action.raw}")
                state.stack += action.amount
                state.committed_total -= action.amount
                state.committed_street -= action.amount
            elif action.kind == ActionKind.FOLD:
                assert action.player is not None
                ledger[action.player].folded = True

        awards = {name: Decimal("0") for name in ledger}
        for award in hand.awards:
            if award.player not in awards:
                raise ValueError(f"Award references unknown player: {award.player}")
            awards[award.player] += award.amount
            ledger[award.player].awarded += award.amount

        final_players = self._freeze_players(ledger)
        committed = sum((state.committed_total for state in ledger.values()), Decimal("0"))
        awarded = sum(awards.values(), Decimal("0"))
        rake = hand.rake or Decimal("0")
        return ReplayResult(
            decisions=tuple(decisions),
            players=final_players,
            committed_pot=committed,
            awarded_total=awarded,
            rake=rake,
            reconciliation_error=committed - awarded - rake,
        )

    def _snapshot(
        self,
        hand: HandHistory,
        action: HandAction,
        ledger: dict[str, _MutableLedger],
    ) -> DecisionSnapshot:
        assert action.player is not None
        actor = ledger[action.player]
        current_bet = max((state.committed_street for state in ledger.values()), default=Decimal("0"))
        to_call = max(Decimal("0"), current_bet - actor.committed_street)
        live_opponents = [
            (name, state)
            for name, state in ledger.items()
            if name != action.player and not state.folded and state.stack + state.committed_street > 0
        ]
        opponent_effective = tuple(
            sorted((name, min(actor.stack, state.stack)) for name, state in live_opponents)
        )
        effective_stack = max((amount for _, amount in opponent_effective), default=Decimal("0"))
        return DecisionSnapshot(
            action_index=action.index,
            street=action.street,
            actor=action.player,
            observed_action=action.kind,
            pot=sum((state.committed_total for state in ledger.values()), Decimal("0")),
            current_bet=current_bet,
            to_call=to_call,
            call_amount=min(to_call, actor.stack),
            actor_stack=actor.stack,
            effective_stack=effective_stack,
            opponent_effective_stacks=opponent_effective,
            board=_board_for(hand, action.street),
            players=self._freeze_players(ledger),
        )

    @staticmethod
    def _freeze_players(ledger: dict[str, _MutableLedger]) -> tuple[PlayerLedger, ...]:
        return tuple(
            PlayerLedger(
                player=name,
                stack=state.stack,
                committed_total=state.committed_total,
                committed_street=state.committed_street,
                folded=state.folded,
                awarded=state.awarded,
            )
            for name, state in sorted(ledger.items())
        )

