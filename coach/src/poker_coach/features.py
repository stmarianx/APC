from __future__ import annotations

from dataclasses import dataclass, field

from .models import ActionKind, HandHistory, Street
from .profiles import PlayerProfile, Tendency


def position_map(hand: HandHistory) -> dict[str, str]:
    """Assign conventional positions from occupied seats and the button."""
    players_by_seat = {player.seat: player for player in hand.players}
    ordered_seats = sorted(players_by_seat)
    button_index = ordered_seats.index(hand.button_seat)
    clockwise = ordered_seats[button_index:] + ordered_seats[:button_index]
    labels_by_count = {
        2: ("BTN/SB", "BB"),
        3: ("BTN", "SB", "BB"),
        4: ("BTN", "SB", "BB", "UTG"),
        5: ("BTN", "SB", "BB", "UTG", "CO"),
        6: ("BTN", "SB", "BB", "UTG", "HJ", "CO"),
        7: ("BTN", "SB", "BB", "UTG", "MP", "HJ", "CO"),
        8: ("BTN", "SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO"),
        9: ("BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"),
    }
    labels = labels_by_count.get(len(clockwise))
    if labels is None:
        labels = tuple(f"SEAT_REL_{index}" for index in range(len(clockwise)))
    return {players_by_seat[seat].name: labels[index] for index, seat in enumerate(clockwise)}


@dataclass
class ProfileBook:
    profiles: dict[str, PlayerProfile] = field(default_factory=dict)

    def profile(self, player: str) -> PlayerProfile:
        if player not in self.profiles:
            self.profiles[player] = PlayerProfile(player)
        return self.profiles[player]

    def observe_hand(self, hand: HandHistory) -> None:
        positions = position_map(hand)
        preflop_actions = [
            action
            for action in hand.actions
            if action.street == Street.PREFLOP and action.kind in {
                ActionKind.FOLD,
                ActionKind.CHECK,
                ActionKind.CALL,
                ActionKind.BET,
                ActionKind.RAISE,
            }
        ]
        voluntary = {action.player for action in preflop_actions if action.kind in {ActionKind.CALL, ActionKind.BET, ActionKind.RAISE}}
        preflop_raisers = {action.player for action in preflop_actions if action.kind == ActionKind.RAISE}
        for player in hand.players:
            self.profile(player.name).observe(Tendency.VPIP, player.name in voluntary, position=positions[player.name])
            self.profile(player.name).observe(Tendency.PFR, player.name in preflop_raisers, position=positions[player.name])

        raises_seen = 0
        for action in preflop_actions:
            assert action.player is not None
            if raises_seen == 1:
                self.profile(action.player).observe(
                    Tendency.THREE_BET,
                    action.kind == ActionKind.RAISE,
                    position=positions[action.player],
                )
            elif raises_seen == 2:
                self.profile(action.player).observe(
                    Tendency.FOLD_TO_THREE_BET,
                    action.kind == ActionKind.FOLD,
                    position=positions[action.player],
                )
            if action.kind == ActionKind.RAISE:
                raises_seen += 1

        preflop_raise_actions = [action for action in preflop_actions if action.kind == ActionKind.RAISE]
        last_preflop_aggressor = preflop_raise_actions[-1].player if preflop_raise_actions else None
        flop_actions = [
            action
            for action in hand.actions
            if action.street == Street.FLOP and action.kind in {
                ActionKind.FOLD,
                ActionKind.CHECK,
                ActionKind.CALL,
                ActionKind.BET,
                ActionKind.RAISE,
            }
        ]
        cbet_index: int | None = None
        prior_bet = False
        if last_preflop_aggressor is not None:
            for index, action in enumerate(flop_actions):
                if action.kind in {ActionKind.BET, ActionKind.RAISE} and action.player != last_preflop_aggressor:
                    prior_bet = True
                if action.player == last_preflop_aggressor:
                    if not prior_bet:
                        self.profile(action.player).observe(
                            Tendency.FLOP_CBET,
                            action.kind == ActionKind.BET,
                            position=positions[action.player],
                        )
                        if action.kind == ActionKind.BET:
                            cbet_index = index
                    break

        if cbet_index is not None:
            for action in flop_actions[cbet_index + 1 :]:
                if action.player == last_preflop_aggressor:
                    continue
                assert action.player is not None
                self.profile(action.player).observe(
                    Tendency.FOLD_TO_FLOP_CBET,
                    action.kind == ActionKind.FOLD,
                    position=positions[action.player],
                )
                if action.kind in {ActionKind.BET, ActionKind.RAISE}:
                    break

        for action in hand.actions:
            if action.street not in (Street.FLOP, Street.TURN, Street.RIVER):
                continue
            if action.kind not in {
                ActionKind.FOLD,
                ActionKind.CHECK,
                ActionKind.CALL,
                ActionKind.BET,
                ActionKind.RAISE,
            }:
                continue
            assert action.player is not None
            self.profile(action.player).observe(
                Tendency.AGGRESSIVE_ACTION,
                action.kind in {ActionKind.BET, ActionKind.RAISE},
                position=positions[action.player],
            )

        preflop_folded = {
            action.player
            for action in hand.actions
            if action.street == Street.PREFLOP and action.kind == ActionKind.FOLD
        }
        showdown_players = {
            action.player
            for action in hand.actions
            if action.street == Street.SHOWDOWN and action.kind in {ActionKind.SHOW, ActionKind.MUCK}
        }
        if len(hand.board) >= 3:
            for player in hand.players:
                if player.name not in preflop_folded:
                    self.profile(player.name).observe(
                        Tendency.WENT_TO_SHOWDOWN,
                        player.name in showdown_players,
                        position=positions[player.name],
                    )
