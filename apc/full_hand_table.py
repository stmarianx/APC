from __future__ import annotations

import hashlib
import json
import random
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from apc.deadline import ActionCommand


D = Decimal
PLAYER_NAMES = ("Hero", "Villain")
STREETS = ("preflop", "flop", "turn", "river")
BOARD_COUNTS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def _coach_types() -> tuple[object, object]:
    try:
        from poker_coach.equity import best_hand_rank, full_deck
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        coach_source = root / "coach" / "src"
        if str(coach_source) not in sys.path:
            sys.path.insert(0, str(coach_source))
        from poker_coach.equity import best_hand_rank, full_deck
    return best_hand_rank, full_deck


def _decimal(value: object, name: str) -> Decimal:
    try:
        result = D(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a decimal number") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _bb(value: Decimal) -> str:
    normalized = value.normalize()
    return "0" if normalized == 0 else format(normalized, "f")


def _fingerprint(value: object) -> str:
    material = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class HeadsUpVirtualHand:
    """Deterministic no-rake heads-up no-limit virtual-chip hand engine."""

    def __init__(
        self,
        *,
        seed: int,
        button_player: int = 0,
        starting_stack_bb: object = "100",
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if button_player not in (0, 1):
            raise ValueError("button_player must be 0 or 1")
        self.seed = seed
        self.button = button_player
        self.starting_stack = _decimal(starting_stack_bb, "starting_stack_bb")
        if self.starting_stack < D("2"):
            raise ValueError("starting_stack_bb must be at least 2")
        _, full_deck = _coach_types()
        deck = list(full_deck())
        random.Random(seed).shuffle(deck)
        self.hole_cards = ((deck[0], deck[1]), (deck[2], deck[3]))
        self.runout = tuple(deck[4:9])
        self.hand_id = _fingerprint(
            {"kind": "heads_up_virtual_hand_v1", "seed": seed, "button": button_player}
        )[:24]
        self.stacks = [self.starting_stack, self.starting_stack]
        self.street_contributions = [D("0"), D("0")]
        self.total_contributions = [D("0"), D("0")]
        self.street = "preflop"
        self.board: tuple[object, ...] = ()
        self.history: list[str] = []
        self.acted: set[int] = set()
        self.current_bet = D("0")
        self.last_full_raise = D("1")
        self.next_actor: int | None = self.button
        self.terminal = False
        self.terminal_reason: str | None = None
        self.winner: int | None = None
        self.final_pot = D("0")
        self.rewards = [D("0"), D("0")]
        self.revision = 0
        self.showdown_revealed = False
        self._post_blind(self.button, D("0.5"))
        self._post_blind(1 - self.button, D("1"))
        self.current_bet = D("1")

    def _post_blind(self, player: int, amount: Decimal) -> None:
        committed = min(amount, self.stacks[player])
        self.stacks[player] -= committed
        self.street_contributions[player] += committed
        self.total_contributions[player] += committed

    def _position(self, player: int) -> str:
        return "BTN" if player == self.button else "BB"

    @property
    def pot(self) -> Decimal:
        return D("0") if self.terminal else sum(self.total_contributions, D("0"))

    def _to_call(self, player: int) -> Decimal:
        return max(D("0"), self.current_bet - self.street_contributions[player])

    def _max_to(self, player: int) -> Decimal:
        opponent = 1 - player
        own_capacity = self.street_contributions[player] + self.stacks[player]
        opponent_capacity = self.street_contributions[opponent] + self.stacks[opponent]
        return min(own_capacity, opponent_capacity)

    def legal_action_buttons(self) -> list[dict[str, str]]:
        if self.terminal or self.next_actor is None:
            return []
        player = self.next_actor
        to_call = self._to_call(player)
        maximum = self._max_to(player)
        buttons: list[dict[str, str]] = []
        if to_call > 0:
            buttons.extend(
                [
                    {"action": "fold"},
                    {"action": "call", "amount_bb": _bb(min(to_call, self.stacks[player]))},
                ]
            )
        else:
            buttons.append({"action": "check"})
        if maximum > self.current_bet and self.stacks[player] > 0:
            if self.current_bet == 0:
                if maximum >= D("1"):
                    buttons.append(
                        {"action": "bet", "minimum_to_bb": "1", "maximum_to_bb": _bb(maximum)}
                    )
            else:
                minimum = self.current_bet + self.last_full_raise
                if maximum >= minimum:
                    buttons.append(
                        {
                            "action": "raise",
                            "minimum_to_bb": _bb(minimum),
                            "maximum_to_bb": _bb(maximum),
                        }
                    )
            buttons.append({"action": "all_in", "to_amount_bb": _bb(maximum)})
        return buttons

    def observation(self) -> dict[str, object]:
        next_actor = self.next_actor
        payload = {
            "schema_version": "1.0.0",
            "environment": "controlled_virtual_chips",
            "scope": "complete_heads_up_hand",
            "hand_id": self.hand_id,
            "revision": self.revision,
            "units": "BB",
            "game": "holdem_no_limit",
            "street": self.street,
            "button_position": self._position(self.button),
            "hero_position": self._position(0),
            "hero_cards": [str(card) for card in self.hole_cards[0]],
            "opponent_cards": [str(card) for card in self.hole_cards[1]]
            if self.showdown_revealed
            else None,
            "board": [str(card) for card in self.board],
            "stacks_bb": {PLAYER_NAMES[index]: _bb(stack) for index, stack in enumerate(self.stacks)},
            "pot_bb": _bb(self.pot),
            "street_contributions_bb": {
                PLAYER_NAMES[index]: _bb(value)
                for index, value in enumerate(self.street_contributions)
            },
            "to_call_bb": "0" if next_actor is None else _bb(self._to_call(next_actor)),
            "next_actor": None if next_actor is None else PLAYER_NAMES[next_actor],
            "next_actor_position": None if next_actor is None else self._position(next_actor),
            "action_buttons": self.legal_action_buttons(),
            "action_history": list(self.history),
            "terminal": self.terminal,
            "terminal_reason": self.terminal_reason,
            "provider": {
                "internal_virtual_chips": True,
                "external_actuation": False,
                "screen_or_input_control": False,
                "rake_bb": "0",
            },
        }
        payload["state_fingerprint"] = _fingerprint(payload)
        return payload

    def _commit_to(self, player: int, target: Decimal) -> Decimal:
        if target < self.street_contributions[player]:
            raise ValueError("target contribution cannot decrease")
        amount = target - self.street_contributions[player]
        if amount > self.stacks[player]:
            raise ValueError("action exceeds player stack")
        self.stacks[player] -= amount
        self.street_contributions[player] += amount
        self.total_contributions[player] += amount
        return amount

    def step(self, command: ActionCommand | dict[str, object]) -> dict[str, object]:
        if self.terminal or self.next_actor is None:
            raise ValueError("virtual hand is already terminal")
        if isinstance(command, dict):
            command = ActionCommand(
                str(command.get("action")),
                amount_bb=None if command.get("amount_bb") is None else str(command["amount_bb"]),
                to_amount_bb=None if command.get("to_amount_bb") is None else str(command["to_amount_bb"]),
            )
        player = self.next_actor
        executed_command = command.payload()
        legal = {row["action"]: row for row in self.legal_action_buttons()}
        if command.action not in legal:
            raise ValueError(f"action is not legal in current virtual hand state: {command.action}")
        to_call = self._to_call(player)
        old_bet = self.current_bet
        action_text: str
        if command.action == "fold":
            action_text = "fold"
            self.history.append(f"{self._position(player)} {action_text}")
            self.revision += 1
            self._settle_fold(1 - player)
            return self._feedback(executed_command)
        if command.action == "check":
            action_text = "check"
            self.acted.add(player)
        elif command.action == "call":
            expected = min(to_call, self.stacks[player])
            if command.amount_bb is not None and D(command.amount_bb) != expected:
                raise ValueError("call amount does not match the visible BB price")
            self._commit_to(player, self.street_contributions[player] + expected)
            executed_command["amount_bb"] = _bb(expected)
            action_text = "call"
            self.acted.add(player)
        else:
            maximum = self._max_to(player)
            if command.action == "all_in":
                target = maximum
            else:
                if command.to_amount_bb is None:
                    raise ValueError("bet/raise requires to_amount_bb")
                target = D(command.to_amount_bb)
                minimum = D(legal[command.action]["minimum_to_bb"])
                if target < minimum or target > maximum:
                    raise ValueError("bet/raise target is outside the legal BB range")
            committed = self._commit_to(player, target)
            executed_command["to_amount_bb"] = _bb(target)
            executed_command["amount_bb"] = _bb(committed)
            self.current_bet = max(self.current_bet, target)
            increment = self.current_bet - old_bet
            if increment >= self.last_full_raise:
                self.last_full_raise = increment
            action_text = (
                f"all_in:{_bb(target)}"
                if command.action == "all_in"
                else f"bet:{_bb(target)}"
                if command.action == "bet"
                else f"raise_to:{_bb(target)}"
            )
            self.acted = {player}
        self.history.append(f"{self._position(player)} {action_text}")
        self.revision += 1
        opponent = 1 - player
        contributions_equal = self.street_contributions[0] == self.street_contributions[1]
        round_closed = contributions_equal and (
            len(self.acted) == 2 or self.stacks[0] == 0 or self.stacks[1] == 0
        )
        if round_closed:
            if self.stacks[0] == 0 or self.stacks[1] == 0:
                self._runout_and_showdown()
            else:
                self._advance_street()
        else:
            self.next_actor = opponent
        return self._feedback(executed_command)

    def _advance_street(self) -> None:
        index = STREETS.index(self.street)
        if self.street == "river":
            self._settle_showdown()
            return
        self.street = STREETS[index + 1]
        self.board = self.runout[: BOARD_COUNTS[self.street]]
        self.street_contributions = [D("0"), D("0")]
        self.current_bet = D("0")
        self.last_full_raise = D("1")
        self.acted = set()
        self.next_actor = 1 - self.button

    def _runout_and_showdown(self) -> None:
        self.street = "river"
        self.board = self.runout
        self._settle_showdown()

    def _settle_fold(self, winner: int) -> None:
        self._settle(winner, "fold", reveal=False)

    def _settle_showdown(self) -> None:
        best_hand_rank, _ = _coach_types()
        ranks = [best_hand_rank((*self.hole_cards[index], *self.runout)) for index in (0, 1)]
        winner = 0 if ranks[0] > ranks[1] else 1 if ranks[1] > ranks[0] else None
        self._settle(winner, "showdown", reveal=True)

    def _settle(self, winner: int | None, reason: str, *, reveal: bool) -> None:
        pot = sum(self.total_contributions, D("0"))
        self.final_pot = pot
        if winner is None:
            self.stacks[0] += pot / 2
            self.stacks[1] += pot / 2
        else:
            self.stacks[winner] += pot
        self.terminal = True
        self.terminal_reason = reason
        self.winner = winner
        self.showdown_revealed = reveal
        self.next_actor = None
        self.rewards = [stack - self.starting_stack for stack in self.stacks]

    def _feedback(self, command: dict[str, str]) -> dict[str, object]:
        result = {
            "schema_version": "1.0.0",
            "hand_id": self.hand_id,
            "revision": self.revision,
            "units": "BB",
            "command": command,
            "state": self.observation(),
            "terminal": self.terminal,
            "completed_hand_feedback": None,
            "external_actuation": False,
        }
        if self.terminal:
            result["completed_hand_feedback"] = {
                "full_hand_completed": True,
                "terminal_reason": self.terminal_reason,
                "final_pot_bb": _bb(self.final_pot),
                "winner": None if self.winner is None else PLAYER_NAMES[self.winner],
                "rewards_bb": {
                    PLAYER_NAMES[index]: _bb(reward)
                    for index, reward in enumerate(self.rewards)
                },
                "final_stacks_bb": {
                    PLAYER_NAMES[index]: _bb(stack)
                    for index, stack in enumerate(self.stacks)
                },
                "showdown_revealed": self.showdown_revealed,
                "board": [str(card) for card in self.board],
                "hole_cards": {
                    PLAYER_NAMES[index]: [str(card) for card in self.hole_cards[index]]
                    for index in (0, 1)
                }
                if self.showdown_revealed
                else None,
            }
        result["transition_fingerprint"] = _fingerprint(result)
        return result
