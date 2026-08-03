from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, IntEnum


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4
    SUMMARY = 5


class ActionKind(str, Enum):
    POST_ANTE = "post_ante"
    POST_SMALL_BLIND = "post_small_blind"
    POST_BIG_BLIND = "post_big_blind"
    POST_DEAD_BLIND = "post_dead_blind"
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    SHOW = "show"
    MUCK = "muck"
    RETURN = "return"


_SUIT_ALIASES = {"c": "c", "d": "d", "h": "h", "s": "s", "♣": "c", "♦": "d", "♥": "h", "♠": "s"}
_CARD_RE = re.compile(r"^(10|[2-9TJQKA])([cdhs♣♦♥♠])$", re.IGNORECASE)


@dataclass(frozen=True, order=True)
class Card:
    rank: str
    suit: str

    def __post_init__(self) -> None:
        rank = self.rank.upper().replace("10", "T")
        suit = _SUIT_ALIASES.get(self.suit.lower(), _SUIT_ALIASES.get(self.suit))
        if rank not in tuple("23456789TJQKA") or suit not in tuple("cdhs"):
            raise ValueError(f"Invalid card: {self.rank}{self.suit}")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "suit", suit)

    @classmethod
    def parse(cls, token: str) -> "Card":
        cleaned = token.strip().strip("[],")
        match = _CARD_RE.match(cleaned)
        if not match:
            raise ValueError(f"Invalid card token: {token!r}")
        return cls(match.group(1), match.group(2))

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


@dataclass(frozen=True)
class Player:
    seat: int
    name: str
    starting_stack: Decimal

    def __post_init__(self) -> None:
        if self.seat <= 0:
            raise ValueError("Seat numbers must be positive")
        if not self.name.strip():
            raise ValueError("Player name cannot be empty")
        if self.starting_stack < 0:
            raise ValueError("Starting stack cannot be negative")


@dataclass(frozen=True)
class HoleCards:
    player: str
    cards: tuple[Card, ...]
    shown: bool = False

    def __post_init__(self) -> None:
        if len(self.cards) != 2:
            raise ValueError("Hold'em hole cards must contain exactly two cards")
        if len(set(self.cards)) != 2:
            raise ValueError("Duplicate hole card")


@dataclass(frozen=True)
class HandAction:
    index: int
    street: Street
    kind: ActionKind
    player: str | None = None
    amount: Decimal = Decimal("0")
    to_amount: Decimal | None = None
    all_in: bool = False
    cards: tuple[Card, ...] = ()
    raw: str = ""

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Action index cannot be negative")
        if self.amount < 0 or (self.to_amount is not None and self.to_amount < 0):
            raise ValueError("Action amounts cannot be negative")
        if self.kind == ActionKind.RAISE and self.to_amount is None:
            raise ValueError("Raise action requires to_amount")


@dataclass(frozen=True)
class PotAward:
    player: str
    amount: Decimal
    pot_name: str = "pot"

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("Pot award cannot be negative")


@dataclass(frozen=True)
class HandHistory:
    hand_id: str
    game: str
    limit: str
    stakes_raw: str
    currency: str | None
    small_blind: Decimal | None
    big_blind: Decimal | None
    played_at_raw: str
    table_name: str
    max_seats: int | None
    button_seat: int
    players: tuple[Player, ...]
    hole_cards: tuple[HoleCards, ...]
    board: tuple[Card, ...]
    actions: tuple[HandAction, ...]
    awards: tuple[PotAward, ...]
    total_pot: Decimal | None = None
    rake: Decimal | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.hand_id:
            raise ValueError("Hand id is required")
        seats = [player.seat for player in self.players]
        names = [player.name for player in self.players]
        if len(seats) != len(set(seats)) or len(names) != len(set(names)):
            raise ValueError("Player seats and names must be unique")
        if self.button_seat not in set(seats):
            raise ValueError("Button seat must reference a player")
        if len(self.board) not in (0, 3, 4, 5):
            raise ValueError("Hold'em board must contain 0, 3, 4, or 5 cards")
        if len(set(self.board)) != len(self.board):
            raise ValueError("Board contains duplicate cards")
        player_names = set(names)
        known_cards: list[Card] = list(self.board)
        hole_players: set[str] = set()
        for holding in self.hole_cards:
            if holding.player not in player_names:
                raise ValueError(f"Hole cards reference unknown player: {holding.player}")
            if holding.player in hole_players:
                raise ValueError(f"Duplicate hole-card record: {holding.player}")
            hole_players.add(holding.player)
            known_cards.extend(holding.cards)
        if len(known_cards) != len(set(known_cards)):
            raise ValueError("Known cards conflict across board and players")
        if [action.index for action in self.actions] != list(range(len(self.actions))):
            raise ValueError("Action indexes must be contiguous from zero")

    def player(self, name: str) -> Player:
        for player in self.players:
            if player.name == name:
                return player
        raise KeyError(name)

    @property
    def hero(self) -> str | None:
        for holding in self.hole_cards:
            if not holding.shown:
                return holding.player
        return None

