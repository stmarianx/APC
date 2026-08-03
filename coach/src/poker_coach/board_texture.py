from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .equity import RANK_VALUE, best_hand_rank
from .models import Card


VALUE_RANK = {value: rank for rank, value in RANK_VALUE.items()}
STRAIGHT_WINDOWS = (
    frozenset((14, 2, 3, 4, 5)),
    *(frozenset(range(low, low + 5)) for low in range(2, 11)),
)
RANGE_CAVEAT = (
    "Board texture alone does not establish range advantage or nut advantage; "
    "those require explicit position, action-history, and range assumptions."
)


def _rank_name(value: int) -> str:
    return VALUE_RANK[value]


def _pairing(rank_counts: Counter[str]) -> str:
    if not rank_counts:
        return "no_board"
    groups = sorted(rank_counts.values(), reverse=True)
    if groups[0] == 4:
        return "quads_on_board"
    if groups[0] == 3 and len(groups) > 1 and groups[1] == 2:
        return "full_house_on_board"
    if groups[0] == 3:
        return "trips_on_board"
    if groups[:2] == [2, 2]:
        return "double_paired"
    if groups[0] == 2:
        return "paired"
    return "unpaired"


def _suit_texture(board_size: int, max_suit_count: int) -> str:
    if board_size == 0:
        return "no_board"
    if max_suit_count == 1:
        return "rainbow"
    if board_size == 3 and max_suit_count == 2:
        return "two_tone"
    if board_size == 3 and max_suit_count == 3:
        return "monotone"
    if max_suit_count == 2:
        return "two_suited"
    if max_suit_count == 3:
        return "three_flush"
    if max_suit_count == 4:
        return "four_flush"
    return "flush_on_board"


def _window_coverage(values: set[int]) -> tuple[int, tuple[str, ...]]:
    if not values:
        return 0, ()
    maximum = max(len(values & window) for window in STRAIGHT_WINDOWS)
    completions = {
        _rank_name(next(iter(window - values)))
        for window in STRAIGHT_WINDOWS
        if len(window - values) == 1
    }
    return maximum, tuple(sorted(completions, key=RANK_VALUE.__getitem__, reverse=True))


def _straight_texture(board_size: int, coverage: int) -> str:
    if board_size == 0:
        return "no_board"
    if coverage >= 5:
        return "straight_on_board"
    if coverage == 4:
        return "four_connected"
    if coverage == 3:
        return "connected"
    return "disconnected"


def _preflop_relation(hero_cards: tuple[Card, ...]) -> str:
    if not hero_cards:
        return "unknown"
    if hero_cards[0].rank == hero_cards[1].rank:
        return "pocket_pair"
    return "suited" if hero_cards[0].suit == hero_cards[1].suit else "offsuit"


def _pair_relation(board: tuple[Card, ...], hero_cards: tuple[Card, ...]) -> str:
    if not board or not hero_cards:
        return _preflop_relation(hero_cards)
    board_values = sorted({RANK_VALUE[card.rank] for card in board}, reverse=True)
    hero_values = [RANK_VALUE[card.rank] for card in hero_cards]
    board_ranks = {card.rank for card in board}
    if hero_cards[0].rank == hero_cards[1].rank:
        value = hero_values[0]
        if hero_cards[0].rank in board_ranks:
            return "pocket_pair_matches_board"
        return "overpair" if value > max(board_values) else "underpair"
    matched = sorted({value for value in hero_values if value in board_values}, reverse=True)
    if len(matched) >= 2:
        return "two_board_pairs"
    if not matched:
        return "no_pair"
    index = board_values.index(matched[0])
    return "top_pair" if index == 0 else "bottom_pair" if index == len(board_values) - 1 else "middle_pair"


def _hero_straight_draw(
    board: tuple[Card, ...], hero_cards: tuple[Card, ...], made_hand: str
) -> tuple[str, tuple[str, ...]]:
    if len(board) >= 5 or not hero_cards or made_hand in {"straight", "straight_flush"}:
        return "none", ()
    board_values = {RANK_VALUE[card.rank] for card in board}
    hero_values = {RANK_VALUE[card.rank] for card in hero_cards}
    combined = board_values | hero_values
    completions: set[int] = set()
    for window in STRAIGHT_WINDOWS:
        missing = window - combined
        private_contribution = (hero_values - board_values) & window
        if len(missing) == 1 and private_contribution:
            completions.update(missing)
    labels = tuple(
        _rank_name(value) for value in sorted(completions, reverse=True)
    )
    if len(labels) >= 2:
        return "open_ended_or_double_gutshot", labels
    if labels:
        return "gutshot_or_edge", labels
    return "none", ()


def _hero_flush_state(
    board: tuple[Card, ...], hero_cards: tuple[Card, ...], made_hand: str
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if not hero_cards:
        return "none", (), ()
    board_suits = Counter(card.suit for card in board)
    hero_suits = Counter(card.suit for card in hero_cards)
    blockers = tuple(
        str(card)
        for card in hero_cards
        if board_suits[card.suit] >= 3
    )
    nut_blockers: list[str] = []
    for card in hero_cards:
        if board_suits[card.suit] < 3:
            continue
        board_ranks = {
            RANK_VALUE[board_card.rank]
            for board_card in board
            if board_card.suit == card.suit
        }
        highest_unseen = max(value for value in range(2, 15) if value not in board_ranks)
        if RANK_VALUE[card.rank] == highest_unseen:
            nut_blockers.append(str(card))
    if made_hand in {"flush", "straight_flush"}:
        draw = "made_flush"
    else:
        draw = "none"
        for suit in "cdhs":
            combined = board_suits[suit] + hero_suits[suit]
            if hero_suits[suit] and len(board) < 5 and combined == 4:
                board_ranks = {
                    RANK_VALUE[card.rank] for card in board if card.suit == suit
                }
                highest_unseen = max(
                    value for value in range(2, 15) if value not in board_ranks
                )
                hero_has_nut = any(
                    card.suit == suit and RANK_VALUE[card.rank] == highest_unseen
                    for card in hero_cards
                )
                draw = "nut_flush_draw" if hero_has_nut else "flush_draw"
                break
            if len(board) == 3 and hero_suits[suit] and combined == 3:
                draw = "backdoor_flush_draw"
    return draw, blockers, tuple(nut_blockers)


@dataclass(frozen=True)
class BoardTexture:
    street: str
    pairing: str
    suit_texture: str
    suit_counts: tuple[tuple[str, int], ...]
    straight_texture: str
    max_straight_window_coverage: int
    board_straight_completion_ranks: tuple[str, ...]
    broadway_cards: int
    high_card: str | None
    hero_made_hand: str
    hero_pair_relation: str
    hero_overcards: tuple[str, ...]
    hero_flush_draw: str
    hero_straight_draw: str
    hero_straight_completion_ranks: tuple[str, ...]
    hero_flush_blockers: tuple[str, ...]
    hero_nut_flush_blockers: tuple[str, ...]
    facts: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "street": self.street,
            "pairing": self.pairing,
            "suit_texture": self.suit_texture,
            "suit_counts": dict(self.suit_counts),
            "straight_texture": self.straight_texture,
            "max_straight_window_coverage": self.max_straight_window_coverage,
            "board_straight_completion_ranks": list(
                self.board_straight_completion_ranks
            ),
            "broadway_cards": self.broadway_cards,
            "high_card": self.high_card,
            "hero": {
                "made_hand": self.hero_made_hand,
                "pair_relation": self.hero_pair_relation,
                "overcards": list(self.hero_overcards),
                "flush_draw": self.hero_flush_draw,
                "straight_draw": self.hero_straight_draw,
                "straight_completion_ranks": list(
                    self.hero_straight_completion_ranks
                ),
                "flush_blockers": list(self.hero_flush_blockers),
                "nut_flush_blockers": list(self.hero_nut_flush_blockers),
            },
            "facts": list(self.facts),
            "range_caveat": RANGE_CAVEAT,
        }


def analyze_board_texture(
    board: Iterable[Card], hero_cards: Iterable[Card] = ()
) -> BoardTexture:
    board_cards = tuple(board)
    hero = tuple(hero_cards)
    if len(board_cards) not in (0, 3, 4, 5):
        raise ValueError("board must contain zero, three, four, or five cards")
    if len(hero) not in (0, 2):
        raise ValueError("hero_cards must be empty or contain exactly two cards")
    if len(set(board_cards + hero)) != len(board_cards + hero):
        raise ValueError("board texture input contains conflicting cards")

    street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(board_cards)]
    rank_counts = Counter(card.rank for card in board_cards)
    suit_counts = Counter(card.suit for card in board_cards)
    board_values = {RANK_VALUE[card.rank] for card in board_cards}
    coverage, board_completions = _window_coverage(board_values)
    pairing = _pairing(rank_counts)
    suit_texture = _suit_texture(
        len(board_cards), max(suit_counts.values(), default=0)
    )
    straight_texture = _straight_texture(len(board_cards), coverage)
    made_hand = (
        "unmade_preflop"
        if not board_cards or not hero
        else best_hand_rank((*board_cards, *hero)).name
    )
    pair_relation = _pair_relation(board_cards, hero)
    straight_draw, hero_completions = _hero_straight_draw(
        board_cards, hero, made_hand
    )
    flush_draw, flush_blockers, nut_flush_blockers = _hero_flush_state(
        board_cards, hero, made_hand
    )
    high_value = max(board_values, default=None)
    overcards = tuple(
        card.rank
        for card in sorted(hero, key=lambda row: RANK_VALUE[row.rank], reverse=True)
        if high_value is not None and RANK_VALUE[card.rank] > high_value
    )

    facts: list[str] = []
    if not board_cards:
        facts.append(f"Preflop holding structure: {pair_relation.replace('_', ' ')}.")
    else:
        facts.append(
            f"The {street} is {pairing.replace('_', ' ')}, {suit_texture.replace('_', ' ')}, "
            f"and {straight_texture.replace('_', ' ')}."
        )
        facts.append(
            f"Hero currently has {made_hand.replace('_', ' ')}"
            f" ({pair_relation.replace('_', ' ')} interaction)."
        )
        if flush_draw not in {"none", "made_flush"}:
            facts.append(f"Hero has a {flush_draw.replace('_', ' ')}.")
        if straight_draw != "none":
            facts.append(
                f"Hero's {straight_draw.replace('_', ' ')} completes on "
                f"{', '.join(hero_completions)}."
            )
        if nut_flush_blockers:
            facts.append(
                f"Hero holds the nut flush blocker: {', '.join(nut_flush_blockers)}."
            )
        elif flush_blockers:
            facts.append(f"Hero blocks flushes with {', '.join(flush_blockers)}.")

    return BoardTexture(
        street=street,
        pairing=pairing,
        suit_texture=suit_texture,
        suit_counts=tuple(sorted(suit_counts.items())),
        straight_texture=straight_texture,
        max_straight_window_coverage=coverage,
        board_straight_completion_ranks=board_completions,
        broadway_cards=sum(card.rank in "TJQKA" for card in board_cards),
        high_card=None if high_value is None else _rank_name(high_value),
        hero_made_hand=made_hand,
        hero_pair_relation=pair_relation,
        hero_overcards=overcards,
        hero_flush_draw=flush_draw,
        hero_straight_draw=straight_draw,
        hero_straight_completion_ranks=hero_completions,
        hero_flush_blockers=flush_blockers,
        hero_nut_flush_blockers=nut_flush_blockers,
        facts=tuple(facts),
    )
