from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from typing import Iterable

from .models import Card


RANK_VALUE = {rank: value for value, rank in enumerate("23456789TJQKA", start=2)}
CATEGORY_NAMES = (
    "high_card",
    "one_pair",
    "two_pair",
    "three_of_a_kind",
    "straight",
    "flush",
    "full_house",
    "four_of_a_kind",
    "straight_flush",
)


def full_deck() -> tuple[Card, ...]:
    return tuple(Card(rank, suit) for rank in "23456789TJQKA" for suit in "cdhs")


@dataclass(frozen=True, order=True)
class HandRank:
    category: int
    kickers: tuple[int, ...]

    @property
    def name(self) -> str:
        return CATEGORY_NAMES[self.category]


@dataclass(frozen=True)
class EquityResult:
    wins: int
    ties: int
    losses: int

    @property
    def trials(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def equity(self) -> Decimal:
        if self.trials == 0:
            raise ValueError("Equity result has no trials")
        return (Decimal(self.wins) + Decimal(self.ties) / 2) / Decimal(self.trials)


@dataclass(frozen=True)
class WeightedCombo:
    cards: tuple[Card, Card]
    weight: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        if len(self.cards) != 2 or len(set(self.cards)) != 2:
            raise ValueError("Weighted combo contains duplicate cards")
        if self.weight < 0:
            raise ValueError("Combo weight cannot be negative")


@dataclass(frozen=True)
class WeightedEquityResult:
    equity: Decimal
    compatible_weight: Decimal
    combinations: int


def _straight_high(values: Iterable[int]) -> int | None:
    unique = set(values)
    if 14 in unique:
        unique.add(1)
    ordered = sorted(unique)
    consecutive = 1
    best: int | None = None
    for prior, current in zip(ordered, ordered[1:]):
        if current == prior + 1:
            consecutive += 1
            if consecutive >= 5:
                best = current
        elif current != prior:
            consecutive = 1
    return best


def rank_five(cards: Iterable[Card]) -> HandRank:
    card_list = tuple(cards)
    if len(card_list) != 5 or len(set(card_list)) != 5:
        raise ValueError("rank_five requires five unique cards")
    values = [RANK_VALUE[card.rank] for card in card_list]
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    groups = sorted(((count, value) for value, count in counts.items()), reverse=True)
    flush = len({card.suit for card in card_list}) == 1
    straight_high = _straight_high(values)

    if flush and straight_high is not None:
        return HandRank(8, (straight_high,))
    if groups[0][0] == 4:
        quad = groups[0][1]
        kicker = max(value for value in values if value != quad)
        return HandRank(7, (quad, kicker))
    if sorted(counts.values()) == [2, 3]:
        trip = max(value for value, count in counts.items() if count == 3)
        pair = max(value for value, count in counts.items() if count == 2)
        return HandRank(6, (trip, pair))
    if flush:
        return HandRank(5, tuple(sorted(values, reverse=True)))
    if straight_high is not None:
        return HandRank(4, (straight_high,))
    if groups[0][0] == 3:
        trip = groups[0][1]
        kickers = sorted((value for value in values if value != trip), reverse=True)
        return HandRank(3, (trip, *kickers))
    pairs = sorted((value for value, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = max(value for value in values if value not in pairs)
        return HandRank(2, (pairs[0], pairs[1], kicker))
    if len(pairs) == 1:
        pair = pairs[0]
        kickers = sorted((value for value in values if value != pair), reverse=True)
        return HandRank(1, (pair, *kickers))
    return HandRank(0, tuple(sorted(values, reverse=True)))


def best_hand_rank(cards: Iterable[Card]) -> HandRank:
    card_list = tuple(cards)
    if not 5 <= len(card_list) <= 7 or len(set(card_list)) != len(card_list):
        raise ValueError("best_hand_rank requires five to seven unique cards")
    return max(rank_five(combo) for combo in combinations(card_list, 5))


def equity_vs_hand(
    hero: tuple[Card, Card],
    villain: tuple[Card, Card],
    board: Iterable[Card] = (),
    dead: Iterable[Card] = (),
) -> EquityResult:
    board_cards = tuple(board)
    dead_cards = tuple(dead)
    if len(board_cards) > 5:
        raise ValueError("Board cannot contain more than five cards")
    known = (*hero, *villain, *board_cards, *dead_cards)
    if len(set(known)) != len(known):
        raise ValueError("Known cards conflict")
    remaining = tuple(card for card in full_deck() if card not in set(known))
    runout_size = 5 - len(board_cards)
    wins = ties = losses = 0
    for runout in combinations(remaining, runout_size):
        complete_board = (*board_cards, *runout)
        hero_rank = best_hand_rank((*hero, *complete_board))
        villain_rank = best_hand_rank((*villain, *complete_board))
        if hero_rank > villain_rank:
            wins += 1
        elif hero_rank == villain_rank:
            ties += 1
        else:
            losses += 1
    return EquityResult(wins, ties, losses)


def equity_vs_range(
    hero: tuple[Card, Card],
    villain_range: Iterable[WeightedCombo],
    board: Iterable[Card] = (),
    dead: Iterable[Card] = (),
) -> WeightedEquityResult:
    board_cards = tuple(board)
    dead_cards = tuple(dead)
    fixed = set((*hero, *board_cards, *dead_cards))
    weighted_equity = Decimal("0")
    compatible_weight = Decimal("0")
    combo_count = 0
    for combo in villain_range:
        if combo.weight == 0 or any(card in fixed for card in combo.cards):
            continue
        result = equity_vs_hand(hero, combo.cards, board_cards, dead_cards)
        weighted_equity += combo.weight * result.equity
        compatible_weight += combo.weight
        combo_count += 1
    if compatible_weight == 0:
        raise ValueError("Villain range has no compatible positive-weight combinations")
    return WeightedEquityResult(weighted_equity / compatible_weight, compatible_weight, combo_count)
