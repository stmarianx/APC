from __future__ import annotations

from functools import lru_cache
from itertools import permutations

from .models import Card


SUITS = ("c", "d", "h", "s")
_RANK_STRENGTH = {rank: index for index, rank in enumerate("23456789TJQKA")}


def _private_order(card: Card) -> tuple[int, int]:
    return (-_RANK_STRENGTH[card.rank], SUITS.index(card.suit))


@lru_cache(maxsize=131_072)
def canonicalize_suit_state(
    board: tuple[Card, ...],
    hero_cards: tuple[Card, ...] = (),
) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
    """Return a deterministic representative of a hold'em suit-isomorphism class.

    Board order is retained because turn and river arrival order can identify a
    different decision tree. Private-card order is normalized because a hold'em
    holding is unordered. Exhausting all 24 suit renamings makes the result
    correct even when board symmetries leave several suits interchangeable.
    """

    cards = board + hero_cards
    if len(set(cards)) != len(cards):
        raise ValueError("Cannot canonicalize conflicting cards")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError("Invalid board length for suit canonicalization")
    if len(hero_cards) not in (0, 2):
        raise ValueError("Invalid private-card length for suit canonicalization")

    best_score: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    best_state: tuple[tuple[Card, ...], tuple[Card, ...]] | None = None
    for target_suits in permutations(SUITS):
        mapping = dict(zip(SUITS, target_suits))
        mapped_board = tuple(Card(card.rank, mapping[card.suit]) for card in board)
        mapped_private = tuple(
            sorted(
                (Card(card.rank, mapping[card.suit]) for card in hero_cards),
                key=_private_order,
            )
        )
        score = (
            tuple(str(card) for card in mapped_board),
            tuple(str(card) for card in mapped_private),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_state = mapped_board, mapped_private
    assert best_state is not None
    return best_state


def suit_isomorphic(
    first_board: tuple[Card, ...],
    first_private: tuple[Card, ...],
    second_board: tuple[Card, ...],
    second_private: tuple[Card, ...],
) -> bool:
    if len(first_board) != len(second_board) or len(first_private) != len(second_private):
        return False
    return canonicalize_suit_state(
        first_board, first_private
    ) == canonicalize_suit_state(second_board, second_private)
