from itertools import permutations
import unittest

from poker_coach import canonicalize_suit_state, suit_isomorphic
from poker_coach.models import Card


SUITS = "cdhs"


def cards(*tokens: str) -> tuple[Card, ...]:
    return tuple(Card.parse(token) for token in tokens)


class SuitIsomorphismTests(unittest.TestCase):
    def test_all_twenty_four_suit_renamings_share_one_canonical_state(self) -> None:
        board = cards("Ah", "7h", "2c", "Ks")
        private = cards("Kh", "Qd")
        expected = canonicalize_suit_state(board, private)
        for targets in permutations(SUITS):
            mapping = dict(zip(SUITS, targets))
            renamed_board = tuple(Card(card.rank, mapping[card.suit]) for card in board)
            renamed_private = tuple(Card(card.rank, mapping[card.suit]) for card in private)
            self.assertEqual(
                canonicalize_suit_state(renamed_board, renamed_private), expected
            )

    def test_private_order_is_irrelevant(self) -> None:
        board = cards("Ah", "7c", "2d")
        private = cards("As", "Kd")
        self.assertEqual(
            canonicalize_suit_state(board, private),
            canonicalize_suit_state(board, tuple(reversed(private))),
        )

    def test_flush_and_blocker_relationships_do_not_collapse(self) -> None:
        board = cards("Ah", "7h", "2c")
        heart_blocker = cards("Kh", "Qd")
        no_heart = cards("Ks", "Qd")
        self.assertFalse(suit_isomorphic(board, heart_blocker, board, no_heart))

    def test_board_arrival_order_remains_part_of_the_state(self) -> None:
        private = cards("As", "Kd")
        self.assertFalse(
            suit_isomorphic(
                cards("Ah", "7c", "2d", "Ks"),
                private,
                cards("Ah", "7c", "Ks", "2d"),
                private,
            )
        )


if __name__ == "__main__":
    unittest.main()
