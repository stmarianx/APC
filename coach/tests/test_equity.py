from decimal import Decimal
import unittest

from poker_coach.equity import WeightedCombo, best_hand_rank, equity_vs_hand, equity_vs_range
from poker_coach.models import Card


def cards(text: str):
    return tuple(Card.parse(token) for token in text.split())


class EquityTests(unittest.TestCase):
    def test_hand_category_ordering_and_wheel(self) -> None:
        straight_flush = best_hand_rank(cards("Ah Kh Qh Jh Th 2c 3d"))
        quads = best_hand_rank(cards("As Ad Ac Ah Kd 2c 3d"))
        wheel = best_hand_rank(cards("As 2d 3h 4c 5s Kd Qd"))
        self.assertGreater(straight_flush, quads)
        self.assertEqual(straight_flush.name, "straight_flush")
        self.assertEqual(wheel.name, "straight")
        self.assertEqual(wheel.kickers, (5,))

    def test_river_win_and_board_tie(self) -> None:
        win = equity_vs_hand(cards("Th 9d"), cards("2d 2s"), cards("Ah Kh Qh Jh 3c"))
        self.assertEqual((win.wins, win.ties, win.losses), (1, 0, 0))
        self.assertEqual(win.equity, Decimal("1"))
        tie = equity_vs_hand(cards("2c 3d"), cards("4c 5d"), cards("Ah Kh Qh Jh Th"))
        self.assertEqual(tie.equity, Decimal("0.5"))

    def test_turn_enumeration_has_44_runouts(self) -> None:
        result = equity_vs_hand(cards("As Ad"), cards("Kc Kd"), cards("2h 3h 4s 9c"))
        self.assertEqual(result.trials, 44)
        self.assertGreater(result.equity, Decimal("0.8"))

    def test_weighted_range_excludes_blocked_combo(self) -> None:
        hero = cards("As Ad")
        board = cards("2h 3h 4s 9c 7d")
        result = equity_vs_range(
            hero,
            (
                WeightedCombo(cards("Kc Kd"), Decimal("1")),
                WeightedCombo(cards("As Qd"), Decimal("10")),
            ),
            board,
        )
        self.assertEqual(result.combinations, 1)
        self.assertEqual(result.compatible_weight, Decimal("1"))
        self.assertEqual(result.equity, Decimal("1"))


if __name__ == "__main__":
    unittest.main()

