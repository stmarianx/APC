from decimal import Decimal
import unittest

from poker_coach.models import Card
from poker_coach.ranges import expand_class, parse_range


class RangeTests(unittest.TestCase):
    def test_standard_class_counts(self) -> None:
        self.assertEqual(len(expand_class("AA")), 6)
        self.assertEqual(len(expand_class("AKs")), 4)
        self.assertEqual(len(expand_class("AKo")), 12)
        self.assertEqual(len(expand_class("AK")), 16)

    def test_plus_notation(self) -> None:
        self.assertEqual(len(expand_class("TT+")), 30)
        self.assertEqual(len(expand_class("AJs+")), 12)

    def test_dead_cards_remove_combinations(self) -> None:
        dead = (Card.parse("As"),)
        self.assertEqual(len(expand_class("AA", dead=dead)), 3)
        self.assertEqual(len(expand_class("AKs", dead=dead)), 3)

    def test_weighted_range(self) -> None:
        result = parse_range("AA, AKs:0.5")
        self.assertEqual(len(result), 10)
        self.assertEqual(sum((combo.weight for combo in result), Decimal("0")), Decimal("8.0"))

    def test_invalid_order_and_weight(self) -> None:
        with self.assertRaises(ValueError):
            expand_class("KAo")
        with self.assertRaises(ValueError):
            parse_range("AA:1.2")


if __name__ == "__main__":
    unittest.main()

