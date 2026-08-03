from decimal import Decimal
import unittest

from poker_coach.decision_math import (
    bluff_break_even_fold_frequency,
    call_break_even_equity,
    call_ev,
    expected_value,
    hit_probability_by_river,
    minimum_defense_frequency,
    polar_bluff_share,
    polar_bluff_to_value_ratio,
    pure_bluff_ev,
    stack_to_pot_ratio,
)


class DecisionMathTests(unittest.TestCase):
    def test_half_pot_geometry(self) -> None:
        self.assertEqual(call_break_even_equity(100, 50), Decimal("0.25"))
        self.assertEqual(call_ev(Decimal("0.25"), 100, 50), Decimal("0.00"))
        self.assertEqual(bluff_break_even_fold_frequency(100, 50), Decimal(1) / Decimal(3))
        self.assertLess(abs(pure_bluff_ev(Decimal(1) / Decimal(3), 100, 50)), Decimal("1e-20"))
        self.assertEqual(minimum_defense_frequency(100, 50), Decimal(2) / Decimal(3))
        self.assertEqual(polar_bluff_to_value_ratio(100, 50), Decimal(1) / Decimal(3))
        self.assertEqual(polar_bluff_share(100, 50), Decimal("0.25"))

    def test_nine_outs_by_river(self) -> None:
        probability = hit_probability_by_river(9)
        self.assertAlmostEqual(float(probability), 0.3497, places=4)

    def test_spr_and_generic_ev(self) -> None:
        self.assertEqual(stack_to_pot_ratio(300, 100), Decimal(3))
        self.assertEqual(expected_value((("0.25", "30"), ("0.75", "-10"))), Decimal(0))

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expected_value((("0.7", 1), ("0.4", -1)))


if __name__ == "__main__":
    unittest.main()
