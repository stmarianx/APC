from decimal import Decimal
import unittest

from poker_coach.equity import WeightedCombo
from poker_coach.models import Card
from poker_coach.range_matchup import analyze_range_matchup
from poker_coach.ranges import parse_range


def cards(text: str) -> tuple[Card, ...]:
    return tuple(Card.parse(token) for token in text.split())


class RangeMatchupTests(unittest.TestCase):
    def test_exact_river_equity_and_relative_nut_share(self) -> None:
        board = cards("2h 3h 4s 9c 7d")
        result = analyze_range_matchup(
            parse_range("AsAd", dead=board),
            parse_range("KcKd,QcQd", dead=board),
            board,
        ).to_dict()
        self.assertEqual(result["method"], "exact_enumeration")
        self.assertEqual(result["compatible_matchups"], 2)
        self.assertEqual(result["equity"]["hero"], "1")
        self.assertEqual(result["equity"]["confidence_95"]["half_width"], "0")
        self.assertEqual(result["current_range_relative_nuts"]["leader"], "hero")
        self.assertEqual(result["hero_range"]["current_nut_share"], "1")
        self.assertEqual(result["villain_range"]["current_nut_share"], "0")

    def test_board_playing_nuts_tie_both_ranges(self) -> None:
        board = cards("Ah Kh Qh Jh Th")
        result = analyze_range_matchup(
            parse_range("2c3d", dead=board),
            parse_range("4c5d", dead=board),
            board,
        ).to_dict()
        self.assertEqual(result["equity"]["hero"], "0.5")
        self.assertEqual(result["equity"]["tie"], "1")
        self.assertEqual(result["current_range_relative_nuts"]["leader"], "even")
        self.assertEqual(result["hero_range"]["current_nut_share"], "1")
        self.assertEqual(result["villain_range"]["current_nut_share"], "1")

    def test_board_blocked_combos_are_audited(self) -> None:
        board = cards("As 7d 2h 9c 4s")
        hero = (
            WeightedCombo(cards("As Kd"), Decimal("1")),
            WeightedCombo(cards("Ah Kh"), Decimal("1")),
        )
        result = analyze_range_matchup(
            hero, parse_range("QcQd", dead=board), board
        ).to_dict()
        self.assertEqual(result["hero_range"]["input_combos"], 2)
        self.assertEqual(result["hero_range"]["active_combos"], 1)
        self.assertEqual(result["hero_range"]["board_blocked_or_zero_combos"], 1)

    def test_cross_range_impossible_combo_cannot_define_relative_nuts(self) -> None:
        board = cards("Ah 7d 2h 9c 4s")
        hero = (
            WeightedCombo(cards("As Ad"), Decimal("1")),
            WeightedCombo(cards("Qc Qd"), Decimal("1")),
        )
        villain = (WeightedCombo(cards("As Kd"), Decimal("1")),)
        result = analyze_range_matchup(hero, villain, board).to_dict()
        self.assertEqual(result["hero_range"]["matchup_compatible_combos"], 1)
        self.assertEqual(
            result["current_range_relative_nuts"]["strongest_hand"]["category"],
            "one_pair",
        )
        self.assertEqual(result["current_range_relative_nuts"]["leader"], "villain")

    def test_monte_carlo_is_deterministic_and_reports_interval(self) -> None:
        board = cards("2c 7d 9h")
        hero = parse_range("AA,KK,QQ,AKs", dead=board)
        villain = parse_range("JJ,TT,AQs,KQs", dead=board)
        first = analyze_range_matchup(
            hero, villain, board, samples=2_000, seed=41, max_exact_outcomes=1
        ).to_dict()
        second = analyze_range_matchup(
            hero, villain, board, samples=2_000, seed=41, max_exact_outcomes=1
        ).to_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["method"], "deterministic_monte_carlo")
        equity = Decimal(first["equity"]["hero"])
        interval = first["equity"]["confidence_95"]
        self.assertLessEqual(Decimal(interval["lower"]), equity)
        self.assertGreaterEqual(Decimal(interval["upper"]), equity)
        self.assertEqual(
            equity + Decimal(first["equity"]["villain"]), Decimal("1")
        )

    def test_preflop_equity_withholds_current_nut_claim(self) -> None:
        result = analyze_range_matchup(
            parse_range("AsAd"),
            parse_range("KcKd"),
            (),
            samples=500,
            seed=3,
            max_exact_outcomes=1,
        ).to_dict()
        self.assertEqual(result["street"], "preflop")
        self.assertIsNone(result["current_range_relative_nuts"]["strongest_hand"])
        self.assertEqual(
            result["current_range_relative_nuts"]["leader"],
            "not_available_preflop",
        )

    def test_invalid_controls_and_fully_conflicting_ranges_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "samples must be between"):
            analyze_range_matchup(parse_range("AA"), parse_range("KK"), samples=1)
        same = parse_range("AsAd")
        with self.assertRaisesRegex(ValueError, "no card-compatible matchups"):
            analyze_range_matchup(same, same, cards("2c 7d 9h"))


if __name__ == "__main__":
    unittest.main()
