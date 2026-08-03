from datetime import timedelta
from decimal import Decimal
import unittest

from poker_coach import PokerStarsParser, analyze_hero_trends, analyze_hands
from test_pokerstars import HAND


def hand_at(hand_id: str, timestamp: str, *, table: str = "Naef III") -> str:
    return (
        HAND.replace("#40000000000", f"#{hand_id}", 1)
        .replace("2010/02/19 12:43:11", timestamp, 1)
        .replace("Table 'Naef III'", f"Table '{table}'", 1)
    )


class HeroTrendTests(unittest.TestCase):
    def test_position_street_and_bb_results_are_aggregated(self) -> None:
        hands = PokerStarsParser().parse_many(
            hand_at("40000000000", "2010/02/19 12:43:11")
            + "\n\n"
            + hand_at("40000000001", "2010/02/19 13:13:11")
        )

        report = analyze_hero_trends(hands)

        self.assertEqual(report["units"], "BB")
        self.assertEqual(report["primary_hero"], "Senecady")
        hero = report["heroes"]["Senecady"]
        self.assertEqual(hero["hands"], 2)
        self.assertEqual(Decimal(hero["net_bb"]), Decimal("116.50"))
        self.assertEqual(hero["vpip"]["observed_rate"], "1")
        self.assertEqual(hero["pfr"]["observed_rate"], "1")
        self.assertEqual(hero["positions"]["HJ"]["hands"], 2)
        self.assertEqual(hero["streets"]["preflop"]["decisions"], 2)
        self.assertEqual(hero["streets"]["preflop"]["aggressive_actions"], 2)
        self.assertEqual(len(hero["sessions"]), 1)

    def test_sessions_split_by_table_and_inactivity_gap(self) -> None:
        hands = PokerStarsParser().parse_many(
            hand_at("40000000000", "2010/02/19 12:00:00")
            + "\n\n"
            + hand_at("40000000001", "2010/02/19 12:30:00")
            + "\n\n"
            + hand_at("40000000002", "2010/02/19 14:00:01")
            + "\n\n"
            + hand_at("40000000003", "2010/02/19 12:15:00", table="Other Table")
        )

        hero = analyze_hero_trends(hands)["heroes"]["Senecady"]

        self.assertEqual(len(hero["sessions"]), 3)
        self.assertEqual(
            sorted(session["hands"] for session in hero["sessions"]), [1, 1, 2]
        )
        self.assertEqual(
            {session["table"] for session in hero["sessions"]},
            {"Naef III", "Other Table"},
        )
        self.assertEqual(
            len({session["session_id"] for session in hero["sessions"]}), 3
        )

    def test_invalid_session_gap_is_rejected(self) -> None:
        hands = PokerStarsParser().parse_many(HAND)
        with self.assertRaisesRegex(ValueError, "session_gap"):
            analyze_hero_trends(hands, session_gap=timedelta(0))

    def test_main_analysis_report_exposes_trends(self) -> None:
        report = analyze_hands(PokerStarsParser().parse_many(HAND))
        self.assertEqual(report["schema_version"], "0.3.0")
        self.assertEqual(report["hero_trends"]["primary_hero"], "Senecady")


if __name__ == "__main__":
    unittest.main()
