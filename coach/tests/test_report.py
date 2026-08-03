import unittest

from poker_coach import PokerStarsParser, analyze_hands
from test_pokerstars import HAND


class ReportTests(unittest.TestCase):
    def test_serializable_analysis_report(self) -> None:
        report = analyze_hands(PokerStarsParser().parse_many(HAND))
        self.assertEqual(report["hands"], 1)
        hand = report["hand_reports"][0]
        self.assertEqual(hand["hand_id"], "40000000000")
        self.assertEqual(hand["reconciliation_error"], "0.00")
        self.assertEqual(len(hand["hero_decisions"]), 1)
        self.assertEqual(hand["hero_decisions"][0]["observed_action"], "raise_to:117")
        self.assertIn("Senecady", report["player_profiles"])
        vpip = next(
            estimate
            for estimate in report["player_profiles"]["Senecady"]
            if estimate["tendency"] == "vpip"
        )
        self.assertEqual(vpip["opportunities"], "1")
        summary = report["profile_summaries"]["Senecady"]
        self.assertEqual(summary["style_label"], "Developing profile")
        self.assertEqual(summary["confidence"], "limited")
        self.assertEqual(summary["metrics"]["vpip"]["opportunities"], "1")
        self.assertIn("Senecady", report["exploit_insights"])
        self.assertTrue(all(row["actionability"] == "observe_only" for row in report["exploit_insights"]["Senecady"]))


if __name__ == "__main__":
    unittest.main()
