from pathlib import Path
import unittest
from dataclasses import replace

from poker_coach.matching import DecisionSolutionMatcher, analyze_with_solutions
from poker_coach.pokerstars import PokerStarsParser
from poker_coach.solver_import import SolverBundleImporter
from poker_coach.models import Card


ROOT = Path(__file__).resolve().parents[1]


class MatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hand = PokerStarsParser().parse_file(
            ROOT / "examples" / "sample_play_money_hand.txt"
        )[0]
        self.bundle = SolverBundleImporter().parse_file(
            ROOT / "examples" / "sample_solver_bundle.json"
        )

    def test_exact_public_state_and_private_hand_match(self) -> None:
        matches = DecisionSolutionMatcher().match_hand(self.hand, self.bundle.spots)
        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual(match.context.street.name, "FLOP")
        self.assertEqual(match.confidence, "exact")
        self.assertEqual(match.matched_action, "bet:0.62")
        self.assertEqual(str(match.ev_loss_bb), "0.15")

    def test_review_generates_prioritized_drill_and_counts_unmatched(self) -> None:
        review = analyze_with_solutions((self.hand,), self.bundle.spots)
        self.assertEqual(review["hero_decisions"], 3)
        self.assertEqual(review["matched_decisions"], 1)
        self.assertEqual(review["unmatched_decisions"], 2)
        self.assertEqual(len(review["drills"]), 1)
        drill = review["drills"][0]
        self.assertEqual(drill["ev_loss_bb"], "0.15")
        self.assertEqual(drill["observed_action"], "bet:0.62")
        self.assertEqual(drill["best_actions"][0]["action"], "bet:0.33")
        summary = review["leak_summary"]
        self.assertEqual(summary["total_ev_loss_bb"], "0.15")
        self.assertEqual(summary["average_ev_loss_bb"], "0.15")
        self.assertEqual(summary["worst_decision"]["street"], "flop")
        self.assertEqual(summary["by_street"]["flop"]["ev_loss_bb"], "0.15")
        self.assertEqual(summary["coverage"], "0.3333333333333333333333333333")

    def test_wrong_private_hand_does_not_match(self) -> None:
        without_sample_hand = self.bundle.spots[:2]
        self.assertEqual(
            DecisionSolutionMatcher().match_hand(self.hand, without_sample_hand),
            (),
        )

    def test_suit_renamed_solution_is_an_exact_mathematical_match(self) -> None:
        original = self.bundle.spots[2]
        renamed_key = replace(
            original.key,
            board=(Card.parse("As"), Card.parse("7d"), Card.parse("2h")),
            hero_cards=(Card.parse("Ac"), Card.parse("Kh")),
        )
        renamed = replace(original, key=renamed_key)
        matches = DecisionSolutionMatcher().match_hand(self.hand, (renamed,))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].confidence, "exact")
        self.assertEqual(matches[0].card_match, "suit_isomorphic")
        self.assertEqual(matches[0].to_dict()["match"]["card_match"], "suit_isomorphic")

    def test_non_isomorphic_blocker_structure_does_not_match(self) -> None:
        original = self.bundle.spots[2]
        different = replace(
            original,
            key=replace(
                original.key,
                hero_cards=(Card.parse("Ac"), Card.parse("Kd")),
            ),
        )
        self.assertEqual(
            DecisionSolutionMatcher().match_hand(self.hand, (different,)), ()
        )


if __name__ == "__main__":
    unittest.main()
