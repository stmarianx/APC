from decimal import Decimal
from pathlib import Path
import unittest

from poker_coach import (
    ActionSolution,
    HandReplayer,
    PokerStarsParser,
    SolutionKey,
    SolvedSpot,
    explain_decision,
)


class ExplanationTests(unittest.TestCase):
    def test_aggressive_geometry_and_solution_provenance(self) -> None:
        path = Path(__file__).resolve().parent.parent / "examples" / "sample_play_money_hand.txt"
        hand = PokerStarsParser().parse_file(path)[0]
        replay = HandReplayer().replay(hand)
        action = next(action for action in hand.actions if action.raw == "Hero: bets 80")
        snapshot = next(snapshot for snapshot in replay.decisions if snapshot.action_index == action.index)
        key = SolutionKey(
            game="holdem_no_limit",
            players=2,
            hero_position="BTN",
            effective_stack_bb=Decimal("97"),
            pot_bb=Decimal("6.5"),
            board=hand.board[:3],
            action_history=("BTN raise_to:3", "BB call"),
            rake_model="play_money_fixture",
        )
        solution = SolvedSpot(
            key,
            (
                ActionSolution("check", Decimal("0.3"), Decimal("18")),
                ActionSolution("bet:80", Decimal("0.7"), Decimal("20")),
            ),
            source="fixture_solver",
        )
        explanation = explain_decision(snapshot, action, solution=solution)
        self.assertEqual(explanation["pot_before_action"], "130")
        self.assertEqual(explanation["aggressive_risk"], "80")
        self.assertEqual(explanation["solution"]["frequency"], "0.7")
        self.assertEqual(explanation["solution"]["ev_loss"], "0")


if __name__ == "__main__":
    unittest.main()

