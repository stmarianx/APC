from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from apc.evaluate_virtual_table import evaluate_virtual_table
from apc.virtual_table import VirtualDecisionTable


ROOT = Path(__file__).resolve().parents[2]
COACH_SRC = ROOT / "coach" / "src"
if str(COACH_SRC) not in sys.path:
    sys.path.insert(0, str(COACH_SRC))

from poker_coach import ActionSolution, SolverBundleImporter


class VirtualDecisionTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = SolverBundleImporter().parse_file(
            ROOT / "coach" / "examples" / "sample_solver_bundle.json"
        )
        cls.spot = cls.bundle.spots[2]

    def test_every_legal_action_produces_terminal_bb_feedback(self) -> None:
        for action in self.spot.actions:
            table = VirtualDecisionTable(self.spot)
            observation = table.observation()
            self.assertFalse(observation["provider"]["external_actuation"])
            self.assertIn(action.action, observation["state"]["legal_actions"])
            feedback = table.step(action.action)
            self.assertEqual(feedback["units"], "BB")
            self.assertEqual(feedback["reward_bb"], format(action.ev, "f"))
            self.assertFalse(feedback["full_hand_completed"])
            self.assertEqual(len(feedback["feedback_fingerprint"]), 64)

    def test_illegal_duplicate_and_stack_exceeding_actions_are_rejected(self) -> None:
        table = VirtualDecisionTable(self.spot)
        with self.assertRaisesRegex(ValueError, "not legal"):
            table.step("fold")
        table.step("check")
        with self.assertRaisesRegex(ValueError, "already terminal"):
            table.step("check")
        oversized = replace(
            self.spot,
            actions=(ActionSolution("bet:100", Decimal("1"), Decimal("0")),),
        )
        with self.assertRaisesRegex(ValueError, "exceeds the effective stack"):
            VirtualDecisionTable(oversized).step("bet:100")

    def test_call_requires_explicit_positive_price(self) -> None:
        call_spot = replace(
            self.spot,
            actions=(ActionSolution("call", Decimal("1"), Decimal("0")),),
        )
        with self.assertRaisesRegex(ValueError, "requires explicit positive"):
            VirtualDecisionTable(call_spot)
        feedback = VirtualDecisionTable(call_spot, to_call_bb="2.5").step("call")
        self.assertEqual(feedback["command"]["amount_bb"], "2.5")

    def test_fixture_provider_audit_covers_all_nodes_and_actions(self) -> None:
        report = evaluate_virtual_table(
            ROOT / "coach" / "examples" / "sample_solver_bundle.json"
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["nodes"], 9)
        self.assertEqual(report["metrics"]["actions"], 21)
        self.assertEqual(report["metrics"]["illegal_action_rejections"], 9)
        self.assertEqual(report["metrics"]["duplicate_step_rejections"], 21)


if __name__ == "__main__":
    unittest.main()
