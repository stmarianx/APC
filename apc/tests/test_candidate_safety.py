from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apc.self_learning.evaluate_candidate_safety import evaluate_candidate_safety
from apc.self_learning.evaluate_candidate_smoke import evaluate_candidate_smoke
from apc.self_learning.train_candidate import predict_candidate


ROOT = Path(__file__).resolve().parents[2]
SOLVER = ROOT / "coach" / "examples" / "sample_solver_bundle.json"
HAND = ROOT / "coach" / "examples" / "sample_play_money_hand.txt"


class CandidateSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        smoke = Path(cls.temporary.name) / "smoke"
        evaluate_candidate_smoke(SOLVER, HAND, smoke, replicas=30)
        cls.checkpoint = smoke / "candidate.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_adversarial_audit_rejects_every_invalid_state(self) -> None:
        report = evaluate_candidate_safety(self.checkpoint, SOLVER)
        self.assertTrue(report["passed"])
        self.assertFalse(report["promotion_eligible"])
        self.assertFalse(report["paired_incumbent_non_regression"])
        self.assertEqual(report["metrics"]["adversarial_invalid_states"], 10)
        self.assertEqual(report["metrics"]["invalid_state_acceptances"], 0)
        self.assertEqual(report["metrics"]["recommendation_or_activation_violations"], 0)
        self.assertTrue(all(row["status"] == "abstain_invalid_state" for row in report["adversarial_rows"]))

    def test_missing_state_legal_actions_abstains_without_probabilities(self) -> None:
        state = {
            "game": "holdem_no_limit",
            "players": 2,
            "hero_position": "BTN",
            "effective_stack_bb": "97",
            "pot_bb": "6.5",
            "to_call_bb": "0",
            "board": ["Ah", "7c", "2d"],
            "hero_cards": ["As", "Kd"],
            "action_history": ["BTN raise_to:3", "BB call", "BB check"],
            "rake_model": "play_money_fixture",
            "utility_model": "chip_ev",
        }
        result = predict_candidate(
            self.checkpoint,
            state,
            ["check", "bet:0.33", "bet:0.62"],
        )
        self.assertEqual(result["status"], "abstain_invalid_state")
        self.assertIn("legal_actions_do_not_match_state", result["reasons"])
        self.assertIsNone(result["probabilities"])
        self.assertFalse(result["recommendation_allowed"])


if __name__ == "__main__":
    unittest.main()
