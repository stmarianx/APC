from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from apc.self_learning.evaluate_paired_policy import bootstrap_mean_ci, evaluate_candidate_against_uniform
from apc.self_learning.evaluate_candidate_smoke import evaluate_candidate_smoke
from apc.self_learning.train_candidate import predict_candidate


ROOT = Path(__file__).resolve().parents[2]
SOLVER = ROOT / "coach" / "examples" / "sample_solver_bundle.json"
HAND = ROOT / "coach" / "examples" / "sample_play_money_hand.txt"


class PairedPolicyEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        smoke = Path(cls.temporary.name) / "smoke"
        evaluate_candidate_smoke(SOLVER, HAND, smoke, replicas=30)
        cls.checkpoint_path = smoke / "candidate.json"
        cls.checkpoint = json.loads(cls.checkpoint_path.read_text(encoding="utf-8"))
        cls.state = {
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
            "legal_actions": ["check", "bet:0.33", "bet:0.62"],
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_checkpoint_inference_is_deterministic_and_legality_gated(self) -> None:
        legal = ["check", "bet:0.33", "bet:0.62"]
        first = predict_candidate(self.checkpoint, self.state, legal)
        second = predict_candidate(self.checkpoint, self.state, legal)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "prediction_ready")
        self.assertAlmostEqual(sum(float(value) for value in first["probabilities"].values()), 1.0)
        rejected = predict_candidate(self.checkpoint, self.state, ["check", "bet:0.75"])
        self.assertEqual(rejected["status"], "abstain_unsupported_actions")
        self.assertFalse(rejected["activation_authorized"])

    def test_bootstrap_interval_is_deterministic_and_bb_positive(self) -> None:
        differences = [Decimal("0.10"), Decimal("0.20"), Decimal("0.30")]
        first = bootstrap_mean_ci(differences, samples=1000)
        second = bootstrap_mean_ci(differences, samples=1000)
        self.assertEqual(first, second)
        self.assertGreater(Decimal(first["lower_bb"]), 0)

    def test_smoke_candidate_has_insufficient_coverage_for_promotion(self) -> None:
        report = evaluate_candidate_against_uniform(
            self.checkpoint_path,
            SOLVER,
            bootstrap_samples=1000,
        )
        self.assertTrue(report["passed"])
        self.assertFalse(report["promotion_eligible"])
        self.assertFalse(report["promotion_gate"]["passed"])
        self.assertEqual(report["coverage"]["evaluated_nodes"], 1)
        self.assertEqual(report["coverage"]["unsupported_nodes"], 8)
        self.assertIn("uniform_reference_is_not_a_declared_incumbent", report["promotion_gate"]["reasons"])
        self.assertFalse(report["promotion_gate"]["activation_authorized"])


if __name__ == "__main__":
    unittest.main()
