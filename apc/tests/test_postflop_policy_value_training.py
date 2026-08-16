from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.postflop_policy_rollout_dataset import build_postflop_policy_dataset
from apc.self_learning.train_postflop_policy_value import (
    evaluate_postflop_policy_value_latency,
    predict_postflop_policy_value,
    train_postflop_policy_value_model,
    validate_postflop_policy_value_checkpoint,
)


class PostflopPolicyValueTrainingTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "dataset"
        checkpoint = root / "checkpoint.json"
        build_postflop_policy_dataset(
            dataset,
            dataset_id="postflop-policy-value-fixture-v1",
            rollouts=30,
            hand_seed_start=7300,
            minimum_rollouts=20,
            minimum_texture_classes=10,
        )
        train_postflop_policy_value_model(dataset, checkpoint)
        return dataset, checkpoint

    def test_training_is_deterministic_and_complete_hand_split_evaluated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            build_postflop_policy_dataset(
                dataset,
                dataset_id="postflop-policy-value-determinism-v1",
                rollouts=30,
                hand_seed_start=7600,
                minimum_rollouts=20,
                minimum_texture_classes=10,
            )
            first = train_postflop_policy_value_model(dataset, root / "first.json")
            second = train_postflop_policy_value_model(dataset, root / "second.json")
            self.assertEqual(first["checkpoint_fingerprint"], second["checkpoint_fingerprint"])
            self.assertTrue(validate_postflop_policy_value_checkpoint(first)["valid"])
            self.assertEqual(first["configuration"]["unsupported_actions"], ["all_in"])
            self.assertTrue(all(first["metrics"][split]["policy_states"] > 0 for split in ("train", "validation", "test")))
            self.assertFalse(first["activation_authorized"])

    def test_prediction_uses_declared_policy_and_abstains_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, checkpoint_path = self.fixture(root)
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (dataset / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            row = next(row for row in rows if row["split"] == "test" and row["counterfactual_action"]["action"] == "bet")
            result = predict_postflop_policy_value(
                checkpoint, row["state"], row["counterfactual_action"], row["opponent_policy"]
            )
            self.assertEqual(result["status"], "offline_postflop_policy_value_prediction_uncalibrated")
            self.assertEqual(result["opponent_policy"], row["opponent_policy"])
            self.assertFalse(result["recommendation_allowed"])
            rejected = predict_postflop_policy_value(
                checkpoint, row["state"], {"action": "all_in", "to_amount_bb": "99"}, "unknown"
            )
            self.assertEqual(rejected["status"], "abstain_unsupported_or_invalid")
            self.assertIn("opponent_policy_not_supported", rejected["reasons"])
            self.assertIn("action_not_supported", rejected["reasons"])

    def test_latency_gate_and_checkpoint_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, checkpoint_path = self.fixture(root)
            report = evaluate_postflop_policy_value_latency(dataset, checkpoint_path, repetitions=20)
            self.assertTrue(report["latency_gate"]["passed"], report)
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["activation_authorized"] = True
            self.assertFalse(validate_postflop_policy_value_checkpoint(checkpoint)["valid"])


if __name__ == "__main__":
    unittest.main()
