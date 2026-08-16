from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.paired_rollout_dataset import build_paired_rollout_dataset
from apc.self_learning.train_paired_value import (
    predict_paired_value,
    train_paired_value_model,
    validate_paired_value_checkpoint,
)


class PairedValueTrainingTests(unittest.TestCase):
    def build_dataset(self, path: Path) -> None:
        build_paired_rollout_dataset(
            path,
            dataset_id="paired-value-fixture-v1",
            rollouts=120,
            hand_seed_start=2000,
            minimum_rollouts=100,
            minimum_hand_classes=50,
        )

    def test_training_is_deterministic_validation_selected_and_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            first = train_paired_value_model(dataset, Path(directory) / "first.json")
            second = train_paired_value_model(dataset, Path(directory) / "second.json")
            self.assertEqual(first["checkpoint_fingerprint"], second["checkpoint_fingerprint"])
            self.assertTrue(validate_paired_value_checkpoint(first)["valid"])
            self.assertFalse(first["activation_authorized"])
            self.assertFalse(first["recommendation_allowed"])
            self.assertIn(first["configuration"]["selected_shrinkage_by_validation"], first["configuration"]["shrinkage_grid"])
            for split in ("train", "validation", "test"):
                self.assertGreater(first["metrics"][split]["examples"], 0)

    def test_supported_prediction_and_all_in_abstention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            checkpoint = train_paired_value_model(dataset, Path(directory) / "model.json")
            rows = [json.loads(line) for line in (dataset / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            call = next(row for row in rows if row["counterfactual_action"]["action"] == "call")
            accepted = predict_paired_value(checkpoint, call["state"], call["counterfactual_action"])
            self.assertEqual(accepted["status"], "offline_paired_value_prediction_uncalibrated")
            self.assertFalse(accepted["recommendation_allowed"])
            all_in = next(row for row in rows if row["counterfactual_action"]["action"] == "all_in")
            rejected = predict_paired_value(checkpoint, all_in["state"], all_in["counterfactual_action"])
            self.assertEqual(rejected["status"], "abstain_unsupported_or_invalid")
            self.assertIn("action_not_supported", rejected["reasons"])

    def test_checkpoint_tampering_and_overwrite_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            output = Path(directory) / "model.json"
            checkpoint = train_paired_value_model(dataset, output)
            checkpoint["recommendation_allowed"] = True
            self.assertFalse(validate_paired_value_checkpoint(checkpoint)["valid"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                train_paired_value_model(dataset, output)


if __name__ == "__main__":
    unittest.main()
