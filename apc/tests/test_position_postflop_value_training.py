from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.postflop_position_rollout_dataset import build_postflop_position_dataset
from apc.self_learning.train_position_postflop_value import (
    evaluate_position_postflop_latency,
    predict_position_postflop_value,
    train_position_postflop_value_model,
    validate_position_postflop_value_checkpoint,
)


class PositionPostflopValueTrainingTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "dataset"
        checkpoint = root / "checkpoint.json"
        build_postflop_position_dataset(
            dataset,
            dataset_id="position-value-fixture-v1",
            rollouts=30,
            hand_seed_start=9400,
            minimum_rollouts=20,
            minimum_texture_classes=10,
        )
        train_position_postflop_value_model(dataset, checkpoint)
        return dataset, checkpoint

    def test_training_is_deterministic_and_evaluates_both_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            build_postflop_position_dataset(
                dataset,
                dataset_id="position-value-determinism-v1",
                rollouts=30,
                hand_seed_start=9700,
                minimum_rollouts=20,
                minimum_texture_classes=10,
            )
            first = train_position_postflop_value_model(dataset, root / "first.json")
            second = train_position_postflop_value_model(dataset, root / "second.json")
            self.assertEqual(first["checkpoint_fingerprint"], second["checkpoint_fingerprint"])
            self.assertTrue(validate_position_postflop_value_checkpoint(first)["valid"])
            self.assertEqual(set(first["metrics"]["test"]["by_position"]), {"BTN", "BB"})
            self.assertEqual(first["configuration"]["unsupported_actions"], ["all_in"])
            self.assertFalse(first["activation_authorized"])

    def test_prediction_conditions_on_position_and_abstains_on_all_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint_path = self.fixture(Path(directory))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (dataset / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            predictions = {}
            for position in ("BTN", "BB"):
                row = next(
                    row
                    for row in rows
                    if row["split"] == "test"
                    and row["hero_position"] == position
                    and row["opponent_policy"] == "made_hand_selective"
                    and row["counterfactual_action"]["action"] == "bet"
                )
                result = predict_position_postflop_value(
                    checkpoint, row["state"], row["counterfactual_action"], row["opponent_policy"]
                )
                self.assertEqual(result["status"], "offline_position_postflop_value_prediction_uncalibrated")
                self.assertEqual(result["hero_position"], position)
                self.assertFalse(result["recommendation_allowed"])
                predictions[position] = result["prediction_fingerprint"]
            self.assertNotEqual(predictions["BTN"], predictions["BB"])
            all_in = next(row for row in rows if row["counterfactual_action"]["action"] == "all_in")
            rejected = predict_position_postflop_value(
                checkpoint, all_in["state"], all_in["counterfactual_action"], all_in["opponent_policy"]
            )
            self.assertEqual(rejected["status"], "abstain_unsupported_or_invalid")
            self.assertIn("action_not_supported", rejected["reasons"])

    def test_latency_gate_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint_path = self.fixture(Path(directory))
            report = evaluate_position_postflop_latency(dataset, checkpoint_path, repetitions=20)
            self.assertTrue(report["latency_gate"]["passed"], report)
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["recommendation_allowed"] = True
            self.assertFalse(validate_position_postflop_value_checkpoint(checkpoint)["valid"])


if __name__ == "__main__":
    unittest.main()
