from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.calibrate_paired_value import (
    calibrate_paired_value,
    predict_calibrated_paired_value,
    validate_paired_value_calibration,
)
from apc.self_learning.paired_rollout_dataset import build_paired_rollout_dataset
from apc.self_learning.train_paired_value import train_paired_value_model


class PairedValueCalibrationTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path]:
        training = root / "training"
        calibration = root / "calibration"
        checkpoint = root / "checkpoint.json"
        build_paired_rollout_dataset(
            training,
            dataset_id="calibration-training-v1",
            rollouts=300,
            hand_seed_start=4000,
            minimum_rollouts=250,
            minimum_hand_classes=100,
        )
        build_paired_rollout_dataset(
            calibration,
            dataset_id="calibration-fresh-v1",
            rollouts=300,
            hand_seed_start=5000,
            minimum_rollouts=250,
            minimum_hand_classes=100,
        )
        train_paired_value_model(training, checkpoint)
        return training, calibration, checkpoint

    def test_fit_is_deterministic_fresh_and_nonactivating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, calibration, checkpoint = self.fixture(root)
            first = calibrate_paired_value(calibration, checkpoint, root / "first.json", bins=5)
            second = calibrate_paired_value(calibration, checkpoint, root / "second.json", bins=5)
            self.assertEqual(first["calibration_fingerprint"], second["calibration_fingerprint"])
            self.assertTrue(validate_paired_value_calibration(first)["valid"])
            self.assertFalse(first["calibration_gate"]["activation_authorized"])
            self.assertFalse(first["calibration_gate"]["recommendation_allowed"])
            self.assertFalse(first["selection"]["test_used_for_fit"])

    def test_supported_prediction_uses_gate_and_all_in_still_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, calibration_data, checkpoint_path = self.fixture(root)
            artifact = calibrate_paired_value(calibration_data, checkpoint_path, root / "calibration.json", bins=5)
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (calibration_data / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            call = next(row for row in rows if row["counterfactual_action"]["action"] == "call")
            result = predict_calibrated_paired_value(checkpoint, artifact, call["state"], call["counterfactual_action"])
            self.assertEqual(result["confidence_calibrated"], artifact["calibration_gate"]["passed"])
            self.assertFalse(result["recommendation_allowed"])
            all_in = next(row for row in rows if row["counterfactual_action"]["action"] == "all_in")
            rejected = predict_calibrated_paired_value(checkpoint, artifact, all_in["state"], all_in["counterfactual_action"])
            self.assertEqual(rejected["status"], "abstain_unsupported_or_invalid")

    def test_same_training_dataset_tamper_and_overwrite_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, calibration, checkpoint = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "must be independent"):
                calibrate_paired_value(training, checkpoint, root / "same.json", bins=5)
            output = root / "calibration.json"
            artifact = calibrate_paired_value(calibration, checkpoint, output, bins=5)
            artifact["calibration_gate"]["activation_authorized"] = True
            self.assertFalse(validate_paired_value_calibration(artifact)["valid"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                calibrate_paired_value(calibration, checkpoint, output, bins=5)


if __name__ == "__main__":
    unittest.main()
