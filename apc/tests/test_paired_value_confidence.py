from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apc.self_learning.evaluate_paired_value_confidence import evaluate_paired_value_confidence
from apc.self_learning.paired_rollout_dataset import build_paired_rollout_dataset
from apc.self_learning.train_paired_value import train_paired_value_model


class PairedValueConfidenceTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "dataset"
        checkpoint = root / "checkpoint.json"
        build_paired_rollout_dataset(
            dataset,
            dataset_id="confidence-fixture-v1",
            rollouts=300,
            hand_seed_start=2500,
            minimum_rollouts=250,
            minimum_hand_classes=100,
        )
        train_paired_value_model(dataset, checkpoint)
        return dataset, checkpoint

    def test_bootstrap_is_deterministic_paired_and_nonactivating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint = self.fixture(Path(directory))
            first = evaluate_paired_value_confidence(dataset, checkpoint, bootstrap_samples=300)
            second = evaluate_paired_value_confidence(dataset, checkpoint, bootstrap_samples=300)
            self.assertEqual(first["evaluation_fingerprint"], second["evaluation_fingerprint"])
            self.assertEqual(first["improvement_intervals"], second["improvement_intervals"])
            self.assertEqual(first["test_examples"], first["test_groups"] * 2)
            self.assertFalse(first["promotion_eligible"])
            self.assertFalse(first["confidence_gate"]["activation_authorized"])
            self.assertGreaterEqual(first["validated_lookup_latency_ms"]["p95"], 0)

    def test_calibration_and_action_specific_intervals_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint = self.fixture(Path(directory))
            report = evaluate_paired_value_confidence(
                dataset, checkpoint, bootstrap_samples=200, calibration_bins=5
            )
            self.assertTrue(report["passed"])
            self.assertEqual(set(report["improvement_intervals"]), {"call", "raise", "aggregate"})
            self.assertEqual(len(report["calibration"]["call"]["bins"]), 5)
            self.assertEqual(len(report["calibration"]["raise"]["bins"]), 5)
            self.assertGreaterEqual(float(report["exact_hand_class_coverage"]), 0.75)
            self.assertLessEqual(float(report["exact_hand_class_coverage"]), 1.0)

    def test_checkpoint_dataset_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, checkpoint = self.fixture(root)
            other = root / "other"
            build_paired_rollout_dataset(
                other,
                dataset_id="other-v1",
                rollouts=300,
                hand_seed_start=3500,
                minimum_rollouts=250,
                minimum_hand_classes=100,
            )
            with self.assertRaisesRegex(ValueError, "fingerprints do not match"):
                evaluate_paired_value_confidence(other, checkpoint, bootstrap_samples=200)


if __name__ == "__main__":
    unittest.main()
