from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.full_hand_dataset import build_full_hand_dataset
from apc.self_learning.train_value import predict_value, train_value_model, validate_value_checkpoint, value_features


class ValueTrainingTests(unittest.TestCase):
    def build_dataset(self, path: Path) -> dict[str, object]:
        return build_full_hand_dataset(
            path,
            dataset_id="value-fixture-v1",
            hands=30,
            hand_seed_start=900,
            minimum_examples=20,
            minimum_groups=20,
        )

    def test_training_is_deterministic_group_evaluated_and_unpromoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            first = train_value_model(dataset, Path(directory) / "first.json", epochs=10, feature_dimension=64)
            second = train_value_model(dataset, Path(directory) / "second.json", epochs=10, feature_dimension=64)
            self.assertEqual(first["checkpoint_fingerprint"], second["checkpoint_fingerprint"])
            self.assertTrue(validate_value_checkpoint(first)["valid"])
            self.assertFalse(first["activation_authorized"])
            self.assertFalse(first["recommendation_allowed"])
            for split in ("train", "validation", "test"):
                self.assertGreater(first["metrics"][split]["examples"], 0)
                self.assertEqual(len(first["metrics"][split]["calibration_bins"]), 5)

    def test_inference_is_fingerprinted_and_invalid_state_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            checkpoint = train_value_model(dataset, Path(directory) / "value.json", epochs=3, feature_dimension=64)
            example = json.loads((dataset / "examples.jsonl").read_text(encoding="utf-8").splitlines()[0])
            result = predict_value(checkpoint, example["state"])
            self.assertEqual(result["status"], "offline_value_prediction_uncalibrated")
            self.assertEqual(len(result["prediction_fingerprint"]), 64)
            self.assertFalse(result["recommendation_allowed"])
            invalid = dict(example["state"])
            invalid["opponent_cards"] = ["As", "Ad"]
            rejected = predict_value(checkpoint, invalid)
            self.assertEqual(rejected["status"], "abstain_invalid_state")
            self.assertIsNone(rejected["predicted_terminal_return_bb"])

    def test_checkpoint_tamper_and_overwrite_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            output = Path(directory) / "value.json"
            checkpoint = train_value_model(dataset, output, epochs=2, feature_dimension=32)
            checkpoint["recommendation_allowed"] = True
            report = validate_value_checkpoint(checkpoint)
            self.assertFalse(report["valid"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                train_value_model(dataset, output, epochs=2, feature_dimension=32)

    def test_feature_schema_is_suit_renaming_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            row = json.loads((dataset / "examples.jsonl").read_text(encoding="utf-8").splitlines()[5])
            state = row["state"]
            renamed = json.loads(json.dumps(state))
            mapping = {"c": "d", "d": "h", "h": "s", "s": "c"}
            renamed["hero_cards"] = [card[:-1] + mapping[card[-1]] for card in state["hero_cards"]]
            renamed["board"] = [card[:-1] + mapping[card[-1]] for card in state["board"]]
            self.assertEqual(value_features(state, 64), value_features(renamed, 64))


if __name__ == "__main__":
    unittest.main()
