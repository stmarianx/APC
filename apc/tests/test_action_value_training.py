from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.full_hand_dataset import build_full_hand_dataset
from apc.self_learning.train_action_value import (
    action_value_features,
    predict_action_value,
    train_action_value_model,
    validate_action_value_checkpoint,
)


class ActionValueTrainingTests(unittest.TestCase):
    def build_dataset(self, path: Path) -> None:
        build_full_hand_dataset(
            path,
            dataset_id="action-value-fixture-v1",
            hands=40,
            hand_seed_start=1200,
            minimum_examples=30,
            minimum_groups=30,
        )

    def test_training_is_deterministic_validation_selected_and_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            first = train_action_value_model(dataset, Path(directory) / "first.json", epochs=5, feature_dimension=128)
            second = train_action_value_model(dataset, Path(directory) / "second.json", epochs=5, feature_dimension=128)
            self.assertEqual(first["checkpoint_fingerprint"], second["checkpoint_fingerprint"])
            self.assertTrue(validate_action_value_checkpoint(first)["valid"])
            self.assertFalse(first["activation_authorized"])
            self.assertFalse(first["recommendation_allowed"])
            self.assertGreaterEqual(first["configuration"]["selected_epoch_by_validation"], 0)
            for split in ("train", "validation", "test"):
                self.assertGreater(first["metrics"][split]["examples"], 0)

    def test_prediction_requires_exactly_legal_visible_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            checkpoint = train_action_value_model(dataset, Path(directory) / "model.json", epochs=2, feature_dimension=128)
            row = json.loads((dataset / "examples.jsonl").read_text(encoding="utf-8").splitlines()[0])
            result = predict_action_value(checkpoint, row["state"], row["behavior"]["chosen_action"])
            self.assertEqual(result["status"], "offline_action_value_prediction_uncalibrated")
            self.assertEqual(len(result["prediction_fingerprint"]), 64)
            rejected = predict_action_value(checkpoint, row["state"], {"action": "not_legal"})
            self.assertEqual(rejected["status"], "abstain_invalid_state_or_action")
            self.assertIsNone(rejected["predicted_terminal_return_bb"])

    def test_checkpoint_tamper_and_overwrite_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            output = Path(directory) / "model.json"
            checkpoint = train_action_value_model(dataset, output, epochs=2, feature_dimension=64)
            checkpoint["activation_authorized"] = True
            self.assertFalse(validate_action_value_checkpoint(checkpoint)["valid"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                train_action_value_model(dataset, output, epochs=2, feature_dimension=64)

    def test_features_are_suit_renaming_invariant_and_action_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            rows = [json.loads(line) for line in (dataset / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            row = next(row for row in rows if len(row["state"]["board"]) >= 3)
            state = row["state"]
            command = row["behavior"]["chosen_action"]
            renamed = json.loads(json.dumps(state))
            mapping = {"c": "d", "d": "h", "h": "s", "s": "c"}
            renamed["hero_cards"] = [card[:-1] + mapping[card[-1]] for card in state["hero_cards"]]
            renamed["board"] = [card[:-1] + mapping[card[-1]] for card in state["board"]]
            self.assertEqual(action_value_features(state, command, 128), action_value_features(renamed, command, 128))
            alternate = next(row["action_buttons"] for row in [state] if len(row["action_buttons"]) > 1)[0]
            if alternate["action"] != command["action"]:
                alternate_command = {"action": alternate["action"]}
                if "amount_bb" in alternate:
                    alternate_command["amount_bb"] = alternate["amount_bb"]
                if "to_amount_bb" in alternate:
                    alternate_command["to_amount_bb"] = alternate["to_amount_bb"]
                self.assertNotEqual(action_value_features(state, command, 128), action_value_features(state, alternate_command, 128))


if __name__ == "__main__":
    unittest.main()
