from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.raised_postflop_rollout_dataset import (
    NODE_FAMILIES,
    build_raised_postflop_dataset,
)
from apc.self_learning.train_raised_postflop_value import (
    evaluate_raised_postflop_latency,
    prepare_raised_postflop_value,
    predict_raised_postflop_value,
    train_raised_postflop_value_model,
    validate_raised_postflop_value_checkpoint,
)
from apc.self_learning.train_value import _sha256


class RaisedPostflopValueTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.dataset = cls.root / "dataset"
        build_raised_postflop_dataset(
            cls.dataset,
            dataset_id="raised-postflop-value-fixture-v1",
            rollouts=35,
            hand_seed_start=25000,
            minimum_rollouts=20,
            minimum_texture_classes=10,
        )
        cls.first_path = cls.root / "first.json"
        cls.second_path = cls.root / "second.json"
        cls.first = train_raised_postflop_value_model(
            cls.dataset, cls.first_path
        )
        cls.second = train_raised_postflop_value_model(
            cls.dataset, cls.second_path
        )
        cls.rows = [
            json.loads(line)
            for line in (cls.dataset / "examples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_training_is_deterministic_and_all_slices_are_reported(self) -> None:
        self.assertEqual(
            self.first["checkpoint_fingerprint"],
            self.second["checkpoint_fingerprint"],
        )
        validation = validate_raised_postflop_value_checkpoint(self.first)
        self.assertTrue(validation["valid"], validation["issues"])
        test = self.first["metrics"]["test"]
        self.assertEqual(set(test["by_position"]), {"BTN", "BB"})
        self.assertEqual(set(test["by_node"]), set(NODE_FAMILIES))
        self.assertEqual(set(test["by_position_node"]), {"BTN", "BB"})
        for position in ("BTN", "BB"):
            self.assertEqual(
                set(test["by_position_node"][position]), set(NODE_FAMILIES)
            )
        self.assertGreaterEqual(
            test["paired_policy_value_bootstrap"]["independent_groups"], 1
        )
        self.assertFalse(self.first["recommendation_allowed"])
        self.assertFalse(self.first["activation_authorized"])

    def test_inference_covers_every_node_and_abstains_on_mismatch(self) -> None:
        test_rows = [row for row in self.rows if row["split"] == "test"]
        for position in ("BTN", "BB"):
            for node in NODE_FAMILIES:
                row = next(
                    item
                    for item in test_rows
                    if item["hero_position"] == position
                    and item["node_family"] == node
                )
                result = predict_raised_postflop_value(
                    self.first,
                    row["state"],
                    row["node_family"],
                    row["counterfactual_action_key"],
                    row["counterfactual_action"],
                    row["opponent_policy"],
                )
                repeated = predict_raised_postflop_value(
                    self.first,
                    row["state"],
                    row["node_family"],
                    row["counterfactual_action_key"],
                    row["counterfactual_action"],
                    row["opponent_policy"],
                )
                self.assertEqual(
                    result["status"],
                    "offline_raised_postflop_value_prediction_uncalibrated",
                )
                self.assertEqual(result["prediction_fingerprint"], repeated["prediction_fingerprint"])
                self.assertEqual(result["units"], "BB")
                self.assertEqual(result["hero_position"], position)
                self.assertEqual(result["node_family"], node)
                self.assertFalse(result["recommendation_allowed"])

        row = next(item for item in test_rows if item["node_family"] == "lead")
        rejected = predict_raised_postflop_value(
            self.first,
            row["state"],
            row["node_family"],
            "raise_3x",
            row["counterfactual_action"],
            row["opponent_policy"],
        )
        self.assertEqual(rejected["status"], "abstain_unsupported_or_invalid")
        self.assertIn("action_key_not_supported", rejected["reasons"])
        self.assertIsNone(rejected["predicted_terminal_return_bb"])

    def test_latency_and_semantic_gate_tamper_detection(self) -> None:
        detached_source = copy.deepcopy(self.first)
        prepared = prepare_raised_postflop_value(detached_source)
        original_fingerprint = prepared._checkpoint["checkpoint_fingerprint"]
        detached_source["status"] = "tampered_after_prepare"
        self.assertEqual(
            prepared._checkpoint["checkpoint_fingerprint"], original_fingerprint
        )
        report = evaluate_raised_postflop_latency(
            self.dataset, prepared, repetitions=20
        )
        self.assertTrue(report["latency_gate"]["passed"], report)
        self.assertGreater(report["unique_prediction_fingerprints"], 1)

        tampered = copy.deepcopy(self.first)
        tampered["generalization_gate"]["passed"] = not tampered[
            "generalization_gate"
        ]["passed"]
        tampered.pop("checkpoint_fingerprint")
        tampered["checkpoint_fingerprint"] = _sha256(tampered)
        validation = validate_raised_postflop_value_checkpoint(tampered)
        self.assertFalse(validation["valid"])
        self.assertTrue(
            any("generalization gate" in issue for issue in validation["issues"])
        )

        tampered = copy.deepcopy(self.first)
        tampered["recommendation_allowed"] = True
        tampered.pop("checkpoint_fingerprint")
        tampered["checkpoint_fingerprint"] = _sha256(tampered)
        self.assertFalse(
            validate_raised_postflop_value_checkpoint(tampered)["valid"]
        )


if __name__ == "__main__":
    unittest.main()
