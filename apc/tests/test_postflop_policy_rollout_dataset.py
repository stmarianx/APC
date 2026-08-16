from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.postflop_policy_rollout_dataset import (
    ACTIONS,
    OPPONENT_POLICIES,
    STREETS,
    build_postflop_policy_dataset,
    validate_postflop_policy_dataset,
)


class PostflopPolicyRolloutDatasetTests(unittest.TestCase):
    def build(self, destination: Path) -> dict[str, object]:
        return build_postflop_policy_dataset(
            destination,
            dataset_id="postflop-policy-fixture-v1",
            rollouts=30,
            hand_seed_start=7000,
            minimum_rollouts=20,
            minimum_texture_classes=10,
        )

    def test_policy_state_pairing_and_selective_action_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset"
            manifest = self.build(destination)
            report = validate_postflop_policy_dataset(destination)
            self.assertTrue(report["valid"], report["issues"])
            self.assertTrue(manifest["training_eligible"])
            self.assertEqual(manifest["example_count"], 810)
            self.assertEqual(report["policy_state_count"], 270)
            self.assertEqual(manifest["streets"], list(STREETS))
            self.assertEqual(manifest["opponent_policies"], list(OPPONENT_POLICIES))
            self.assertTrue(all(key in manifest["opponent_action_counts"] for key in manifest["eligibility"]["required_selective_actions"]))
            rows = [json.loads(line) for line in (destination / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            policy_states = {}
            for row in rows:
                group = policy_states.setdefault(row["policy_state_id"], {"actions": set(), "states": set()})
                group["actions"].add(row["counterfactual_action"]["action"])
                group["states"].add(row["provenance"]["pre_state_fingerprint"])
                self.assertIsNone(row["state"]["opponent_cards"])
            self.assertTrue(all(group["actions"] == set(ACTIONS) and len(group["states"]) == 1 for group in policy_states.values()))

    def test_build_is_deterministic_and_street_differentiated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_manifest = self.build(first)
            second_manifest = self.build(second)
            self.assertEqual(first_manifest["dataset_fingerprint"], second_manifest["dataset_fingerprint"])
            self.assertEqual(first_manifest["street_differentiation"], second_manifest["street_differentiation"])
            selective = [
                float(row["max_minus_min_street_mean_bb"])
                for key, row in first_manifest["street_differentiation"].items()
                if key.startswith("made_hand_selective:")
            ]
            self.assertTrue(any(value > 0 for value in selective))
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.build(first)

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset"
            self.build(destination)
            path = destination / "examples.jsonl"
            rows = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["opponent_policy"] = "unknown"
            rows[0] = json.dumps(first)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            report = validate_postflop_policy_dataset(destination)
            self.assertFalse(report["valid"])
            self.assertTrue(any("fingerprint" in issue or "policy" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
