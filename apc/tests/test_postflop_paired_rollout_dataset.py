from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.postflop_paired_rollout_dataset import (
    ACTIONS,
    STREETS,
    build_postflop_paired_dataset,
    validate_postflop_paired_dataset,
)


class PostflopPairedRolloutDatasetTests(unittest.TestCase):
    def build(self, destination: Path) -> dict[str, object]:
        return build_postflop_paired_dataset(
            destination,
            dataset_id="postflop-paired-fixture-v1",
            rollouts=30,
            hand_seed_start=6000,
            minimum_rollouts=20,
            minimum_texture_classes=10,
        )

    def test_all_streets_actions_are_same_state_paired_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset"
            manifest = self.build(destination)
            report = validate_postflop_paired_dataset(destination)
            self.assertTrue(report["valid"], report["issues"])
            self.assertTrue(manifest["training_eligible"])
            self.assertEqual(manifest["example_count"], 270)
            self.assertEqual(report["state_count"], 90)
            rows = [json.loads(line) for line in (destination / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            states = {}
            for row in rows:
                state = states.setdefault(row["state_id"], {"actions": set(), "fingerprints": set()})
                state["actions"].add(row["counterfactual_action"]["action"])
                state["fingerprints"].add(row["provenance"]["pre_state_fingerprint"])
                self.assertIsNone(row["state"]["opponent_cards"])
            self.assertTrue(all(state["actions"] == set(ACTIONS) and len(state["fingerprints"]) == 1 for state in states.values()))
            self.assertEqual({row["street"] for row in rows}, set(STREETS))

    def test_build_is_deterministic_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_manifest = self.build(first)
            second_manifest = self.build(second)
            self.assertEqual(first_manifest["dataset_fingerprint"], second_manifest["dataset_fingerprint"])
            self.assertEqual(first_manifest["variance_audit"], second_manifest["variance_audit"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.build(first)

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset"
            self.build(destination)
            path = destination / "examples.jsonl"
            rows = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["state"]["opponent_cards"] = ["As", "Ad"]
            rows[0] = json.dumps(first)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            report = validate_postflop_paired_dataset(destination)
            self.assertFalse(report["valid"])
            self.assertTrue(any("fingerprint" in issue or "private" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
