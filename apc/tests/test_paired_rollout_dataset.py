from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.paired_rollout_dataset import (
    ACTION_ORDER,
    build_paired_rollout_dataset,
    validate_paired_rollout_dataset,
)


class PairedRolloutDatasetTests(unittest.TestCase):
    def build(self, destination: Path) -> dict[str, object]:
        return build_paired_rollout_dataset(
            destination,
            dataset_id="paired-fixture-v1",
            rollouts=30,
            hand_seed_start=1500,
            minimum_rollouts=20,
            minimum_hand_classes=10,
        )

    def test_every_group_has_same_state_and_all_counterfactual_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "paired"
            manifest = self.build(destination)
            report = validate_paired_rollout_dataset(destination)
            self.assertTrue(report["valid"], report["issues"])
            self.assertTrue(manifest["training_eligible"])
            self.assertEqual(manifest["example_count"], 120)
            rows = [json.loads(line) for line in (destination / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            groups = {}
            for row in rows:
                group = groups.setdefault(row["group_id"], {"actions": set(), "states": set(), "splits": set()})
                group["actions"].add(row["counterfactual_action"]["action"])
                group["states"].add(row["provenance"]["pre_state_fingerprint"])
                group["splits"].add(row["split"])
                self.assertIsNone(row["state"]["opponent_cards"])
            self.assertTrue(all(group["actions"] == set(ACTION_ORDER) for group in groups.values()))
            self.assertTrue(all(len(group["states"]) == len(group["splits"]) == 1 for group in groups.values()))
            fold_returns = [row["learning_signal"]["hero_return_bb"] for row in rows if row["counterfactual_action"]["action"] == "fold"]
            self.assertEqual(set(fold_returns), {"-0.5"})

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

    def test_tampering_breaks_file_and_example_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "paired"
            self.build(destination)
            path = destination / "examples.jsonl"
            rows = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["learning_signal"]["hero_return_bb"] = "99"
            rows[0] = json.dumps(first)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            report = validate_paired_rollout_dataset(destination)
            self.assertFalse(report["valid"])
            self.assertTrue(any("fingerprint" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
