from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.full_hand_dataset import build_full_hand_dataset, validate_full_hand_dataset


class FullHandDatasetTests(unittest.TestCase):
    def build(self, destination: Path) -> dict[str, object]:
        return build_full_hand_dataset(
            destination,
            dataset_id="full-hand-fixture-v1",
            hands=30,
            hand_seed_start=700,
            minimum_examples=20,
            minimum_groups=20,
        )

    def test_build_is_bb_only_grouped_private_and_training_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset"
            manifest = self.build(destination)
            report = validate_full_hand_dataset(destination)
            self.assertTrue(report["valid"], report["issues"])
            self.assertTrue(manifest["training_eligible"])
            self.assertFalse(manifest["policy_promotion_eligible"])
            self.assertEqual(manifest["group_count"], 30)
            self.assertTrue(all(manifest["split_counts"][key] for key in ("train", "validation", "test")))
            rows = [json.loads(line) for line in (destination / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["units"] == "BB" for row in rows))
            self.assertTrue(all(row["state"]["opponent_cards"] is None for row in rows))
            self.assertTrue(all(row["learning_signal"]["solver_target"] is False for row in rows))
            groups = {}
            for row in rows:
                self.assertEqual(groups.setdefault(row["group_id"], row["split"]), row["split"])

    def test_build_is_deterministic_and_destination_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_manifest = self.build(first)
            second_manifest = self.build(second)
            self.assertEqual(first_manifest["dataset_fingerprint"], second_manifest["dataset_fingerprint"])
            self.assertEqual((first / "examples.jsonl").read_bytes(), (second / "examples.jsonl").read_bytes())
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.build(first)

    def test_example_and_manifest_tampering_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset"
            self.build(destination)
            path = destination / "examples.jsonl"
            rows = path.read_text(encoding="utf-8").splitlines()
            row = json.loads(rows[0])
            row["learning_signal"]["hero_return_bb"] = "999"
            rows[0] = json.dumps(row)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            report = validate_full_hand_dataset(destination)
            self.assertFalse(report["valid"])
            self.assertTrue(any("fingerprint" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
