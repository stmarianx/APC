from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.postflop_position_rollout_dataset import (
    HERO_POSITIONS,
    build_postflop_position_dataset,
    validate_postflop_position_dataset,
)
from apc.self_learning.postflop_policy_rollout_dataset import OPPONENT_POLICIES
from apc.self_learning.postflop_paired_rollout_dataset import ACTIONS, STREETS


class PostflopPositionRolloutDatasetTests(unittest.TestCase):
    def build(self, destination: Path) -> dict[str, object]:
        return build_postflop_position_dataset(
            destination,
            dataset_id="postflop-position-fixture-v1",
            rollouts=30,
            hand_seed_start=9000,
            minimum_rollouts=20,
            minimum_texture_classes=10,
        )

    def test_both_positions_are_card_matched_private_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset"
            manifest = self.build(destination)
            report = validate_postflop_position_dataset(destination)
            self.assertTrue(report["valid"], report["issues"])
            self.assertTrue(manifest["training_eligible"], manifest["eligibility"])
            self.assertEqual(manifest["example_count"], 1620)
            self.assertEqual(report["position_state_count"], 180)
            self.assertEqual(report["policy_state_count"], 540)
            self.assertEqual(manifest["hero_positions"], list(HERO_POSITIONS))
            self.assertTrue(all(key in manifest["opponent_action_counts"] for key in manifest["eligibility"]["required_selective_actions"]))
            rows = [json.loads(line) for line in (destination / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
            groups = {}
            for row in rows:
                self.assertIsNone(row["state"]["opponent_cards"])
                self.assertEqual(row["state"]["hero_position"], row["hero_position"])
                self.assertEqual(row["state"]["next_actor"], "Hero")
                key = (row["group_id"], row["street"], row["hero_position"])
                groups.setdefault(key, set()).add(tuple([*row["state"]["hero_cards"], *row["state"]["board"]]))
            for group_id in {row["group_id"] for row in rows}:
                for street in STREETS:
                    self.assertEqual(groups[(group_id, street, "BTN")], groups[(group_id, street, "BB")])

    def test_build_is_deterministic_and_position_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_manifest = self.build(first)
            second_manifest = self.build(second)
            self.assertEqual(first_manifest["dataset_fingerprint"], second_manifest["dataset_fingerprint"])
            self.assertEqual(first_manifest["bb_minus_btn_action_advantage_bb"], second_manifest["bb_minus_btn_action_advantage_bb"])
            for position in HERO_POSITIONS:
                for policy in OPPONENT_POLICIES:
                    for street in STREETS:
                        self.assertEqual(
                            set(first_manifest["paired_comparisons"][position][policy][street]),
                            {"bet_minus_check", "all_in_minus_check"},
                        )
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.build(first)

    def test_tampered_position_or_card_match_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset"
            self.build(destination)
            path = destination / "examples.jsonl"
            rows = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["hero_position"] = "BTN" if first["hero_position"] == "BB" else "BB"
            rows[0] = json.dumps(first)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            report = validate_postflop_position_dataset(destination)
            self.assertFalse(report["valid"])
            self.assertTrue(any("fingerprint" in issue or "position" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
