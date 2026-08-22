from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path

from apc.self_learning.postflop_paired_rollout_dataset import STREETS
from apc.self_learning.postflop_policy_rollout_dataset import OPPONENT_POLICIES
from apc.self_learning.postflop_position_rollout_dataset import HERO_POSITIONS
from apc.self_learning.raised_postflop_rollout_dataset import (
    ACTION_KEYS_BY_NODE,
    NODE_FAMILIES,
    build_raised_postflop_dataset,
    validate_raised_postflop_dataset,
)
from apc.self_learning.replay_dataset import _canonical, _sha256_bytes


class RaisedPostflopRolloutDatasetTests(unittest.TestCase):
    def build(self, destination: Path) -> dict[str, object]:
        return build_raised_postflop_dataset(
            destination,
            dataset_id="raised-postflop-fixture-v1",
            rollouts=30,
            hand_seed_start=23000,
            minimum_rollouts=20,
            minimum_texture_classes=10,
        )

    @staticmethod
    def rows(destination: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in (destination / "examples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    def test_raised_nodes_are_complete_legal_private_and_card_matched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset"
            manifest = self.build(destination)
            report = validate_raised_postflop_dataset(destination)
            self.assertTrue(report["valid"], report["issues"])
            self.assertTrue(manifest["training_eligible"], manifest["eligibility"])
            self.assertEqual(manifest["example_count"], 30 * 216)
            self.assertEqual(report["state_count"], 30 * 18)
            self.assertEqual(report["policy_state_count"], 30 * 54)
            self.assertEqual(manifest["node_families"], list(NODE_FAMILIES))
            self.assertEqual(
                manifest["hero_action_keys_by_node"],
                {
                    node: list(actions)
                    for node, actions in ACTION_KEYS_BY_NODE.items()
                },
            )

            rows = self.rows(destination)
            cards: dict[tuple[str, str, str], dict[str, tuple[str, ...]]] = {}
            observed_targets: dict[str, set[str]] = {
                key: set() for key in ("bet_33", "bet_67", "bet_100", "raise_min", "raise_3x")
            }
            for row in rows:
                state = row["state"]
                self.assertIsNone(state["opponent_cards"])
                self.assertEqual(state["next_actor"], "Hero")
                self.assertTrue(row["provenance"]["raised_preflop_pot"])
                self.assertIn("raise_to:2.5", " ".join(state["action_history"]))
                key = (row["group_id"], row["street"], row["node_family"])
                cards.setdefault(key, {})[row["hero_position"]] = tuple(
                    [*state["hero_cards"], *state["board"]]
                )
                action_key = row["counterfactual_action_key"]
                target = row["counterfactual_action"].get("to_amount_bb")
                if action_key in observed_targets:
                    observed_targets[action_key].add(str(target))
                if row["node_family"] == "lead":
                    self.assertEqual(state["to_call_bb"], "0")
                else:
                    self.assertGreater(Decimal(state["to_call_bb"]), 0)
            for positions in cards.values():
                self.assertEqual(set(positions), set(HERO_POSITIONS))
                self.assertEqual(positions["BTN"], positions["BB"])
            self.assertEqual(observed_targets["bet_33"], {"1.65"})
            self.assertEqual(observed_targets["bet_67"], {"3.35"})
            self.assertEqual(observed_targets["bet_100"], {"5"})
            self.assertTrue(observed_targets["raise_min"].isdisjoint(observed_targets["raise_3x"]))

    def test_build_is_deterministic_and_has_every_paired_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_manifest = self.build(first)
            second_manifest = self.build(second)
            self.assertEqual(
                first_manifest["dataset_fingerprint"],
                second_manifest["dataset_fingerprint"],
            )
            self.assertEqual(
                first_manifest["paired_comparisons"],
                second_manifest["paired_comparisons"],
            )
            for position in HERO_POSITIONS:
                for policy in OPPONENT_POLICIES:
                    for street in STREETS:
                        nodes = first_manifest["paired_comparisons"][position][policy][street]
                        self.assertEqual(set(nodes), set(NODE_FAMILIES))
                        self.assertEqual(
                            set(nodes["lead"]),
                            {"bet_33_minus_check", "bet_67_minus_check", "bet_100_minus_check"},
                        )
                        for node in ("facing_33", "facing_75"):
                            self.assertEqual(
                                set(nodes[node]),
                                {"fold_minus_call", "raise_min_minus_call", "raise_3x_minus_call"},
                            )
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.build(first)

    def test_validator_detects_private_leakage_illegal_action_and_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset"
            self.build(destination)
            rows = self.rows(destination)
            first = rows[0]
            first["state"]["opponent_cards"] = ["As", "Kd"]
            first["counterfactual_action"] = {
                "action": "all_in",
                "to_amount_bb": "100",
            }
            first["counterfactual_action_key"] = "unknown_action"
            first["split"] = (
                "validation" if first["split"] != "validation" else "test"
            )
            material = dict(first)
            material.pop("example_sha256")
            first["example_sha256"] = _sha256_bytes(_canonical(material))
            examples_bytes = b"".join(_canonical(row) + b"\n" for row in rows)
            (destination / "examples.jsonl").write_bytes(examples_bytes)

            manifest_path = destination / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["examples_sha256"] = _sha256_bytes(examples_bytes)
            split_counts = Counter(str(row["split"]) for row in rows)
            manifest["split_counts"] = {
                split: split_counts.get(split, 0)
                for split in ("train", "validation", "test")
            }
            manifest.pop("dataset_fingerprint")
            manifest["dataset_fingerprint"] = _sha256_bytes(_canonical(manifest))
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

            report = validate_raised_postflop_dataset(destination)
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("private" in issue for issue in report["issues"]),
                report["issues"],
            )
            self.assertTrue(
                any("not legal" in issue for issue in report["issues"]),
                report["issues"],
            )
            self.assertTrue(
                any("action" in issue for issue in report["issues"]),
                report["issues"],
            )
            self.assertTrue(
                any("split-leaked" in issue for issue in report["issues"]),
                report["issues"],
            )


if __name__ == "__main__":
    unittest.main()
