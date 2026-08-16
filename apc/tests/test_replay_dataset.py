from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from apc.self_learning.replay_dataset import build_replay_dataset, validate_replay_dataset


ROOT = Path(__file__).resolve().parents[2]
COACH_SRC = ROOT / "coach" / "src"
if str(COACH_SRC) not in sys.path:
    sys.path.insert(0, str(COACH_SRC))

from poker_coach import PokerStarsParser, SolverBundleImporter


class ReplayDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hands = PokerStarsParser().parse_file(
            ROOT / "coach" / "examples" / "sample_play_money_hand.txt"
        )
        cls.bundle = SolverBundleImporter().parse_file(
            ROOT / "coach" / "examples" / "sample_solver_bundle.json"
        )
        cls.sources = {
            "hands": "a" * 64,
            "solver": "b" * 64,
        }

    def build(self, destination: Path) -> dict[str, object]:
        return build_replay_dataset(
            destination,
            self.hands,
            self.bundle.spots,
            dataset_id="fixture-replay-v1",
            source_fingerprints=self.sources,
        )

    def test_build_is_immutable_grouped_and_bb_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "replay"
            manifest = self.build(destination)
            self.assertFalse(manifest["training_eligible"])
            self.assertEqual(manifest["example_count"], 1)
            self.assertEqual(manifest["build"]["input_hero_decisions"], 3)
            self.assertEqual(manifest["build"]["exclusions"], {"unmatched_solver_state": 2})
            report = validate_replay_dataset(destination)
            self.assertTrue(report["valid"], report["issues"])
            example = json.loads((destination / "examples.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(example["units"], "BB")
            self.assertFalse(example["target"]["gto_verified"])
            self.assertTrue(example["completed_hand_feedback"]["observed_action_covered"])
            self.assertEqual(example["completed_hand_feedback"]["ev_loss_bb"], "0.15")

    def test_repeated_build_is_deterministic_but_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_manifest = self.build(first)
            second_manifest = self.build(second)
            self.assertEqual(first_manifest["dataset_fingerprint"], second_manifest["dataset_fingerprint"])
            self.assertEqual((first / "examples.jsonl").read_bytes(), (second / "examples.jsonl").read_bytes())
            with self.assertRaises(ValueError):
                self.build(first)

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "replay"
            self.build(destination)
            example = json.loads((destination / "examples.jsonl").read_text(encoding="utf-8"))
            example["state"]["pot_bb"] = "999"
            (destination / "examples.jsonl").write_text(json.dumps(example) + "\n", encoding="utf-8")
            report = validate_replay_dataset(destination)
            self.assertFalse(report["valid"])
            self.assertIn("examples file fingerprint mismatch", report["issues"])
            self.assertTrue(any("fingerprint mismatch" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
