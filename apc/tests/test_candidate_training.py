from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from apc.self_learning.replay_dataset import build_replay_dataset
from apc.self_learning.evaluate_candidate_smoke import evaluate_candidate_smoke
from apc.self_learning.train_candidate import hashed_features, train_candidate, validate_candidate_checkpoint


ROOT = Path(__file__).resolve().parents[2]
COACH_SRC = ROOT / "coach" / "src"
if str(COACH_SRC) not in sys.path:
    sys.path.insert(0, str(COACH_SRC))

from poker_coach import PokerStarsParser, SolverBundleImporter


class CandidateTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source_hand = PokerStarsParser().parse_file(
            ROOT / "coach" / "examples" / "sample_play_money_hand.txt"
        )[0]
        cls.hands = tuple(
            replace(source_hand, hand_id=f"candidate-fixture-{index:03d}")
            for index in range(60)
        )
        cls.bundle = SolverBundleImporter().parse_file(
            ROOT / "coach" / "examples" / "sample_solver_bundle.json"
        )

    def build_dataset(self, path: Path) -> dict[str, object]:
        return build_replay_dataset(
            path,
            self.hands,
            self.bundle.spots,
            dataset_id="candidate-smoke-v1",
            source_fingerprints={"hands": "c" * 64, "solver": "d" * 64},
            split_ratios=(Decimal("0.60"), Decimal("0.20"), Decimal("0.20")),
            minimum_examples=30,
            minimum_groups=30,
        )

    def test_grouped_dataset_trains_deterministic_unpromoted_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            manifest = self.build_dataset(dataset)
            self.assertTrue(manifest["training_eligible"])
            self.assertTrue(all(manifest["split_counts"][split] > 0 for split in ("train", "validation", "test")))
            first = train_candidate(dataset, Path(directory) / "first.json", epochs=20, feature_dimension=64)
            second = train_candidate(dataset, Path(directory) / "second.json", epochs=20, feature_dimension=64)
            self.assertEqual(first["checkpoint_fingerprint"], second["checkpoint_fingerprint"])
            self.assertFalse(first["activation_authorized"])
            self.assertFalse(first["incumbent_replaced"])
            self.assertEqual(first["status"], "candidate_not_promoted")
            self.assertTrue(validate_candidate_checkpoint(first)["valid"])
            for split in ("train", "validation", "test"):
                self.assertGreater(first["metrics"][split]["examples"], 0)
                self.assertEqual(first["metrics"][split]["top_action_agreement"], "1")

    def test_noneligible_fixture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            source_hand = self.hands[0]
            build_replay_dataset(
                dataset,
                (source_hand,),
                self.bundle.spots,
                dataset_id="too-small",
                source_fingerprints={"hands": "e" * 64, "solver": "f" * 64},
            )
            with self.assertRaisesRegex(ValueError, "not training eligible"):
                train_candidate(dataset, Path(directory) / "candidate.json")

    def test_checkpoint_tamper_and_overwrite_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            self.build_dataset(dataset)
            output = Path(directory) / "candidate.json"
            checkpoint = train_candidate(dataset, output, epochs=2, feature_dimension=32)
            checkpoint["activation_authorized"] = True
            report = validate_candidate_checkpoint(checkpoint)
            self.assertFalse(report["valid"])
            self.assertIn("candidate checkpoint cannot authorize activation or replace the incumbent", report["issues"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                train_candidate(dataset, output, epochs=2, feature_dimension=32)

    def test_end_to_end_smoke_remains_nonpromotional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            report = evaluate_candidate_smoke(
                ROOT / "coach" / "examples" / "sample_solver_bundle.json",
                ROOT / "coach" / "examples" / "sample_play_money_hand.txt",
                output,
                replicas=30,
            )
            self.assertTrue(report["passed"])
            self.assertFalse(report["promotion_eligible"])
            self.assertFalse(report["activation_authorized"])
            self.assertFalse(report["incumbent_replaced"])
            self.assertTrue((output / "candidate.json").is_file())
            self.assertTrue((output / "smoke_report.json").is_file())

    def test_feature_schema_is_suit_renaming_invariant(self) -> None:
        state = {
            "game": "holdem_no_limit",
            "players": 2,
            "hero_position": "BTN",
            "effective_stack_bb": "97",
            "pot_bb": "6.5",
            "board": ["Ah", "7c", "2d"],
            "hero_cards": ["As", "Kd"],
            "action_history": ["BTN raise_to:3", "BB call", "BB check"],
        }
        renamed = dict(state)
        renamed["board"] = ["Ac", "7d", "2s"]
        renamed["hero_cards"] = ["Ah", "Ks"]
        self.assertEqual(hashed_features(state, 64), hashed_features(renamed, 64))


if __name__ == "__main__":
    unittest.main()
