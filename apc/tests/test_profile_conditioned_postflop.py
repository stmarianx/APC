from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.full_hand_table import _coach_types
from apc.evaluate_profile_conditioned import (
    build_profile_conditioned_audit,
    validate_profile_conditioned_audit,
)
from apc.self_learning.postflop_policy_rollout_dataset import build_postflop_policy_dataset
from apc.self_learning.profile_conditioned_postflop import (
    evaluate_profile_conditioned_dataset,
    evaluate_profile_conditioned_latency,
    predict_profile_conditioned_postflop,
)
from apc.self_learning.train_postflop_policy_value import train_postflop_policy_value_model

_coach_types()
from poker_coach.opponent_model import infer_opponent_policy_mixture
from poker_coach.profiles import PlayerProfile, Tendency


class ProfileConditionedPostflopTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        dataset = root / "dataset"
        build_postflop_policy_dataset(
            dataset,
            dataset_id="profile-conditioned-postflop-fixture-v1",
            rollouts=30,
            hand_seed_start=8100,
            minimum_rollouts=20,
            minimum_texture_classes=10,
        )
        checkpoint = train_postflop_policy_value_model(dataset, root / "checkpoint.json")
        return dataset, checkpoint

    @staticmethod
    def archetype(folds: int, aggressive: int, showdowns: int) -> dict[str, object]:
        profile = PlayerProfile("training:villain")
        for tendency, successes, trials in (
            (Tendency.FOLD_TO_FLOP_CBET, folds, 50),
            (Tendency.AGGRESSIVE_ACTION, aggressive, 50),
            (Tendency.WENT_TO_SHOWDOWN, showdowns, 50),
        ):
            for success in [True] * successes + [False] * (trials - successes):
                profile.observe(tendency, success)
        return infer_opponent_policy_mixture(profile)

    @classmethod
    def overfolder(cls) -> dict[str, object]:
        return cls.archetype(45, 5, 5)

    @staticmethod
    def candidate_states(dataset: Path) -> list[dict[str, object]]:
        rows = [json.loads(line) for line in (dataset / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
        return [row["state"] for row in rows if row["split"] == "test" and row["counterfactual_action"]["action"] == "check"]

    def test_sparse_profile_remains_observe_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint = self.fixture(Path(directory))
            mixture = infer_opponent_policy_mixture(PlayerProfile("unknown"))
            result = predict_profile_conditioned_postflop(checkpoint, self.candidate_states(dataset)[0], mixture)
            self.assertEqual(result["status"], "profile_evidence_observe_only")
            self.assertIsNone(result["profile_conditioned_action"])
            self.assertFalse(result["recommendation_allowed"])
            self.assertEqual(result["units"], "BB")

    def test_evidenced_profile_produces_only_uncertainty_stable_offline_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint = self.fixture(Path(directory))
            mixture = self.overfolder()
            results = [predict_profile_conditioned_postflop(checkpoint, state, mixture) for state in self.candidate_states(dataset)]
            stable = next(result for result in results if result["status"] == "offline_profile_conditioned_action_stable_not_recommendation")
            self.assertEqual(stable["profile_conditioned_action"]["action"], "bet")
            self.assertTrue(stable["evidence_gate_passed"])
            self.assertTrue(stable["uncertainty_stable_action"])
            self.assertGreaterEqual(float(stable["selected_minus_alternative_uncertainty_lower_bb"]), 0)
            self.assertFalse(stable["activation_authorized"])
            self.assertFalse(stable["external_actuation"])

    def test_tampered_mixture_is_rejected_and_latency_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint = self.fixture(Path(directory))
            state = self.candidate_states(dataset)[0]
            mixture = self.overfolder()
            report = evaluate_profile_conditioned_latency(checkpoint, state, mixture, repetitions=20)
            self.assertTrue(report["deterministic_prediction"])
            self.assertTrue(report["latency_gate"]["passed"], report)
            mixture["opponent_policy_weights"]["check_call"] = "1"
            with self.assertRaisesRegex(ValueError, "mixture is invalid"):
                predict_profile_conditioned_postflop(checkpoint, state, mixture)

    def test_dataset_audit_covers_three_distinct_evidenced_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint = self.fixture(Path(directory))
            report = evaluate_profile_conditioned_dataset(
                dataset,
                checkpoint,
                {
                    "overfolder": self.archetype(45, 5, 5),
                    "sticky_passive": self.archetype(5, 5, 40),
                    "aggressive_selective": self.archetype(20, 45, 20),
                },
            )
            self.assertEqual(report["profile_scenarios"], 3)
            self.assertEqual(report["profile_state_evaluations"], report["test_visible_states"] * 3)
            self.assertTrue(report["all_outputs_non_authorizing"])
            self.assertTrue(report["gate"]["passed"], report)

    def test_frozen_audit_artifact_validates_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = self.fixture(root)
            artifact = root / "audit-artifact"
            manifest = build_profile_conditioned_audit(artifact, dataset, root / "checkpoint.json")
            report = validate_profile_conditioned_audit(artifact)
            self.assertTrue(report["valid"], report["issues"])
            self.assertTrue(manifest["gate_passed"])
            mixtures_path = artifact / "mixtures.json"
            mixtures = json.loads(mixtures_path.read_text(encoding="utf-8"))
            mixtures["overfolder"]["recommendation_allowed"] = True
            mixtures_path.write_text(json.dumps(mixtures), encoding="utf-8")
            self.assertFalse(validate_profile_conditioned_audit(artifact)["valid"])


if __name__ == "__main__":
    unittest.main()
