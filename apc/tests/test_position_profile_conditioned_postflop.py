from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from apc.evaluate_position_profile_conditioned import (
    build_position_profile_conditioned_audit,
    validate_position_profile_conditioned_audit,
)
from apc.full_hand_table import _coach_types
from apc.self_learning.position_profile_conditioned_postflop import (
    evaluate_position_profile_conditioned_dataset,
    evaluate_position_profile_conditioned_latency,
    predict_position_profile_conditioned_postflop,
)
from apc.self_learning.postflop_position_rollout_dataset import (
    build_postflop_position_dataset,
)
from apc.self_learning.train_position_postflop_value import (
    train_position_postflop_value_model,
)
from apc.self_learning.train_value import _sha256

_coach_types()
from poker_coach.opponent_model import infer_opponent_policy_mixture
from poker_coach.profiles import PlayerProfile, Tendency


class PositionProfileConditionedPostflopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.dataset = cls.root / "dataset"
        build_postflop_position_dataset(
            cls.dataset,
            dataset_id="position-profile-conditioned-fixture-v1",
            rollouts=180,
            hand_seed_start=19100,
            minimum_rollouts=100,
            minimum_texture_classes=10,
        )
        cls.checkpoint_path = cls.root / "checkpoint.json"
        cls.checkpoint = train_position_postflop_value_model(
            cls.dataset, cls.checkpoint_path
        )
        if cls.checkpoint["generalization_gate"]["passed"] is not True:
            cls.checkpoint["generalization_gate"]["passed"] = True
            cls.checkpoint.pop("checkpoint_fingerprint")
            cls.checkpoint["checkpoint_fingerprint"] = _sha256(cls.checkpoint)
            cls.checkpoint_path.write_text(
                json.dumps(cls.checkpoint, indent=2) + "\n", encoding="utf-8"
            )
        rows = [
            json.loads(line)
            for line in (cls.dataset / "examples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        cls.states = list(
            {
                row["state_id"]: row["state"]
                for row in rows
                if row["split"] == "test"
                and row["counterfactual_action"]["action"] == "check"
            }.values()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @staticmethod
    def archetype(folds: int, aggressive: int, showdowns: int) -> dict[str, object]:
        profile = PlayerProfile("training:position-villain")
        for tendency, successes in (
            (Tendency.FOLD_TO_FLOP_CBET, folds),
            (Tendency.AGGRESSIVE_ACTION, aggressive),
            (Tendency.WENT_TO_SHOWDOWN, showdowns),
        ):
            for success in [True] * successes + [False] * (50 - successes):
                profile.observe(tendency, success)
        return infer_opponent_policy_mixture(profile)

    @classmethod
    def mixtures(cls) -> dict[str, dict[str, object]]:
        return {
            "overfolder": cls.archetype(45, 5, 5),
            "sticky_passive": cls.archetype(5, 5, 40),
            "aggressive_selective": cls.archetype(20, 45, 20),
        }

    def test_sparse_profile_is_observe_only_in_both_positions(self) -> None:
        mixture = infer_opponent_policy_mixture(PlayerProfile("unknown"))
        for position in ("BTN", "BB"):
            state = next(row for row in self.states if row["hero_position"] == position)
            result = predict_position_profile_conditioned_postflop(
                self.checkpoint, state, mixture
            )
            self.assertEqual(result["status"], "profile_evidence_observe_only")
            self.assertEqual(result["hero_position"], position)
            self.assertIsNone(result["profile_conditioned_action"])
            self.assertFalse(result["recommendation_allowed"])
            self.assertEqual(result["units"], "BB")

    def test_evidenced_profile_exposes_only_robust_offline_actions(self) -> None:
        mixture = self.mixtures()["overfolder"]
        results = [
            predict_position_profile_conditioned_postflop(
                self.checkpoint, state, mixture
            )
            for state in self.states
        ]
        stable = [
            row
            for row in results
            if row["status"]
            == "offline_position_profile_action_stable_not_recommendation"
        ]
        self.assertTrue(stable)
        self.assertTrue({row["hero_position"] for row in stable}.issubset({"BTN", "BB"}))
        for row in stable:
            self.assertTrue(row["position_value_generalization_gate_passed"])
            self.assertTrue(row["evidence_gate_passed"])
            self.assertTrue(row["uncertainty_stable_action"])
            self.assertGreaterEqual(
                float(row["selected_minus_alternative_uncertainty_lower_bb"]), 0
            )
            self.assertFalse(row["activation_authorized"])
            self.assertFalse(row["external_actuation"])

    def test_invalid_inputs_and_closed_value_gate_abstain(self) -> None:
        mixture = self.mixtures()["overfolder"]
        invalid_state = copy.deepcopy(self.states[0])
        invalid_state["hero_position"] = "UTG"
        rejected = predict_position_profile_conditioned_postflop(
            self.checkpoint, invalid_state, mixture
        )
        self.assertEqual(rejected["status"], "abstain_unsupported_or_invalid_state")
        self.assertIn("hero_position_not_supported", rejected["reasons"])

        closed = copy.deepcopy(self.checkpoint)
        closed["generalization_gate"]["passed"] = False
        closed.pop("checkpoint_fingerprint")
        closed["checkpoint_fingerprint"] = _sha256(closed)
        rejected = predict_position_profile_conditioned_postflop(
            closed, self.states[0], mixture
        )
        self.assertEqual(
            rejected["status"], "abstain_position_value_generalization_gate"
        )
        self.assertIsNone(rejected["profile_conditioned_action"])

        tampered = copy.deepcopy(mixture)
        tampered["opponent_policy_weights"]["check_call"] = "1"
        with self.assertRaisesRegex(ValueError, "mixture is invalid"):
            predict_position_profile_conditioned_postflop(
                self.checkpoint, self.states[0], tampered
            )

    def test_position_sliced_audit_latency_and_artifact_integrity(self) -> None:
        mixtures = self.mixtures()
        report = evaluate_position_profile_conditioned_dataset(
            self.dataset, self.checkpoint, mixtures
        )
        self.assertEqual(set(report["test_visible_states_by_position"]), {"BTN", "BB"})
        self.assertEqual(report["profile_state_evaluations"], len(self.states) * 3)
        self.assertTrue(report["all_outputs_non_authorizing"])
        for profile in mixtures:
            self.assertEqual(
                set(report["stable_action_coverage_by_position"][profile]),
                {"BTN", "BB"},
            )
        self.assertTrue(report["gate"]["passed"], report)

        latency = evaluate_position_profile_conditioned_latency(
            self.checkpoint, self.states[0], mixtures["overfolder"], repetitions=20
        )
        self.assertTrue(latency["deterministic_prediction"])
        self.assertTrue(latency["latency_gate"]["passed"], latency)

        artifact = self.root / "position-profile-audit"
        manifest = build_position_profile_conditioned_audit(
            artifact, self.dataset, self.checkpoint_path
        )
        validation = validate_position_profile_conditioned_audit(artifact)
        self.assertTrue(manifest["gate_passed"])
        self.assertTrue(validation["valid"], validation["issues"])
        mixtures_path = artifact / "mixtures.json"
        payload = json.loads(mixtures_path.read_text(encoding="utf-8"))
        payload["overfolder"]["recommendation_allowed"] = True
        mixtures_path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertFalse(
            validate_position_profile_conditioned_audit(artifact)["valid"]
        )


if __name__ == "__main__":
    unittest.main()
