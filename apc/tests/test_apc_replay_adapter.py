from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from apc.neural.model import APCArchitecture, APCNetwork
from apc.neural.continual_training import (
    action_margin_retention_loss,
    audit_completed_replay_candidate,
    incumbent_argmax_hinge_loss,
    train_completed_replay_candidate,
)
from apc.neural.evaluate_continual_candidate import paired_hand_bootstrap, validate_fresh_replay_report
from apc.neural.diverse_self_play_replay import (
    build_diverse_virtual_replay_buffer,
    generate_diverse_virtual_replay,
    observed_profile_features,
)
from apc.neural.replay_adapter import effective_stack_scale_bb, encode_completed_hand_replays, load_replay_temporal_corpus
from apc.neural.replay_buffer import APCReplayBuffer
from apc.neural.self_play_replay import build_virtual_replay_buffer, generate_virtual_completed_replay
from apc.neural.train_candidate import encode_rows


class APCReplayAdapterTests(unittest.TestCase):
    @staticmethod
    def replay(index: int = 1) -> dict[str, object]:
        return {
            "schema_version": "1.0.0", "model_name": "APC", "units": "BB",
            "source_environment": "controlled_virtual_chips",
            "session_id": f"session-{index}", "hand_id": f"hand-{index}",
            "split_group_id": f"group-{index}", "source_fingerprint": format(index, "064x"),
            "full_hand_completed": True, "external_actuation": False,
            "events": [
                {"observed_monotonic_ms": 1000, "state_fingerprint": f"state-{index}-1",
                 "legal_action_keys": ["fold", "call", "raise"], "chosen_action_key": "call",
                 "chosen_action": {"action": "call", "amount_bb": "1.5"},
                 "player_profile_features": [0.2, 0.3, 0.4, 0.5, 0.1, 0.0, 0.0, 0.0],
                 "canonical_state": {"units": "BB", "opponent_cards": None, "pot_bb": "5", "to_call_bb": "1.5"}},
                {"observed_monotonic_ms": 2000, "state_fingerprint": f"state-{index}-2",
                 "legal_action_keys": ["check", "bet"], "chosen_action_key": "check",
                 "canonical_state": {"units": "BB", "opponent_cards": None, "pot_bb": "8", "to_call_bb": "0"}},
            ],
            "completed_hand_feedback": {"full_hand_completed": True, "hero_reward_bb": "3.5"},
        }

    @staticmethod
    def strategy_rows() -> list[dict[str, object]]:
        rows = []
        commands = [
            ("fold", {"action": "fold"}, -1.5),
            ("call", {"action": "call", "amount_bb": "1.5"}, 1.0),
            ("raise_min", {"action": "raise", "to_amount_bb": "3"}, 2.0),
            ("raise_3x", {"action": "raise", "to_amount_bb": "4.5"}, 1.5),
        ]
        state = {
            "units": "BB", "opponent_cards": None, "pot_bb": "5", "to_call_bb": "1.5",
            "stacks_bb": {"Hero": "98", "Villain": "97"},
            "street_contributions_bb": {"Hero": "0", "Villain": "1.5"},
            "hero_position": "BB", "street": "flop", "hero_cards": ["As", "Kd"],
            "board": ["Th", "4s", "2c"], "action_history": [],
            "action_buttons": [{"action": "fold"}, {"action": "call"}, {"action": "raise"}],
        }
        for split_index, split in enumerate(("train", "validation", "test")):
            for action_key, command, value in commands:
                rows.append({
                    "split": split, "group_id": f"strategy-hand-{split_index}",
                    "policy_state_id": f"strategy-state-{split_index}", "hero_position": "BB",
                    "node_family": "facing_33", "opponent_policy": "check_call",
                    "counterfactual_action_key": action_key, "counterfactual_action": command,
                    "learning_signal": {"hero_return_bb": str(value)}, "state": state,
                })
        return rows

    def test_completed_hand_becomes_ordered_private_safe_temporal_windows(self) -> None:
        corpus = encode_completed_hand_replays([(self.replay(), "train")], max_events=4)
        self.assertEqual(corpus.state_tokens.shape, (2, 4, 16, 24))
        self.assertEqual(corpus.state_padding_mask.shape, (2, 4, 16))
        self.assertEqual(corpus.manifest["completed_hands"], 1)
        self.assertEqual(corpus.manifest["decisions"], 2)
        self.assertFalse(corpus.manifest["opponent_private_cards_used"])
        self.assertEqual(corpus.manifest["value_loss_scale"], "public_effective_stack_bb_per_decision")
        np.testing.assert_array_equal(corpus.target_scale_bb, np.asarray([5.0, 8.0], dtype=np.float32))
        self.assertTrue(corpus.modality_available[0, 2])
        self.assertFalse(corpus.modality_available[1, 2])
        self.assertAlmostEqual(float(corpus.chosen_size_features[0, 0]), float(np.tanh(1.5 / 25.0)), places=6)
        self.assertTrue(corpus.state_padding_mask[0, 1:].all())
        self.assertTrue((corpus.state_tokens[1, 0, ~corpus.state_padding_mask[1, 0], 3] > 0).all())

    def test_effective_stack_loss_scale_is_public_bb_and_stack_aware(self) -> None:
        shallow = {"stacks_bb": {"Hero": "38", "Villain": "37"}, "street_contributions_bb": {"Hero": "2", "Villain": "3"}}
        deep = {"stacks_bb": {"Hero": "198", "Villain": "197"}, "street_contributions_bb": {"Hero": "2", "Villain": "3"}}
        self.assertEqual(effective_stack_scale_bb(shallow), 40.0)
        self.assertEqual(effective_stack_scale_bb(deep), 200.0)
        with self.assertRaisesRegex(ValueError, "outside the BB domain"):
            effective_stack_scale_bb({"stacks_bb": {"Hero": "-1", "Villain": "40"}})

    def test_buffer_loading_preserves_group_split_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            buffer = APCReplayBuffer(temporary)
            for index in range(1, 12):
                buffer.ingest(self.replay(index))
            first = load_replay_temporal_corpus(buffer, max_events=3)
            second = load_replay_temporal_corpus(temporary, max_events=3)
            self.assertEqual(first.manifest, second.manifest)
            np.testing.assert_array_equal(first.state_tokens, second.state_tokens)
            self.assertEqual(set(first.split), {"train", "validation", "test"})
            by_hand = {}
            for fingerprint, split in zip(first.replay_fingerprints, first.split):
                self.assertEqual(by_hand.setdefault(fingerprint, split), split)

    def test_network_accepts_temporal_states_and_rejects_bad_masks(self) -> None:
        corpus = encode_completed_hand_replays([(self.replay(), "train")], max_events=3)
        model = APCNetwork(APCArchitecture(hidden_dimension=32, transformer_layers=1, attention_heads=4, dropout=0.0, profile_hidden=16, visual_channels=(4, 8, 8))).eval()
        output = model(
            torch.from_numpy(corpus.state_tokens),
            torch.from_numpy(corpus.state_padding_mask),
            torch.from_numpy(corpus.profile_features),
            torch.from_numpy(corpus.modality_available),
            torch.from_numpy(corpus.legal_action_mask),
            candidate_action_index=torch.from_numpy(corpus.chosen_action_index),
            candidate_size_features=torch.from_numpy(corpus.chosen_size_features),
        )
        self.assertEqual(tuple(output["candidate_action_value_bb"].shape), (2,))
        with self.assertRaisesRegex(ValueError, "padding mask"):
            model(
                torch.from_numpy(corpus.state_tokens),
                torch.from_numpy(corpus.state_padding_mask[:, 0]),
                torch.from_numpy(corpus.profile_features),
                torch.from_numpy(corpus.modality_available),
                torch.from_numpy(corpus.legal_action_mask),
            )

    def test_continual_training_is_deterministic_preserves_incumbent_and_tests_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            buffer = APCReplayBuffer(temporary)
            for index in range(1, 18):
                replay = self.replay(index)
                replay["completed_hand_feedback"]["hero_reward_bb"] = str((index % 5) - 2)
                buffer.ingest(replay)
            corpus = load_replay_temporal_corpus(buffer, max_events=3)
            architecture = APCArchitecture(hidden_dimension=32, transformer_layers=1, attention_heads=4, dropout=0.0, profile_hidden=16, visual_channels=(4, 8, 8))
            torch.manual_seed(31)
            incumbent = APCNetwork(architecture)
            incumbent_before = {name: tensor.detach().clone() for name, tensor in incumbent.state_dict().items()}
            first, first_metrics = train_completed_replay_candidate(corpus, incumbent, seed=23, epochs=2, batch_size=8)
            second, second_metrics = train_completed_replay_candidate(corpus, incumbent, seed=23, epochs=2, batch_size=8)
            self.assertEqual(first_metrics, second_metrics)
            for name, tensor in first.state_dict().items():
                torch.testing.assert_close(tensor, second.state_dict()[name], rtol=0, atol=0)
            for name, tensor in incumbent.state_dict().items():
                torch.testing.assert_close(tensor, incumbent_before[name], rtol=0, atol=0)
            report = audit_completed_replay_candidate(corpus, incumbent, first, first_metrics)
            self.assertTrue(report["gates"]["rollback_verified"])
            self.assertFalse(report["gates"]["promotion_authorized"])
            self.assertFalse(report["weights_updated_during_hand"])
            self.assertFalse(report["selection_uses_test"])

    def test_virtual_self_play_builds_private_safe_completed_replay(self) -> None:
        first = generate_virtual_completed_replay(81)
        second = generate_virtual_completed_replay(81)
        self.assertEqual(first, second)
        self.assertTrue(first["full_hand_completed"])
        self.assertFalse(first["external_actuation"])
        self.assertTrue(all(event["canonical_state"]["opponent_cards"] is None for event in first["events"]))
        with tempfile.TemporaryDirectory() as temporary:
            report = build_virtual_replay_buffer(temporary, hands=18, seed_start=100, hands_per_session=1)
            self.assertTrue(report["training_eligible"], report)
            self.assertEqual(report["hands_added"], 18)
            self.assertGreater(report["decisions"], 18)

    def test_strategy_rehearsal_runs_in_incumbent_coordinates(self) -> None:
        rows = [(self.replay(index), split) for index, split in ((1, "train"), (2, "validation"), (3, "test"))]
        corpus = encode_completed_hand_replays(rows, max_events=3)
        strategy = encode_rows(self.strategy_rows())
        architecture = APCArchitecture(hidden_dimension=32, transformer_layers=1, attention_heads=4, dropout=0.0, profile_hidden=16, visual_channels=(4, 8, 8))
        torch.manual_seed(41)
        incumbent = APCNetwork(architecture)
        original_mean = float(incumbent.value_mean_bb)
        original_scale = float(incumbent.value_scale_bb)
        candidate, metrics = train_completed_replay_candidate(
            corpus, incumbent, seed=29, epochs=1, batch_size=2,
            strategy_rehearsal=strategy, strategy_rehearsal_weight=0.5,
            strategy_rehearsal_batch_size=4,
        )
        self.assertIsNotNone(metrics["history"][0]["strategy_rehearsal_loss"])
        self.assertEqual(float(candidate.value_mean_bb), original_mean)
        self.assertEqual(float(candidate.value_scale_bb), original_scale)

    def test_action_margin_retention_covers_all_four_action_orderings(self) -> None:
        incumbent = torch.tensor([0.0, 1.0, 3.0, 2.0, -1.0, 4.0, 2.0, 0.0])
        self.assertEqual(float(action_margin_retention_loss(incumbent.clone(), incumbent, 5.0)), 0.0)
        reordered = incumbent.clone()
        reordered[1], reordered[2] = incumbent[2], incumbent[1]
        self.assertGreater(float(action_margin_retention_loss(reordered, incumbent, 5.0)), 0.0)
        with self.assertRaisesRegex(ValueError, "complete four-action groups"):
            action_margin_retention_loss(torch.ones(3), torch.ones(3), 5.0)
        with self.assertRaisesRegex(ValueError, "scale must be positive"):
            action_margin_retention_loss(torch.ones(4), torch.ones(4), 0.0)

    def test_incumbent_argmax_hinge_directly_penalizes_selection_flips(self) -> None:
        incumbent = torch.tensor([0.0, 4.0, 1.0, 2.0, 3.0, -1.0, 1.0, 0.0])
        self.assertEqual(float(incumbent_argmax_hinge_loss(incumbent.clone(), incumbent)), 0.0)
        compressed = torch.tensor([0.0, 1.5, 1.0, 2.0, 3.0, -1.0, 1.0, 0.0])
        self.assertGreater(float(incumbent_argmax_hinge_loss(compressed, incumbent)), 0.0)
        restored = compressed.clone()
        restored[1] = 4.0
        self.assertEqual(float(incumbent_argmax_hinge_loss(restored, incumbent)), 0.0)
        with self.assertRaisesRegex(ValueError, "complete four-action groups"):
            incumbent_argmax_hinge_loss(torch.ones(5), torch.ones(5))

    def test_paired_replay_bootstrap_is_complete_hand_grouped_and_deterministic(self) -> None:
        corpus = encode_completed_hand_replays(
            [(self.replay(index), "test") for index in range(1, 5)], max_events=3
        )
        indices = corpus.indices("test")
        actual = corpus.target_return_bb[indices]
        incumbent_policy = np.zeros((len(indices), 6), dtype=np.float32)
        candidate_policy = np.zeros((len(indices), 6), dtype=np.float32)
        incumbent_policy[:, 0] = 1
        candidate_policy[np.arange(len(indices)), corpus.chosen_action_index[indices]] = 1
        incumbent = {"value": actual + 2, "policy": incumbent_policy, "temporal": np.zeros(len(indices))}
        candidate = {"value": actual, "policy": candidate_policy, "temporal": np.ones(len(indices))}
        first = paired_hand_bootstrap(corpus, indices, incumbent, candidate, samples=200, seed=17)
        second = paired_hand_bootstrap(corpus, indices, incumbent, candidate, samples=200, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["complete_hands"], 4)
        self.assertEqual(first["resampling_unit"], "complete_hand")
        self.assertGreater(float(first["mae_improvement_bb"]["lower_95"]), 0)
        self.assertGreaterEqual(float(first["action_accuracy_improvement"]["lower_95"]), 0)

    def test_fresh_replay_report_validator_recomputes_evidence_gates(self) -> None:
        metric = {"mae_bb": "1", "rmse_bb": "1.2", "bias_bb": "0", "observed_action_accuracy": "0.5", "temporal_consistency_mean": "0.7"}
        report = {
            "schema_version": "1.0.0", "model_name": "APC",
            "audit_kind": "fresh_completed_replay_paired_incumbent", "status": "evaluated_not_promoted",
            "audit_replay": {"used_for_training_or_selection": False, "evaluated_split": "test"},
            "incumbent_checkpoint_fingerprint": "a" * 64,
            "candidate_checkpoint_fingerprint": "b" * 64,
            "incumbent": metric, "candidate": metric,
            "paired_bootstrap": {
                "complete_hands": 20,
                "mae_improvement_bb": {"lower_95": "0.1"},
                "rmse_improvement_bb": {"lower_95": "-0.1"},
                "action_accuracy_improvement": {"lower_95": "0"},
            },
            "latency": {"p95_ms": "8", "threshold_p95_ms": "50", "passed": True},
            "gates": {
                "minimum_20_complete_test_hands": True,
                "mae_improvement_lower_95_above_zero": True,
                "rmse_improvement_lower_95_above_zero": False,
                "action_accuracy_lower_95_nonnegative": True,
                "calibration_passed": False, "promotion_authorized": False,
            },
            "recommendation_allowed": False, "activation_authorized": False,
        }
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        report["report_fingerprint"] = hashlib.sha256(canonical(report)).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(validate_fresh_replay_report(path)["valid"])
            report["gates"]["rmse_improvement_lower_95_above_zero"] = True
            report.pop("report_fingerprint")
            report["report_fingerprint"] = hashlib.sha256(canonical(report)).hexdigest()
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertFalse(validate_fresh_replay_report(path)["valid"])

    def test_observed_profile_is_bounded_and_uncertainty_decreases_with_evidence(self) -> None:
        empty = observed_profile_features([])
        evidenced = observed_profile_features(["check", "call", "raise", "fold"])
        self.assertEqual(len(empty), 8)
        self.assertTrue(all(0 <= value <= 1 for value in empty + evidenced))
        self.assertGreater(empty[6], evidenced[6])
        self.assertGreater(evidenced[5], empty[5])

    def test_diverse_replay_holds_opponent_policy_out_of_training_profiles(self) -> None:
        first = generate_diverse_virtual_replay(
            501, session_id="diverse-fixture", hero_policy="pressure",
            opponent_policy="selective", starting_stack_bb="40",
        )
        second = generate_diverse_virtual_replay(
            501, session_id="diverse-fixture", hero_policy="pressure",
            opponent_policy="selective", starting_stack_bb="40",
        )
        self.assertEqual(first, second)
        self.assertTrue(all(len(event["player_profile_features"]) == 8 for event in first["events"]))
        with tempfile.TemporaryDirectory() as temporary:
            report = build_diverse_virtual_replay_buffer(
                temporary, hands=192, seed_start=5000, hands_per_session=3,
                stack_depths_bb=("40",),
            )
            self.assertTrue(report["training_eligible"], report)
            self.assertEqual(report["profile_conditioned_decisions"], report["decisions"])
            self.assertTrue(all("opponent=selective" not in key for key in report["split_configuration_counts"]["train"]))
            for split in ("validation", "test"):
                self.assertTrue(all("opponent=selective" in key for key in report["split_configuration_counts"][split]))
            self.assertEqual(report["covered_training_configurations"], report["required_training_configurations"])
            self.assertEqual(report["covered_held_out_configurations"], report["required_held_out_configurations"])


if __name__ == "__main__":
    unittest.main()
