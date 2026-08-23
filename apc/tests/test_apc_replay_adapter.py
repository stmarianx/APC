from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch

from apc.neural.model import APCArchitecture, APCNetwork
from apc.neural.continual_training import (
    audit_completed_replay_candidate,
    train_completed_replay_candidate,
)
from apc.neural.replay_adapter import encode_completed_hand_replays, load_replay_temporal_corpus
from apc.neural.replay_buffer import APCReplayBuffer
from apc.neural.self_play_replay import build_virtual_replay_buffer, generate_virtual_completed_replay


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

    def test_completed_hand_becomes_ordered_private_safe_temporal_windows(self) -> None:
        corpus = encode_completed_hand_replays([(self.replay(), "train")], max_events=4)
        self.assertEqual(corpus.state_tokens.shape, (2, 4, 16, 24))
        self.assertEqual(corpus.state_padding_mask.shape, (2, 4, 16))
        self.assertEqual(corpus.manifest["completed_hands"], 1)
        self.assertEqual(corpus.manifest["decisions"], 2)
        self.assertFalse(corpus.manifest["opponent_private_cards_used"])
        self.assertTrue(corpus.modality_available[0, 2])
        self.assertFalse(corpus.modality_available[1, 2])
        self.assertAlmostEqual(float(corpus.chosen_size_features[0, 0]), float(np.tanh(1.5 / 25.0)), places=6)
        self.assertTrue(corpus.state_padding_mask[0, 1:].all())
        self.assertTrue((corpus.state_tokens[1, 0, ~corpus.state_padding_mask[1, 0], 3] > 0).all())

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


if __name__ == "__main__":
    unittest.main()
