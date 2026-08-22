from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apc.neural.replay_buffer import APCReplayBuffer


class APCReplayBufferTests(unittest.TestCase):
    @staticmethod
    def replay(index: int) -> dict[str, object]:
        return {
            "schema_version": "1.0.0", "model_name": "APC", "units": "BB",
            "source_environment": "controlled_virtual_chips",
            "session_id": f"session-{index}", "hand_id": f"hand-{index}",
            "split_group_id": f"group-{index}", "source_fingerprint": format(index + 1, "064x"),
            "full_hand_completed": True, "external_actuation": False,
            "events": [
                {"observed_monotonic_ms": 1000, "state_fingerprint": f"state-{index}-1",
                 "legal_action_keys": ["fold", "call", "raise"], "chosen_action_key": "call",
                 "canonical_state": {"units": "BB", "opponent_cards": None, "pot_bb": "5", "to_call_bb": "1.5"}},
                {"observed_monotonic_ms": 2000, "state_fingerprint": f"state-{index}-2",
                 "legal_action_keys": ["check", "bet"], "chosen_action_key": "check",
                 "canonical_state": {"units": "BB", "opponent_cards": None, "pot_bb": "8", "to_call_bb": "0"}},
            ],
            "completed_hand_feedback": {"full_hand_completed": True, "hero_reward_bb": str(index - 5)},
        }

    def test_ingest_is_content_addressed_idempotent_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            buffer = APCReplayBuffer(temporary)
            first = buffer.ingest(self.replay(1))
            repeated = buffer.ingest(self.replay(1))
            self.assertTrue(first["added"])
            self.assertFalse(repeated["added"])
            self.assertEqual(first["replay_fingerprint"], repeated["replay_fingerprint"])
            self.assertTrue(buffer.validate()["valid"])
            object_path = Path(temporary) / "objects" / f"{first['replay_fingerprint']}.json"
            object_path.write_bytes(object_path.read_bytes() + b" ")
            report = buffer.validate()
            self.assertFalse(report["valid"])
            self.assertTrue(any("fingerprint" in issue for issue in report["issues"]))

    def test_split_and_prioritized_sampling_are_deterministic_and_retain_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            buffer = APCReplayBuffer(temporary)
            fingerprints = []
            for index in range(40):
                result = buffer.ingest(self.replay(index))
                if result["split"] == "train":
                    fingerprints.append(result["replay_fingerprint"])
            self.assertGreaterEqual(len(fingerprints), 4)
            incumbent = tuple(fingerprints[:2])
            first = buffer.sample_training_batch(8, seed=17, incumbent_fingerprints=incumbent, minimum_incumbent_fraction=0.25)
            second = buffer.sample_training_batch(8, seed=17, incumbent_fingerprints=incumbent, minimum_incumbent_fraction=0.25)
            self.assertEqual(first, second)
            sampled = {buffer.ingest(row)["replay_fingerprint"] for row in first}
            self.assertTrue(set(incumbent).issubset(sampled))
            self.assertTrue(buffer.validate()["valid"])

    def test_incomplete_or_identity_conflicting_replay_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            buffer = APCReplayBuffer(temporary)
            buffer.ingest(self.replay(3))
            conflict = self.replay(3)
            conflict["completed_hand_feedback"]["hero_reward_bb"] = "99"
            with self.assertRaisesRegex(ValueError, "identity"):
                buffer.ingest(conflict)
            incomplete = self.replay(4)
            incomplete["full_hand_completed"] = False
            with self.assertRaisesRegex(ValueError, "ineligible"):
                buffer.ingest(incomplete)
