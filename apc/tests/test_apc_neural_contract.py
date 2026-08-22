from __future__ import annotations

import copy
import unittest

from apc.neural.contract import (
    APC_MODEL_NAME,
    load_apc_neural_config,
    validate_apc_neural_config,
    validate_completed_hand_replay,
)


class APCNeuralContractTests(unittest.TestCase):
    def test_default_config_is_neural_temporal_and_fingerprinted(self) -> None:
        first = load_apc_neural_config()
        second = load_apc_neural_config()
        self.assertEqual(first["model_name"], APC_MODEL_NAME)
        self.assertTrue(first["neural_network"])
        self.assertEqual(first["model_family"], "multimodal_temporal_neural_network")
        self.assertEqual(first["config_fingerprint"], second["config_fingerprint"])
        self.assertEqual(first["framework"]["preferred"], "pytorch")
        self.assertTrue(first["framework"]["dependency_install_required"])
        self.assertIn("legal_action_policy_logits", first["architecture"]["output_heads"])
        self.assertIn("action_value_bb", first["architecture"]["output_heads"])

    def test_config_rejects_mid_hand_weight_updates_and_non_neural_aliases(self) -> None:
        config = load_apc_neural_config()
        config.pop("config_fingerprint")
        config["live_learning"]["policy_weight_updates_during_hand"] = True
        report = validate_apc_neural_config(config)
        self.assertFalse(report["valid"])
        self.assertTrue(any("live-learning" in issue for issue in report["issues"]))

        config = load_apc_neural_config()
        config.pop("config_fingerprint")
        config["model_name"] = "APC-tabular"
        config["neural_network"] = False
        self.assertFalse(validate_apc_neural_config(config)["valid"])

    @staticmethod
    def replay() -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "model_name": "APC",
            "units": "BB",
            "source_environment": "controlled_virtual_chips",
            "session_id": "training-session-1",
            "hand_id": "hand-1",
            "split_group_id": "training-session-1",
            "source_fingerprint": "a" * 64,
            "full_hand_completed": True,
            "events": [
                {
                    "observed_monotonic_ms": 1000,
                    "state_fingerprint": "state-1",
                    "legal_action_keys": ["fold", "call", "raise"],
                    "chosen_action_key": "call",
                    "canonical_state": {
                        "units": "BB",
                        "opponent_cards": None,
                        "pot_bb": "5",
                        "to_call_bb": "1.5"
                    }
                },
                {
                    "observed_monotonic_ms": 2000,
                    "state_fingerprint": "state-2",
                    "legal_action_keys": ["check", "bet"],
                    "chosen_action_key": "check",
                    "canonical_state": {
                        "units": "BB",
                        "opponent_cards": None,
                        "pot_bb": "8",
                        "to_call_bb": "0"
                    }
                }
            ],
            "completed_hand_feedback": {
                "full_hand_completed": True,
                "hero_reward_bb": "3.5"
            },
            "external_actuation": False
        }

    def test_completed_live_replay_is_eligible_and_leakage_is_rejected(self) -> None:
        first = validate_completed_hand_replay(self.replay())
        second = validate_completed_hand_replay(self.replay())
        self.assertTrue(first["valid"], first["issues"])
        self.assertTrue(first["eligible_for_candidate_training"])
        self.assertEqual(first["event_count"], 2)
        self.assertEqual(first["replay_fingerprint"], second["replay_fingerprint"])

        leaked = copy.deepcopy(self.replay())
        leaked["events"][0]["canonical_state"]["opponent_cards"] = ["As", "Kd"]
        report = validate_completed_hand_replay(leaked)
        self.assertFalse(report["valid"])
        self.assertFalse(report["eligible_for_candidate_training"])
        self.assertTrue(any("leaks" in issue for issue in report["issues"]))

        incomplete = copy.deepcopy(self.replay())
        incomplete["full_hand_completed"] = False
        self.assertFalse(validate_completed_hand_replay(incomplete)["valid"])


if __name__ == "__main__":
    unittest.main()
