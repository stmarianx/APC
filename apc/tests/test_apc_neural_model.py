from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from apc.neural.features import encode_state
from apc.neural.contract import load_apc_neural_config
from apc.neural.model import APCArchitecture, APCNetwork, load_apc_weights, save_apc_weights
from apc.neural.train_candidate import _fingerprint, encode_rows, train_apc_candidate, validate_checkpoint


class APCNeuralModelTests(unittest.TestCase):
    @staticmethod
    def state(cards: list[str] | None = None) -> dict[str, object]:
        return {
            "units": "BB",
            "opponent_cards": None,
            "pot_bb": "6.5",
            "to_call_bb": "1.5",
            "stacks_bb": {"Hero": "97.5", "Villain": "96"},
            "street_contributions_bb": {"Hero": "0", "Villain": "1.5"},
            "hero_position": "BB",
            "street": "flop",
            "hero_cards": (cards or ["As", "Kd"])[:2],
            "board": (cards or ["As", "Kd", "Th", "4s", "2c"])[2:],
            "action_history": ["BTN raise_to:2.5", "BB call", "BB check", "BTN bet:1.5"],
            "action_buttons": [
                {"action": "fold"}, {"action": "call", "amount_bb": "1.5"},
                {"action": "raise", "minimum_to_bb": "3"}, {"action": "all_in", "to_amount_bb": "97.5"},
            ],
        }

    @classmethod
    def rows(cls) -> list[dict[str, object]]:
        rows = []
        actions = [
            ("fold", {"action": "fold"}, -1.5),
            ("call", {"action": "call", "amount_bb": "1.5"}, 2.0),
            ("raise_min", {"action": "raise", "to_amount_bb": "3"}, 3.0),
            ("raise_3x", {"action": "raise", "to_amount_bb": "4.5"}, 2.5),
        ]
        for split_index, split in enumerate(("train", "validation", "test")):
            for action, command, value in actions:
                rows.append({
                    "split": split,
                    "group_id": f"hand-{split_index}",
                    "policy_state_id": f"state-{split_index}",
                    "hero_position": "BB",
                    "node_family": "facing_33",
                    "opponent_policy": "check_call",
                    "counterfactual_action_key": action,
                    "counterfactual_action": command,
                    "learning_signal": {"hero_return_bb": str(value + split_index * 0.1)},
                    "state": cls.state(),
                })
        return rows

    def test_state_features_are_suit_renaming_invariant_and_private(self) -> None:
        first, mask = encode_state(self.state(["As", "Kd", "Th", "4s", "2c"]))
        second, second_mask = encode_state(self.state(["Ah", "Kc", "Ts", "4h", "2d"]))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(mask, second_mask)
        leaked = self.state()
        leaked["opponent_cards"] = ["Qc", "Qd"]
        with self.assertRaisesRegex(ValueError, "opponent private"):
            encode_state(leaked)

    def test_legal_mask_and_pickle_free_weights_round_trip(self) -> None:
        architecture = APCArchitecture(hidden_dimension=32, transformer_layers=1, attention_heads=4, dropout=0.0, profile_hidden=16, visual_channels=(4, 8, 8))
        torch.manual_seed(7)
        model = APCNetwork(architecture).eval()
        corpus = encode_rows(self.rows())
        tokens = torch.from_numpy(corpus.tokens[:1])
        output = model(
            tokens, torch.from_numpy(corpus.padding[:1]), torch.zeros(1, 8),
            torch.from_numpy(corpus.modality_available[:1]), torch.from_numpy(corpus.legal[:1]),
            candidate_action_index=torch.from_numpy(corpus.action[:1]),
            candidate_size_features=torch.from_numpy(corpus.sizes[:1]),
        )
        self.assertLess(float(output["policy_logits"][0, 3].detach()), -1e8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.apc"
            report = save_apc_weights(model, path)
            self.assertTrue(path.read_bytes().startswith(b"APCNEURAL1\n"))
            loaded = load_apc_weights(path, report["weights_sha256"]).eval()
            repeated = loaded(
                tokens, torch.from_numpy(corpus.padding[:1]), torch.zeros(1, 8),
                torch.from_numpy(corpus.modality_available[:1]), torch.from_numpy(corpus.legal[:1]),
                candidate_action_index=torch.from_numpy(corpus.action[:1]),
                candidate_size_features=torch.from_numpy(corpus.sizes[:1]),
            )
            torch.testing.assert_close(output["candidate_action_value_bb"], repeated["candidate_action_value_bb"])
            visual_output = loaded(
                tokens, torch.from_numpy(corpus.padding[:1]), torch.zeros(1, 8),
                torch.tensor([[True, True, False]]), torch.from_numpy(corpus.legal[:1]),
                visual_frames=torch.zeros(1, 2, 3, 32, 48),
                visual_frame_padding_mask=torch.tensor([[False, True]]),
            )
            self.assertEqual(tuple(visual_output["policy_logits"].shape), (1, 6))
            self.assertGreater(float(visual_output["modality_weights"][0, 0].detach()), 0.0)
            tampered = bytearray(path.read_bytes())
            tampered[-1] ^= 1
            path.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_apc_weights(path, report["weights_sha256"])

    def test_tiny_group_exclusive_training_is_deterministic(self) -> None:
        corpus = encode_rows(self.rows())
        architecture = APCArchitecture(hidden_dimension=32, transformer_layers=1, attention_heads=4, dropout=0.0, profile_hidden=16, visual_channels=(4, 8, 8))
        first, first_metrics = train_apc_candidate(corpus, architecture=architecture, seed=19, epochs=2, batch_size=4, learning_rate=1e-3)
        second, second_metrics = train_apc_candidate(corpus, architecture=architecture, seed=19, epochs=2, batch_size=4, learning_rate=1e-3)
        self.assertEqual(first_metrics, second_metrics)
        for name, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, second.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(first_metrics["test"]["examples"], 4)
        self.assertEqual(first_metrics["test"]["policy_states"], 1)

    def test_checkpoint_validator_recomputes_semantic_gates(self) -> None:
        architecture = APCArchitecture(hidden_dimension=32, transformer_layers=1, attention_heads=4, dropout=0.0, profile_hidden=16, visual_channels=(4, 8, 8))
        model = APCNetwork(architecture)
        metric = {
            "examples": 4, "policy_states": 1, "mae_bb": "1", "rmse_bb": "1.2",
            "bias_bb": "0", "train_global_mean_baseline_mae_bb": "2",
            "decision_accuracy": "0.5", "chosen_action_regret_bb": "0.2", "uncertainty_eace": "0.1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = save_apc_weights(model, root / "weights.apc")
            architecture_payload = asdict(architecture)
            architecture_payload["visual_channels"] = list(architecture_payload["visual_channels"])
            checkpoint = {
                "schema_version": "1.0.0", "model_name": "APC",
                "model_family": "multimodal_temporal_neural_network",
                "framework": {"name": "pytorch", "version": torch.__version__, "device": "cpu"},
                "architecture_contract_fingerprint": load_apc_neural_config()["config_fingerprint"],
                "architecture": architecture_payload,
                "training": {"pipeline_version": "3.0.0", "group_exclusive_split": True,
                    "counterfactual_groups_kept_complete_per_batch": True,
                    "validation_selection": "minimum_chosen_action_regret_then_mae_bb",
                    "trained_modalities": ["canonical_state_sequence", "player_profile"],
                    "untrained_modalities": ["visible_frame_sequence"]},
                "weights": {"file": "weights.apc", **weights},
                "metrics": {"train": metric, "validation": metric, "test": metric},
                "latency": {"p95_ms": "6", "threshold_p95_ms": "50", "passed": True},
                "gates": {"finite_outputs": True, "test_value_improves_global_mean": True,
                    "strategy_p95_under_50_ms": True, "visible_table_training_ready": False,
                    "evaluated_coaching_ready": False},
                "status": "offline_neural_candidate_unpromoted",
                "recommendation_allowed": False, "activation_authorized": False,
            }
            checkpoint["checkpoint_fingerprint"] = _fingerprint(checkpoint)
            path = root / "checkpoint.json"
            path.write_text(json.dumps(checkpoint), encoding="utf-8")
            self.assertTrue(validate_checkpoint(path)["valid"])
            checkpoint["gates"]["test_value_improves_global_mean"] = False
            checkpoint.pop("checkpoint_fingerprint")
            checkpoint["checkpoint_fingerprint"] = _fingerprint(checkpoint)
            path.write_text(json.dumps(checkpoint), encoding="utf-8")
            report = validate_checkpoint(path)
            self.assertFalse(report["valid"])
            self.assertTrue(any("value gate" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
