from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from poker_coach import (
    PersistentProfileStore,
    infer_opponent_policy_mixture,
    validate_opponent_policy_mixture,
)
from poker_coach.profiles import PlayerProfile, Tendency
from poker_coach.opponent_model import _fingerprint


class OpponentModelTests(unittest.TestCase):
    def test_profile_store_is_duplicate_safe_atomic_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            store = PersistentProfileStore()
            first = store.observe("training:villain", "hand-1:fold", Tendency.FOLD_TO_FLOP_CBET, True, position="BB")
            duplicate = store.observe("training:villain", "hand-1:fold", Tendency.FOLD_TO_FLOP_CBET, False, position="BB")
            self.assertEqual(first["status"], "profile_updated")
            self.assertEqual(duplicate["status"], "duplicate_ignored")
            self.assertEqual(store.revision, 1)
            store.save(path)
            loaded = PersistentProfileStore.load(path)
            posterior = loaded.profile("training:villain").estimate(Tendency.FOLD_TO_FLOP_CBET, position="BB")
            self.assertEqual(str(posterior.effective_observations), "1")
            self.assertEqual(loaded.snapshot()["snapshot_sha256"], store.snapshot()["snapshot_sha256"])

    def test_stale_writer_and_snapshot_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            initial = PersistentProfileStore()
            initial.observe("training:villain", "event-1", Tendency.AGGRESSIVE_ACTION, True)
            initial.save(path)
            first = PersistentProfileStore.load(path)
            stale = PersistentProfileStore.load(path)
            first.observe("training:villain", "event-2", Tendency.AGGRESSIVE_ACTION, False)
            first.save(path)
            stale.observe("training:villain", "event-3", Tendency.AGGRESSIVE_ACTION, True)
            with self.assertRaisesRegex(ValueError, "stale writer"):
                stale.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["revision"] += 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid or corrupted"):
                PersistentProfileStore.load(path)

    @staticmethod
    def profile(folds: int, fold_trials: int, aggressive: int, aggressive_trials: int, showdowns: int, showdown_trials: int) -> PlayerProfile:
        profile = PlayerProfile("training:villain")
        for tendency, successes, trials in (
            (Tendency.FOLD_TO_FLOP_CBET, folds, fold_trials),
            (Tendency.AGGRESSIVE_ACTION, aggressive, aggressive_trials),
            (Tendency.WENT_TO_SHOWDOWN, showdowns, showdown_trials),
        ):
            for success in [True] * successes + [False] * (trials - successes):
                profile.observe(tendency, success)
        return profile

    def test_sparse_profile_is_uniform_observe_only_and_fingerprinted(self) -> None:
        mixture = infer_opponent_policy_mixture(PlayerProfile("unknown"))
        self.assertTrue(validate_opponent_policy_mixture(mixture)["valid"])
        self.assertFalse(mixture["evidence_gate"]["passed"])
        self.assertEqual(set(mixture["opponent_policy_weights"].values()), {"0.333333333333"})
        mixture["recommendation_allowed"] = True
        self.assertFalse(validate_opponent_policy_mixture(mixture)["valid"])

    def test_refingerprinted_unsafe_uncertainty_contract_is_rejected(self) -> None:
        mixture = infer_opponent_policy_mixture(PlayerProfile("unknown"))
        mixture["weight_uncertainty_approximate_95"]["check_call"] = ["0.8", "0.2"]
        material = dict(mixture)
        material.pop("mixture_fingerprint")
        mixture["mixture_fingerprint"] = _fingerprint(material)
        validation = validate_opponent_policy_mixture(mixture)
        self.assertFalse(validation["valid"])
        self.assertIn("mixture uncertainty intervals are invalid", validation["issues"])

    def test_evidenced_archetypes_map_directionally_with_uncertainty(self) -> None:
        overfolder = infer_opponent_policy_mixture(self.profile(45, 50, 5, 50, 5, 50))
        sticky = infer_opponent_policy_mixture(self.profile(5, 50, 5, 50, 40, 50))
        aggressive = infer_opponent_policy_mixture(self.profile(20, 50, 45, 50, 20, 50))
        self.assertEqual(max(overfolder["opponent_policy_weights"], key=overfolder["opponent_policy_weights"].get), "fold_to_pressure")
        self.assertEqual(max(sticky["opponent_policy_weights"], key=sticky["opponent_policy_weights"].get), "check_call")
        self.assertEqual(max(aggressive["opponent_policy_weights"], key=aggressive["opponent_policy_weights"].get), "made_hand_selective")
        for mixture in (overfolder, sticky, aggressive):
            self.assertTrue(validate_opponent_policy_mixture(mixture)["valid"])
            self.assertTrue(mixture["evidence_gate"]["passed"], mixture["evidence_gate"])
            self.assertTrue(all(len(bounds) == 2 for bounds in mixture["weight_uncertainty_approximate_95"].values()))


if __name__ == "__main__":
    unittest.main()
