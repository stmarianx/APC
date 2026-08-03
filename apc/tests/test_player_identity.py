from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apc.player_identity import NameObservation, PlayerIdentityRegistry, normalize_player_name


def observation(seat: int, name: str, frame: int, confidence: float = 0.95) -> NameObservation:
    return NameObservation(seat, name, confidence, f"{frame:064x}", frame * 100)


class PlayerIdentityRegistryTests(unittest.TestCase):
    def test_repeated_high_confidence_name_resolves_stable_profile_key(self) -> None:
        registry = PlayerIdentityRegistry("training-table")
        result = None
        for frame in range(1, 4):
            result = registry.observe_batch("table-a", [observation(2, "Villain_42", frame)])[0]
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["profile_key"], "training-table:villain_42")
        self.assertEqual(len(result["identity_id"]), 24)

    def test_conflicting_ocr_candidate_remains_uncertain(self) -> None:
        registry = PlayerIdentityRegistry("training-table")
        registry.observe_batch("table-a", [observation(2, "PlayerOne", 1)])
        registry.observe_batch("table-a", [observation(2, "Player0ne", 2)])
        result = registry.observe_batch("table-a", [observation(2, "PlayerOne", 3)])[0]
        self.assertEqual(result["status"], "developing")
        self.assertEqual(len(result["candidates"]), 2)

    def test_same_name_in_two_seats_is_ambiguous(self) -> None:
        registry = PlayerIdentityRegistry("training-table", minimum_frames=1)
        results = registry.observe_batch(
            "table-a",
            [observation(2, "Clone", 1), observation(3, "Clone", 1)],
        )
        self.assertTrue(all(result["status"] == "ambiguous_collision" for result in results))

    def test_duplicate_frame_does_not_inflate_evidence(self) -> None:
        registry = PlayerIdentityRegistry("training-table")
        first = observation(2, "Villain", 1)
        registry.observe_batch("table-a", [first])
        duplicate = registry.observe_batch("table-a", [first])[0]
        self.assertTrue(duplicate["duplicate_frame"])
        self.assertEqual(duplicate["frames"], 1)

    def test_snapshot_round_trip_preserves_fingerprint_and_resolution(self) -> None:
        registry = PlayerIdentityRegistry("training-table")
        for frame in range(1, 4):
            registry.observe_batch("table-a", [observation(2, "Villain", frame)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identities.json"
            registry.save(path)
            loaded = PlayerIdentityRegistry.load(path)
            self.assertEqual(loaded.snapshot(), registry.snapshot())

    def test_name_normalization_is_unicode_and_case_stable(self) -> None:
        self.assertEqual(normalize_player_name("  PlayerＡ  "), "playera")


if __name__ == "__main__":
    unittest.main()
