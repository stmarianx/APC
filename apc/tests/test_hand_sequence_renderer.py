from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apc.synthetic.render_hand_sequences import (
    _event_for_street,
    _initial_stacks,
    _session_seed,
    generate_hand_sequence_dataset,
)


class HandSequenceRendererTests(unittest.TestCase):
    def test_session_seed_is_stable_and_isolated(self) -> None:
        self.assertEqual(_session_seed(123, 4), _session_seed(123, 4))
        self.assertNotEqual(_session_seed(123, 4), _session_seed(123, 5))

    def test_interrupted_generation_resumes_without_duplicate_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            partial = generate_hand_sequence_dataset(
                root,
                sessions=3,
                hands_per_session=2,
                seed=2026080302,
                session_limit=1,
            )
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["project"]["frames"], 16)
            self.assertIsNone(partial["manifest"])

            completed = generate_hand_sequence_dataset(
                root,
                sessions=3,
                hands_per_session=2,
                seed=2026080302,
                resume=True,
            )
            self.assertTrue(completed["complete"])
            self.assertEqual(completed["project"]["frames"], 48)
            self.assertEqual(completed["skipped_complete_sessions"], 1)
            fingerprint = completed["validation"]["computed_fingerprints"]

            idempotent = generate_hand_sequence_dataset(
                root,
                sessions=3,
                hands_per_session=2,
                seed=2026080302,
                resume=True,
            )
            self.assertEqual(idempotent["project"]["frames"], 48)
            self.assertEqual(idempotent["rendered_sessions"], 0)
            self.assertEqual(idempotent["skipped_complete_sessions"], 3)
            self.assertEqual(idempotent["validation"]["computed_fingerprints"], fingerprint)

    def test_all_in_is_reserved_for_river_and_exhausts_visible_stack(self) -> None:
        stacks = _initial_stacks(seats=2, global_hand_index=5)
        for street_index in range(3):
            self.assertNotEqual(
                _event_for_street(
                    global_hand_index=5,
                    street_index=street_index,
                    seats=2,
                    stacks=stacks,
                ).action,
                "all_in",
            )
        river = _event_for_street(
            global_hand_index=5,
            street_index=3,
            seats=2,
            stacks=stacks,
        )
        self.assertEqual(river.action, "all_in")
        self.assertEqual(river.amount_bb, stacks[river.actor_seat])

    def test_early_actions_never_end_the_hand(self) -> None:
        stacks = _initial_stacks(seats=6, global_hand_index=3)
        actions = {
            _event_for_street(
                global_hand_index=3,
                street_index=street_index,
                seats=6,
                stacks=stacks,
            ).action
            for street_index in range(3)
        }
        self.assertFalse(actions & {"fold", "all_in"})


if __name__ == "__main__":
    unittest.main()
