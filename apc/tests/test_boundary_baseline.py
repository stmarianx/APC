from __future__ import annotations

import unittest

from apc.perception.boundary_baseline import _transitions


class BoundaryBaselineTests(unittest.TestCase):
    def test_transition_labels_current_frame_hand_start(self) -> None:
        rows = [
            (None, {"capture_session_id": "s", "sequence_index": 0, "state": {"hand_start": True}}),
            (None, {"capture_session_id": "s", "sequence_index": 1, "state": {"hand_start": False}}),
            (None, {"capture_session_id": "s", "sequence_index": 2, "state": {"hand_start": True}}),
        ]
        transitions = _transitions(rows)
        self.assertEqual([row[-1] for row in transitions], [False, True])

    def test_missing_boundary_label_is_rejected(self) -> None:
        rows = [
            (None, {"capture_session_id": "s", "sequence_index": 0, "state": {"hand_start": True}}),
            (None, {"capture_session_id": "s", "sequence_index": 1, "state": {}}),
        ]
        with self.assertRaisesRegex(ValueError, "hand_start"):
            _transitions(rows)


if __name__ == "__main__":
    unittest.main()
