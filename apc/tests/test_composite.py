from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apc.perception.composite import (
    MISSING_CRITICAL_FIELDS,
    _card_token,
    _safe_perception_head,
    _visible_card_integrity,
    _visual_signatures_or_abstain,
)


class CompositePerceptionTests(unittest.TestCase):
    def test_card_token_uses_canonical_rank_suit_order(self) -> None:
        self.assertEqual(_card_token({"rank": "A", "suit": "h"}), "Ah")

    def test_incomplete_pipeline_must_keep_recommendation_inputs_missing(self) -> None:
        self.assertIn("effective_stack_bb", MISSING_CRITICAL_FIELDS)
        self.assertIn("action_history", MISSING_CRITICAL_FIELDS)
        self.assertNotIn("seat_stacks_bb", MISSING_CRITICAL_FIELDS)

    def test_optional_visual_identity_failure_abstains_without_raising(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "blank.png"
            Image.new("RGB", (200, 100), "black").save(image)
            rows, error = _visual_signatures_or_abstain(
                image,
                [
                    {
                        "seat_no": 1,
                        "seat_box": {"x": 0.1, "y": 0.1, "width": 0.4, "height": 0.6},
                    }
                ],
            )
        self.assertEqual(rows, [])
        self.assertIn("insufficient foreground", error)

    def test_duplicate_or_street_inconsistent_cards_are_rejected(self) -> None:
        duplicate = _visible_card_integrity(
            ["Ah", "Ah"],
            ["2s", "8d", "5s"],
            "flop",
        )
        self.assertEqual(duplicate["status"], "rejected")
        self.assertFalse(duplicate["checks"]["cards_unique"])
        wrong_street = _visible_card_integrity(["Ah", "Qh"], ["2s"], "flop")
        self.assertFalse(wrong_street["checks"]["board_card_count"])

    def test_valid_four_color_flop_cards_pass_integrity(self) -> None:
        audit = _visible_card_integrity(
            ["Ah", "Qh"],
            ["2s", "8d", "5s"],
            "flop",
        )
        self.assertEqual(audit["status"], "accepted")

    def test_independent_head_failure_returns_auditable_error(self) -> None:
        def broken() -> object:
            raise ValueError("invalid numeric token")

        value, error = _safe_perception_head("table_state_perception", broken)
        self.assertIsNone(value)
        self.assertEqual(error["field"], "table_state_perception")
        self.assertEqual(error["reason"], "perception_head_failed")
        self.assertIn("invalid numeric token", error["detail"])


if __name__ == "__main__":
    unittest.main()
