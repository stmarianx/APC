from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from apc.perception.composite import (
    MISSING_CRITICAL_FIELDS,
    _card_token,
    _safe_perception_head,
    _visible_card_integrity,
    _visual_signatures_or_abstain,
    infer_visible_state,
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

    def test_optional_turn_clock_is_exposed_in_canonical_milliseconds(self) -> None:
        from PIL import Image

        base_prediction = {
            "layout_id": {"value": "six-max", "confidence": 0.99},
            "theme_id": {"value": "midnight", "confidence": 0.98},
            "street": {"value": "flop", "confidence": 0.97},
            "legal_actions": {"value": "check+bet", "confidence": 0.96},
        }
        card_prediction = {
            "hero_cards": [
                {"rank": "A", "suit": "h", "confidence": 0.95},
                {"rank": "Q", "suit": "h", "confidence": 0.94},
            ],
            "board_cards": [
                {"rank": "2", "suit": "s", "confidence": 0.93},
                {"rank": "8", "suit": "d", "confidence": 0.92},
                {"rank": "5", "suit": "s", "confidence": 0.91},
            ],
        }
        table_prediction = {
            "hero_seat": 1,
            "dealer_seat": 4,
            "pot_bb": {"value": "6.5", "confidence": 0.90},
            "to_call_bb": {"value": "0", "confidence": 0.89},
        }
        clock_prediction = {
            "remaining_seconds": 12,
            "remaining_ms": 12_000,
            "confidence": 0.88,
            "clock_box": {"x": 0.02, "y": 0.89, "width": 0.08, "height": 0.06},
        }
        name_prediction = {
            "layout_id": base_prediction["layout_id"],
            "player_names": [
                {
                    "seat_no": 1,
                    "player_name": "PLAYER01",
                    "confidence": 0.87,
                    "seat_box": {"x": 0.4, "y": 0.8, "width": 0.1, "height": 0.1},
                    "frame_sha256": "f" * 64,
                }
            ],
        }
        checkpoint = SimpleNamespace(payload={"checkpoint_sha256": "a" * 64})
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            Image.new("RGB", (200, 100), "black").save(image)
            with (
                patch("apc.perception.composite.predict_image", return_value=base_prediction),
                patch("apc.perception.composite.predict_cards", return_value=card_prediction),
                patch("apc.perception.composite.predict_table_state", return_value=table_prediction),
                patch("apc.perception.composite.predict_stacks", return_value=[]),
                patch("apc.perception.composite.predict_turn_clock", return_value=clock_prediction),
                patch("apc.perception.composite.predict_player_names", return_value=name_prediction),
                patch("apc.perception.composite._visual_signatures_or_abstain", return_value=([], None)),
            ):
                result = infer_visible_state(
                    image,
                    base_checkpoint=checkpoint,
                    card_checkpoint={"checkpoint_sha256": "b" * 64},
                    table_state_checkpoint={"checkpoint_sha256": "c" * 64},
                    stack_checkpoint={"checkpoint_sha256": "d" * 64},
                    turn_clock_checkpoint={"checkpoint_sha256": "e" * 64},
                    name_ocr_checkpoint={"checkpoint_sha256": "f" * 64},
                )

        self.assertTrue(result["visible_state"]["hero_to_act"])
        self.assertEqual(result["visible_state"]["decision_time_remaining_ms"], 12_000)
        self.assertEqual(result["visible_state"]["decision_deadline_source"], "visible_timer")
        self.assertEqual(result["field_confidence"]["decision_time_remaining_ms"], 0.88)
        self.assertEqual(result["checkpoint_provenance"]["turn_clock_sha256"], "e" * 64)
        self.assertEqual(
            result["visible_state"]["recognized_player_names"][0]["player_name"],
            "PLAYER01",
        )
        self.assertEqual(result["field_confidence"]["recognized_player_names"], 0.87)
        self.assertEqual(result["checkpoint_provenance"]["name_ocr_sha256"], "f" * 64)


if __name__ == "__main__":
    unittest.main()
