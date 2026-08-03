from __future__ import annotations

import unittest

from apc.perception.stack_baseline import (
    LEGACY_MODEL_KIND,
    MODEL_KIND,
    _shape_geometry,
    _token_shape,
    digit_box,
)


class StackBaselineTests(unittest.TestCase):
    def test_digit_boxes_are_ordered_inside_seat(self) -> None:
        seat = {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.2}
        boxes = [digit_box(seat, digit_count=3, digit_index=index) for index in range(3)]
        self.assertLess(boxes[0]["x"], boxes[1]["x"])
        self.assertLess(boxes[1]["x"], boxes[2]["x"])
        for box in boxes:
            self.assertGreaterEqual(box["x"], seat["x"])
            self.assertLessEqual(box["x"] + box["width"], seat["x"] + seat["width"])

    def test_digit_box_supports_decimal_token_geometry(self) -> None:
        seat = {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.2}
        boxes = [
            digit_box(seat, digit_count=5, digit_index=index, decimal_index=3)
            for index in range(5)
        ]
        self.assertEqual(len(boxes), 5)
        self.assertTrue(all(left["x"] < right["x"] for left, right in zip(boxes, boxes[1:])))
        self.assertLess(boxes[3]["width"], boxes[2]["width"])

    def test_token_shape_preserves_decimal_position_without_value_label(self) -> None:
        self.assertEqual(_token_shape("86.5"), "decimal:2:1")
        self.assertEqual(_shape_geometry("decimal:2:1"), (4, 2))
        self.assertEqual(_token_shape("106"), "integer:3")

    def test_new_model_kind_is_distinct_from_legacy_integer_checkpoint(self) -> None:
        self.assertNotEqual(MODEL_KIND, LEGACY_MODEL_KIND)

    def test_digit_box_rejects_unsupported_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "one to five"):
            digit_box({"x": 0, "y": 0, "width": 1, "height": 1}, digit_count=6, digit_index=0)


if __name__ == "__main__":
    unittest.main()
