from __future__ import annotations

import unittest

from apc.perception.table_state_baseline import _relative_box, number_character_box


class TableStateBaselineTests(unittest.TestCase):
    def test_relative_box_can_include_dealer_region_outside_seat(self) -> None:
        result = _relative_box(
            {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.2},
            (0.8, -0.2, 1.1, 0.4),
        )
        self.assertAlmostEqual(result["x"], 0.52)
        self.assertAlmostEqual(result["y"], 0.26)
        self.assertAlmostEqual(result["width"], 0.12)
        self.assertAlmostEqual(result["height"], 0.12)

    def test_decimal_character_uses_narrower_numeric_geometry(self) -> None:
        container = {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.2}
        digit = number_character_box(
            container,
            region_width_px=180,
            prefix_px=16,
            character_count=4,
            character_index=1,
            decimal_index=2,
        )
        decimal = number_character_box(
            container,
            region_width_px=180,
            prefix_px=16,
            character_count=4,
            character_index=2,
            decimal_index=2,
        )
        self.assertLess(decimal["width"], digit["width"])


if __name__ == "__main__":
    unittest.main()
