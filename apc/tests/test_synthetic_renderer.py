from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from apc.synthetic.render_table import LAYOUTS, THEMES, action_display, normalized_box, seat_boxes


class SyntheticRendererContractTests(unittest.TestCase):
    def test_layout_boxes_are_normalized_and_nonoverlapping_by_identity(self) -> None:
        for seats, _ in LAYOUTS:
            boxes = seat_boxes(seats)
            self.assertEqual(len(boxes), seats)
            self.assertEqual(len(set(boxes)), seats)
            for box in boxes:
                normalized = normalized_box(box)
                self.assertGreater(normalized["width"], 0)
                self.assertGreater(normalized["height"], 0)
                self.assertGreaterEqual(normalized["x"], 0)
                self.assertLessEqual(normalized["x"] + normalized["width"], 1)
                self.assertGreaterEqual(normalized["y"], 0)
                self.assertLessEqual(normalized["y"] + normalized["height"], 1)

    def test_bootstrap_variants_cover_three_layouts_and_two_themes(self) -> None:
        self.assertEqual({seats for seats, _ in LAYOUTS}, {2, 6, 9})
        self.assertGreaterEqual(len(THEMES), 2)
        self.assertEqual(len({theme["id"] for theme in THEMES}), len(THEMES))

    def test_hero_seat_clears_the_fixed_hole_card_region(self) -> None:
        for seats, _ in LAYOUTS:
            hero = seat_boxes(seats)[0]
            self.assertLessEqual(hero[3], 580)

    def test_call_price_is_visible_and_machine_labeled(self) -> None:
        self.assertEqual(action_display("call", "2.5"), ("Call 2.5 BB", "2.5"))
        self.assertEqual(action_display("check", "0"), ("Check", None))


if __name__ == "__main__":
    unittest.main()
