from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from apc.synthetic.render_table import (
    LAYOUTS,
    THEMES,
    CLOCK_VALUES_MS,
    NAME_OCR_CHARSET,
    NAME_OCR_LENGTH,
    action_display,
    normalized_box,
    render_frame,
    seat_boxes,
    synthetic_ocr_player_names,
)


class SyntheticRendererContractTests(unittest.TestCase):
    @unittest.skipUnless(Path("C:/Windows/Fonts/segoeuib.ttf").is_file(), "Local Unicode font unavailable")
    def test_stacked_unicode_cards_have_explicit_contained_glyph_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            rendered = render_frame(Path(directory) / "unicode.png", rng=random.Random(7),
                session_id="unicode-session", sequence_index=0, seats=6, layout_id="six-max",
                theme=THEMES[0], street="flop", card_design="stacked_symbols",
                rank_font=Path("C:/Windows/Fonts/segoeuib.ttf"), include_glyph_boxes=True)
            cards = rendered.annotation["objects"]["hero_cards"] + rendered.annotation["objects"]["board_cards"]
            for card in cards:
                parent = card["box"]
                for kind in ("rank_box", "suit_box"):
                    child = card[kind]
                    self.assertGreater(child["width"], 0)
                    self.assertGreater(child["height"], 0)
                    for axis, size in (("x", "width"), ("y", "height")):
                        self.assertGreaterEqual(child[axis], parent[axis])
                        self.assertLessEqual(child[axis] + child[size], parent[axis] + parent[size] + 1e-9)

    def test_stacked_unicode_cards_require_an_explicit_font(self):
        with self.assertRaisesRegex(ValueError, "explicit local font"):
            render_frame(Path("unused.png"), rng=random.Random(7), session_id="unicode-session",
                sequence_index=0, seats=6, layout_id="six-max", theme=THEMES[0], street="flop",
                card_design="stacked_symbols")

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
        self.assertEqual(set("0123456789"), set("".join(str(value // 1000) for value in CLOCK_VALUES_MS)))

    def test_hero_seat_clears_the_fixed_hole_card_region(self) -> None:
        for seats, _ in LAYOUTS:
            hero = seat_boxes(seats)[0]
            self.assertLessEqual(hero[3], 580)

    def test_call_price_is_visible_and_machine_labeled(self) -> None:
        self.assertEqual(action_display("call", "2.5"), ("Call 2.5 BB", "2.5"))
        self.assertEqual(action_display("check", "0"), ("Check", None))

    def test_optional_turn_clock_has_exact_canonical_milliseconds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rendered = render_frame(
                Path(directory) / "clock.png",
                rng=random.Random(7),
                session_id="clock-session",
                sequence_index=0,
                seats=6,
                layout_id="six-max",
                theme=THEMES[0],
                street="flop",
                decision_time_remaining_ms=12_000,
            )
            self.assertEqual(rendered.annotation["state"]["decision_time_remaining_ms"], 12_000)
            self.assertEqual(rendered.annotation["objects"]["turn_clock"]["remaining_ms"], 12_000)
            self.assertTrue(rendered.image_path.is_file())

    def test_synthetic_ocr_names_are_distinct_and_fixed_advance(self) -> None:
        names = synthetic_ocr_player_names(random.Random(19), 9)
        self.assertEqual(len(names), 9)
        self.assertEqual(len(set(names.values())), 9)
        self.assertTrue(
            all(
                len(name) == NAME_OCR_LENGTH
                and set(name) <= set(NAME_OCR_CHARSET)
                for name in names.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
