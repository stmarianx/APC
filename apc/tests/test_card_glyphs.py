import random
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from apc.perception.card_glyphs import glyph_tensor, surface_pair_tensor, adaptive_pair_tensor, corner_pair_tensor
from apc.synthetic.render_table import render_frame, THEMES
from apc.synthetic.render_table import generate_dataset
from apc.perception.train_card_glyphs import token_dataset


class CardGlyphTests(unittest.TestCase):
    @unittest.skipUnless(Path("C:/Windows/Fonts/segoeuib.ttf").is_file(), "Local Unicode font unavailable")
    def test_mixed_unicode_dataset_loads_verified_glyph_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_dataset(Path(directory) / "mixed", sessions=6, seed=29,
                include_glyph_boxes=True, card_design="mixed_symbols",
                rank_font=Path("C:/Windows/Fonts/segoeuib.ttf"),
                alternate_builtin_font=True, card_surface="mixed")
            self.assertTrue(result["validation"]["valid"])
            for split in ("train", "validation", "test"):
                tokens, ranks, suits = token_dataset(result["manifest"], split, glyph_pair=True)
                self.assertEqual(tokens.shape[1:], (6, 32, 32))
                self.assertEqual(len(tokens), len(ranks))
                self.assertEqual(len(tokens), len(suits))

    def test_surface_pairs_preserve_two_centered_separated_glyphs(self):
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 15, 58, 30), fill="black")
        draw.ellipse((39, 62, 59, 80), fill="green")
        token, present = surface_pair_tensor(image, [0, 0, 1, 1])
        self.assertTrue(present)
        self.assertEqual(token.shape, (6, 32, 32))
        expected_rank, _ = glyph_tensor(image, [.4, .15, .59, .31], border_margin=False, background_rgb=[255]*3)
        expected_suit, _ = glyph_tensor(image, [.39, .62, .60, .81], border_margin=False, background_rgb=[255]*3)
        torch.testing.assert_close(token, torch.cat([expected_rank, expected_suit]))

    def test_surface_pairs_reject_blank_extra_bands_and_invalid_crop(self):
        image = Image.new("RGB", (100, 100), "white")
        self.assertFalse(surface_pair_tensor(image, [0, 0, 1, 1])[1])
        draw = ImageDraw.Draw(image)
        for top in (10, 40, 70):
            draw.rectangle((40, top, 60, top+10), fill="black")
        self.assertFalse(surface_pair_tensor(image, [0, 0, 1, 1])[1])
        with self.assertRaises(ValueError):
            surface_pair_tensor(image, [-.1, 0, 1, 1])

    def test_adaptive_falls_back_for_corner_cards_with_large_decorative_suit(self):
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 25, 20), fill="black")
        draw.rectangle((10, 30, 25, 40), fill="green")
        draw.ellipse((50, 65, 85, 85), fill="green")
        self.assertFalse(surface_pair_tensor(image, [0, 0, 1, 1])[1])
        expected, expected_present = corner_pair_tensor(image, [0, 0, 1, 1])
        actual, present = adaptive_pair_tensor(image, [0, 0, 1, 1])
        self.assertTrue(expected_present and present)
        torch.testing.assert_close(actual, expected)

    def test_explicit_glyph_dataset_loads_six_channel_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            for design, surface in (("centered_token", "light"), ("corner_symbols", "light"), ("corner_symbols", "dim")):
                result = generate_dataset(Path(directory) / (design + surface), sessions=3, seed=19,
                                          include_glyph_boxes=True, card_design=design, card_surface=surface)
                self.assertTrue(result["validation"]["valid"])
                tokens, ranks, suits = token_dataset(result["manifest"], "train", glyph_pair=True)
                self.assertEqual(tokens.shape[1:], (6, 32, 32))
                self.assertEqual(len(tokens), len(ranks))
                self.assertEqual(len(tokens), len(suits))

    def test_one_pixel_shift_preserves_normalized_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            frame = render_frame(path, rng=random.Random(12), session_id="session", sequence_index=0,
                                 seats=6, layout_id="six-max", theme=THEMES[0], street="flop")
            box = frame.annotation["objects"]["hero_cards"][0]["box"]
            coords = [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]]
            with Image.open(path) as image:
                baseline, present = glyph_tensor(image, coords)
                shifted = [value + 1 / (image.width if i % 2 == 0 else image.height) for i, value in enumerate(coords)]
                changed, shifted_present = glyph_tensor(image, shifted)
            self.assertTrue(present and shifted_present)
            torch.testing.assert_close(baseline, changed)

    def test_blank_crop_is_explicitly_missing(self):
        token, present = glyph_tensor(Image.new("RGB", (100, 100), "white"), [.1, .1, .9, .9])
        self.assertFalse(present)
        self.assertEqual(token.abs().sum().item(), 0)
