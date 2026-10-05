import random
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from apc.perception.card_glyphs import glyph_tensor
from apc.synthetic.render_table import render_frame, THEMES
from apc.synthetic.render_table import generate_dataset
from apc.perception.train_card_glyphs import token_dataset


class CardGlyphTests(unittest.TestCase):
    def test_explicit_glyph_dataset_loads_six_channel_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            for design in ("centered_token", "corner_symbols"):
                result = generate_dataset(Path(directory) / design, sessions=3, seed=19,
                                          include_glyph_boxes=True, card_design=design)
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
