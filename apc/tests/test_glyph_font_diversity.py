import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from apc.annotator.project import AnnotationProject
from apc.perception.train_card_glyphs import token_dataset
from apc.synthetic.render_table import generate_dataset, render_frame, validate_symbol_font, THEMES


FONT = Path("C:/Windows/Fonts/segoeuib.ttf")
OTHER_FONT = Path("C:/Windows/Fonts/segoeui.ttf")


class GlyphFontDiversityTests(unittest.TestCase):
    def test_identical_missing_glyphs_are_rejected(self):
        class MissingGlyph:
            size = (1, 1)
            def __bytes__(self):
                return b"\xff"
        factory = SimpleNamespace(truetype=lambda *args, **kwargs: SimpleNamespace(getmask=lambda _: MissingGlyph()))
        with patch("apc.synthetic.render_table._pil", return_value=(None, None, factory)), \
                self.assertRaisesRegex(ValueError, "distinguish all ranks and suits"):
            validate_symbol_font("unused-font", [22])

    def test_font_size_requires_supported_range_and_stacked_design(self):
        for size, design in ((29, "stacked_symbols"), (True, "stacked_symbols"), (26, "corner_symbols")):
            with self.subTest(size=size, design=design), self.assertRaises(ValueError):
                render_frame(Path("unused.png"), rng=random.Random(1), session_id="test", sequence_index=0,
                    seats=2, layout_id="heads-up", theme=THEMES[0], street="flop",
                    card_design=design, glyph_font_size=size)

    @unittest.skipUnless(FONT.is_file() and OTHER_FONT.is_file(), "Local Unicode fonts unavailable")
    def test_distinct_font_pool_and_sizes_produce_trainable_verified_glyphs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_dataset(Path(directory) / "project", sessions=24, seed=81,
                card_design="mixed_symbols", rank_font=FONT, extra_rank_fonts=(OTHER_FONT,),
                glyph_font_sizes=(18, 26), include_glyph_boxes=True, alternate_builtin_font=True,
                card_surface="mixed")
            self.assertTrue(result["validation"]["valid"])
            self.assertEqual(len(set(result["font_pool_sha256"])), 2)
            self.assertEqual(result["glyph_font_sizes"], [18, 26])
            project = AnnotationProject(result["project"]["root"])
            themes = {record.environment["theme_id"] for record in project.records}
            for digest in result["font_pool_sha256"]:
                self.assertTrue(any(digest[:12] in theme for theme in themes))
            for size in (18, 26):
                self.assertTrue(any("glyph-size-" + str(size) in theme for theme in themes))
            for split in ("train", "validation", "test"):
                tokens, ranks, suits = token_dataset(result["manifest"], split, glyph_pair=True)
                self.assertEqual(tokens.shape[1:], (6, 32, 32))
                self.assertEqual(len(tokens), len(ranks))
                self.assertEqual(len(tokens), len(suits))

    @unittest.skipUnless(FONT.is_file(), "Local Unicode font unavailable")
    def test_duplicate_font_binaries_fail_before_creating_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            with self.assertRaisesRegex(ValueError, "distinct font binaries"):
                generate_dataset(root, sessions=3, seed=1, card_design="mixed_symbols",
                                 rank_font=FONT, extra_rank_fonts=(FONT,))
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
