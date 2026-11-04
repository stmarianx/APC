import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from apc.perception.train_card_glyphs import train, augment_tokens
from apc.perception.infer_cards import CardReader


class GlyphTrainingTests(unittest.TestCase):
    def test_augmentation_is_seeded_bounded_and_preserves_blank_and_source(self):
        original = torch.rand(4, 6, 32, 32)
        original[0] = 0
        unchanged = original.clone()
        torch.manual_seed(123)
        first = augment_tokens(original)
        torch.manual_seed(123)
        second = augment_tokens(original)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(original, unchanged)
        self.assertEqual(first.shape, original.shape)
        self.assertEqual(first[0].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreaterEqual(first.min().item(), 0)
        self.assertLessEqual(first.max().item(), 1)
        self.assertFalse(torch.equal(first, original))
        with self.assertRaises(ValueError):
            augment_tokens(torch.zeros(4, 4, 32, 32))

    def test_pair_checkpoint_cannot_be_loaded_as_whole_card_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            dataset = (torch.rand(4, 6, 32, 32), torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))
            with patch("apc.perception.train_card_glyphs.token_dataset", return_value=dataset):
                report = train(manifest, root / "run", epochs=1, glyph_pair=True, augment=True)
            self.assertEqual(report["model_kind"], "annotated_card_glyph_pair_recognizer")
            self.assertEqual(report["token_augmentation"], "color_scale_translation_v1")
            with self.assertRaises(ValueError):
                CardReader(root / "run" / "glyph_weights.pt")

    def test_runner_checkpoint_selection_and_reader_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            dataset = (torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))
            with patch("apc.perception.train_card_glyphs.token_dataset", return_value=dataset):
                report = train(manifest, root / "run", epochs=1)
            self.assertEqual(report["test_exact_box"]["cards"], 4)
            self.assertFalse(report["full_table_readiness"])
            reader = CardReader(root / "run" / "glyph_weights.pt")
            self.assertTrue(reader.glyph_model)
            self.assertTrue((root / "run" / "evaluation.json").is_file())
            with self.assertRaises(ValueError):
                train(manifest, root / "run")
