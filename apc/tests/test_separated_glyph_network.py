import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from apc.perception.card_glyphs import SeparatedCardGlyphNetwork
from apc.perception.infer_cards import CardReader
from apc.perception.train_card_glyphs import train


class SeparatedGlyphNetworkTests(unittest.TestCase):
    def test_rank_and_suit_outputs_depend_only_on_their_own_glyph(self):
        torch.manual_seed(17)
        model = SeparatedCardGlyphNetwork().eval()
        tokens = torch.rand(2, 6, 32, 32)
        original = model(tokens)
        changed_rank = tokens.clone()
        changed_rank[:, :3] = 0
        changed_suit = tokens.clone()
        changed_suit[:, 3:] = 0
        torch.testing.assert_close(model(changed_rank)["suit"], original["suit"])
        torch.testing.assert_close(model(changed_suit)["rank"], original["rank"])
        self.assertFalse(torch.equal(model(changed_rank)["rank"], original["rank"]))
        self.assertFalse(torch.equal(model(changed_suit)["suit"], original["suit"]))

    def test_suit_gradients_do_not_flow_through_rank_input(self):
        tokens = torch.rand(2, 6, 32, 32, requires_grad=True)
        model = SeparatedCardGlyphNetwork()
        model(tokens)["suit"].sum().backward()
        self.assertEqual(float(tokens.grad[:, :3].abs().sum()), 0)
        self.assertGreater(float(tokens.grad[:, 3:].abs().sum()), 0)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 32, 32))

    def test_trained_split_checkpoint_roundtrips_with_explicit_pair_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            data = (torch.rand(4, 6, 32, 32), torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))
            with patch("apc.perception.train_card_glyphs.token_dataset", return_value=data):
                report = train(manifest, root / "run", epochs=1, glyph_pair=True, separate_heads=True)
            self.assertEqual(report["architecture_version"], "card-glyph-pair-split-v1")
            path = root / "run/glyph_weights.pt"
            reader = CardReader(path, corner_pairs=True, surface_pairs=True)
            self.assertIsInstance(reader.model, SeparatedCardGlyphNetwork)
            self.assertEqual(reader.pair_extraction, "surface_two_band_then_corner")
            with self.assertRaises(ValueError):
                CardReader(path)
            with self.assertRaisesRegex(ValueError, "paired glyph data"):
                train(manifest, root / "invalid", epochs=1, separate_heads=True)
            self.assertFalse((root / "invalid").exists())

    def test_robust_validation_views_do_not_expand_the_test_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            data = (torch.rand(4, 6, 32, 32), torch.arange(4), torch.arange(4))
            with patch("apc.perception.train_card_glyphs.token_dataset", return_value=data):
                report = train(manifest, root / "run", epochs=1, glyph_pair=True,
                               separate_heads=True, validation_augmentations=2)
            self.assertEqual(report["history"][0]["validation"]["cards"], 12)
            self.assertEqual(report["test_exact_box"]["cards"], 4)
            self.assertIsInstance(report["history"][0]["validation_loss"], float)
            self.assertEqual(report["validation_augmentations"], 2)


if __name__ == "__main__":
    unittest.main()
