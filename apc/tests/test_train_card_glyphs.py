import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from apc.perception.train_card_glyphs import train
from apc.perception.infer_cards import CardReader


class GlyphTrainingTests(unittest.TestCase):
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
