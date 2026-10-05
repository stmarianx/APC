import random
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from apc.perception.infer_cards import CardReader
from apc.perception.region_model import RegionRecognizer
from apc.synthetic.render_table import render_frame, THEMES


class CardReaderTests(unittest.TestCase):
    def checkpoint(self, root):
        path = root / "weights.pt"
        torch.save({"state_dict": RegionRecognizer(16).state_dict(), "crop_size": 16,
                    "architecture_version": "spatial-grid-v2",
                    "model_kind": "region_recognizer_requires_proposed_boxes"}, path)
        return path

    def test_unknown_model_abstains_from_fixture_cards(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame.png"
            render_frame(frame, rng=random.Random(12), session_id="session", sequence_index=0,
                         seats=6, layout_id="six-max", theme=THEMES[0], street="flop")
            result = CardReader(self.checkpoint(root), confidence_threshold=1).read(frame)
            self.assertEqual(len(result["cards"]), 5)
            self.assertTrue(all(item["card"] is None for item in result["cards"]))
            self.assertFalse(result["confidence_calibrated"])
            self.assertFalse(result["full_table_readiness"])
            # Even high softmax scores are not confirmed identities before
            # calibration/proposal-shift evaluation has passed.
            reader = CardReader(self.checkpoint(root), confidence_threshold=0)
            high_scores = reader.read(frame)
            self.assertTrue(all(item["card"] is None for item in high_scores["cards"]))

    def test_blank_and_invalid_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root)
            image = root / "blank.png"
            Image.new("RGB", (640, 360), "black").save(image)
            self.assertEqual(CardReader(checkpoint).read(image)["cards"], [])
            for threshold in (-1, 2, float("nan")):
                with self.assertRaises(ValueError):
                    CardReader(checkpoint, confidence_threshold=threshold)
