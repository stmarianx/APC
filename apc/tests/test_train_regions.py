import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from apc.perception.spatial_targets import encode_annotation
from apc.perception.train_regions import train
from apc.tests.test_validate_dataset import annotation


class RegionRunnerTests(unittest.TestCase):
    def test_train_evaluate_checkpoint_on_fixture_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            splits = {}
            for index, split in enumerate(("train", "validation", "test")):
                image = root / (split + ".png")
                Image.new("RGB", (96, 64), (index * 40, 70, 100)).save(image)
                item = annotation(split, split, str(image), "a" * 64)
                splits[split] = [encode_annotation(item)]
            # This test exercises the runner, not manifest validation; that
            # boundary has independent hash/session tests in test_validate_dataset.
            with patch("apc.perception.train_regions.load_split", side_effect=lambda path, split: splits[split]):
                result = train(manifest, root / "run", epochs=1, crop_size=16)
            self.assertEqual(result["test"]["frames"], 1)
            self.assertFalse(result["full_table_readiness"])
            saved = torch.load(root / "run" / "region_weights.pt", weights_only=True)
            self.assertIn("state_dict", saved)
            report = json.loads((root / "run" / "evaluation.json").read_text())
            self.assertEqual(report["test"]["heads"]["rank"]["supervised_regions"], 5)
            with self.assertRaises(ValueError):
                train(manifest, root / "run", epochs=1)
