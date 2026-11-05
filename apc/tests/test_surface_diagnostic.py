import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from apc.perception.diagnose_surface_refinement import diagnose


class SurfaceDiagnosticTests(unittest.TestCase):
    def test_modes_are_matched_for_diagnostics_without_confirmed_output(self):
        # Existing file supplies a stable checksum; inference and loading are
        # mocked so no saved audit labels can influence proposal generation.
        fixture = Path(__file__)
        frames = [{"image": fixture, "targets": [{"identity": "Ac", "box_xyxy": [.2, .2, .5, .6]}]}]
        prediction = {"rank": torch.tensor([[1.] + [0.] * 12]),
                      "suit": torch.tensor([[1., 0., 0., 0.]]), "ink_present": torch.tensor([True])}
        with patch("apc.perception.diagnose_surface_refinement.load_collection", return_value=({}, frames)), \
             patch("apc.perception.diagnose_surface_refinement.CardReader") as reader, \
             patch("apc.perception.diagnose_surface_refinement.NeuralCardLocator") as locator, \
             patch("apc.perception.diagnose_surface_refinement.Image.open", return_value=Image.new("RGB", (100, 100))):
            locator.return_value.propose.return_value = {"proposals": [{"box_xyxy": [.2, .2, .5, .6]}]}
            reader.return_value.predict_regions.return_value = prediction
            result = diagnose(fixture, fixture, fixture)
        self.assertTrue(result["reference_reuse"])
        self.assertFalse(result["selection_uses_labels"])
        self.assertFalse(result["full_table_readiness"])
        self.assertEqual(set(result["frames"][0]["modes"]), {"raw", "legacy", "adaptive"})
        self.assertEqual(result["frames"][0]["modes"]["raw"][0]["target"], "Ac")
        self.assertEqual(reader.return_value.predict_regions.call_count, 3)


if __name__ == "__main__":
    unittest.main()
