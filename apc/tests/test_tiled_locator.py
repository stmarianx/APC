import unittest
from unittest.mock import patch

import torch
from PIL import Image

from apc.perception.card_proposals import suppress_duplicate_surfaces
from apc.perception.infer_card_locator import NeuralCardLocator


class TiledLocatorTests(unittest.TestCase):
    def test_deduplication_uses_only_geometry_and_keeps_adjacent_cards(self):
        boxes = [{"box_xyxy": [.1, .1, .2, .3], "score": .8, "candidate": "Ac"},
                 {"box_xyxy": [.1, .1, .2, .3], "score": .9, "candidate": "Kd"},
                 {"box_xyxy": [.21, .1, .31, .3], "score": .7, "candidate": "Kd"}]
        result = suppress_duplicate_surfaces(boxes)
        self.assertEqual(result, [boxes[1], boxes[2]])
        self.assertEqual(len(boxes), 3)
        self.assertEqual(suppress_duplicate_surfaces([]), [])
        for value in (0, -1, float("nan"), 2):
            with self.assertRaises(ValueError):
                suppress_duplicate_surfaces(boxes, threshold=value)

    def test_tiled_proposals_map_to_full_frame_and_need_surface_evidence(self):
        locator = object.__new__(NeuralCardLocator)
        locator.tiled_views = True
        locator.refine_surfaces = locator.adaptive_surfaces = True
        locator.deduplicate_surfaces = False
        batches = []
        def model(inputs):
            batches.append(inputs.shape)
            return torch.zeros(5, 5, 23, 40)
        locator.model = model
        proposal = {"box_xyxy": [.25, .25, .75, .75], "score": .8}
        mapped = []
        def refine(image, item, **kwargs):
            mapped.append(item)
            # Reject one tile without bounded evidence, not the original view.
            return dict(item, refinement="adaptive_no_bounded_surface" if item["source_view"] in (0, 4) else "adaptive_bounded_surface")
        with patch("apc.perception.infer_card_locator.Image.open", return_value=Image.new("RGB", (300, 180))), \
             patch("apc.perception.infer_card_locator.decode_locator", side_effect=lambda _: [dict(proposal)]), \
             patch("apc.perception.card_proposals.refine_card_surface", side_effect=refine):
            result = locator.propose("unused.png")
        self.assertEqual(batches, [torch.Size([5, 3, 184, 320])])
        self.assertEqual([p["source_view"] for p in result["proposals"]], [0, 1, 2, 3])
        for actual, expected in zip(mapped[4]["box_xyxy"], [.5, .5, 5/6, 5/6]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(mapped[0]["box_xyxy"], proposal["box_xyxy"])
        self.assertFalse(result["confidence_calibrated"])
        self.assertFalse(result["full_table_readiness"])

    def test_new_options_require_adaptive_mode(self):
        for options in ({"tiled_views": True}, {"deduplicate_surfaces": True}):
            with self.assertRaisesRegex(ValueError, "require adaptive"):
                NeuralCardLocator("missing.pt", **options)


if __name__ == "__main__":
    unittest.main()
