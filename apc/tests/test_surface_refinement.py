import unittest
from PIL import Image, ImageDraw
from apc.perception.card_proposals import refine_card_surface


class SurfaceRefinementTests(unittest.TestCase):
    def test_scanline_matches_pixel_components(self):
        import numpy as np
        from apc.perception.card_proposals import scanline_components
        from apc.perception.table_locator import _components
        rng = np.random.default_rng(42)
        for probability in (0, .1, .5, .9, 1):
            for _ in range(10):
                mask = rng.random((25, 31)) < probability
                self.assertEqual(scanline_components(mask), _components(mask))

    def test_light_and_dim_rectangles_refine_without_identity_labels(self):
        for color in ("#f7f3e8", "#8b8983"):
            image = Image.new("RGB", (200,200), "#183b34")
            ImageDraw.Draw(image).rectangle((60,40,99,119), fill=color)
            result = refine_card_surface(image, {"box_xyxy": [.32,.22,.48,.58]})
            self.assertEqual(result["box_xyxy"], [.3,.2,.5,.6])
            self.assertEqual(result["refinement"], "local_surface_component")

    def test_invalid_box_rejected(self):
        with self.assertRaises(ValueError):
            refine_card_surface(Image.new("RGB",(20,20)), {"box_xyxy": [0,0,0,1]})

    def test_adaptive_recovers_narrow_proposal_without_neighbor_drift(self):
        for color in ("#f7f3e8", "#8b8983"):
            image = Image.new("RGB", (200, 200), "#183b34")
            draw = ImageDraw.Draw(image)
            draw.rectangle((60, 40, 99, 119), fill=color)
            draw.rectangle((110, 40, 149, 119), fill=color)
            draw.rectangle((65, 130, 170, 155), fill=color)
            proposal = {"box_xyxy": [.37, .22, .47, .58], "score": .8}
            before = image.tobytes()
            legacy = refine_card_surface(image, proposal)
            adaptive = refine_card_surface(image, proposal, adaptive=True)
            self.assertNotEqual(legacy["box_xyxy"], [.3, .2, .5, .6])
            self.assertEqual(adaptive["box_xyxy"], [.3, .2, .5, .6])
            self.assertEqual(adaptive["refinement"], "adaptive_bounded_surface")
            self.assertEqual(adaptive["expansion"], 1.4)
            self.assertEqual(adaptive["score"], .8)
            self.assertEqual(proposal, {"box_xyxy": [.37, .22, .47, .58], "score": .8})
            self.assertEqual(before, image.tobytes())

    def test_adaptive_keeps_original_if_surface_is_unbounded(self):
        image = Image.new("RGB", (200, 200), "#f7f3e8")
        proposal = {"box_xyxy": [.3, .2, .5, .6]}
        result = refine_card_surface(image, proposal, adaptive=True)
        self.assertEqual(result["box_xyxy"], proposal["box_xyxy"])
        self.assertEqual(result["refinement"], "adaptive_no_bounded_surface")

    def test_adaptive_mode_requires_refinement_and_locator(self):
        from apc.perception.infer_card_locator import NeuralCardLocator
        from apc.perception.evaluate_card_reader import evaluate_reader
        with self.assertRaisesRegex(ValueError, "require"):
            NeuralCardLocator("nonexistent.pt", adaptive_surfaces=True)
        with self.assertRaisesRegex(ValueError, "require"):
            evaluate_reader("nonexistent.pt", "nonexistent.json", adaptive_surfaces=True)
