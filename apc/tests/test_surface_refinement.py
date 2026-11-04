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
