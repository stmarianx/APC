import unittest
from PIL import Image, ImageDraw
from apc.perception.card_proposals import refine_card_surface


class SurfaceRefinementTests(unittest.TestCase):
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
