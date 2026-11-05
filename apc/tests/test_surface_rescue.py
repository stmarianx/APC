from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image,ImageDraw

from apc.perception.surface_rescue import surface_proposals,SurfaceRescueLocator


class SurfaceRescueTests(unittest.TestCase):
    def image(self):
        image = Image.new("RGB",(320,400),(20,50,30))
        draw = ImageDraw.Draw(image)
        for x in (100,148):
            draw.rounded_rectangle((x,160,x+43,223),radius=5,fill=(245,243,233))
            draw.rectangle((x+7,170,x+12,184),fill="black")
        return image

    def test_proposes_distinct_light_surfaces_without_identity_or_count_input(self):
        proposals = surface_proposals(self.image())
        self.assertEqual(len(proposals),2)
        self.assertEqual(proposals[0]["box_xyxy"],[100/320,160/400,144/320,224/400])
        self.assertEqual(proposals[1]["box_xyxy"],[148/320,160/400,192/320,224/400])
        self.assertTrue(all("candidate" not in p and "card" not in p for p in proposals))
        self.assertEqual(surface_proposals(Image.new("RGB",(320,400),"black")),[])

    def test_neural_box_precedence_and_input_preservation(self):
        neural = {"box_xyxy":[100/320,160/400,144/320,224/400],"score":.9,"kind":"neural_card_candidate"}
        base = SimpleNamespace(propose=lambda path:{"proposals":[neural]})
        with patch("apc.perception.surface_rescue.Image.open",return_value=self.image()):
            result = SurfaceRescueLocator(base).propose("unused")
        self.assertEqual(len(result["proposals"]),2)
        self.assertEqual(result["proposals"][0],neural)
        self.assertNotIn("surface_fill",neural)
