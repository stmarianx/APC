import unittest

from PIL import Image,ImageDraw

from apc.perception.bright_surface_candidate import recover_surface


class BrightSurfaceTests(unittest.TestCase):
    def test_recovers_complete_light_card_from_background_majority_proposal(self):
        image = Image.new("RGB",(120,160),(20,50,30))
        ImageDraw.Draw(image).rectangle((40,40,79,99),fill=(245,243,233))
        proposal = {"box_xyxy":[30/120,35/160,100/120,100/160],"refinement":"adaptive_no_bounded_surface"}
        result = recover_surface(image,proposal)
        self.assertEqual(result["box_xyxy"],[40/120,40/160,80/120,100/160])
        self.assertEqual(result["surface_recovery"],"bright_seed_v1")
        self.assertEqual(proposal["refinement"],"adaptive_no_bounded_surface")

    def test_dark_scene_does_not_invent_card(self):
        proposal = {"box_xyxy":[.2,.2,.8,.8],"refinement":"adaptive_no_bounded_surface"}
        self.assertEqual(recover_surface(Image.new("RGB",(100,100),"black"),proposal),proposal)

    def test_already_bounded_surface_is_unchanged(self):
        proposal = {"box_xyxy":[.2,.2,.8,.8],"refinement":"adaptive_bounded_surface"}
        self.assertEqual(recover_surface(Image.new("RGB",(100,100),"white"),proposal),proposal)


if __name__ == "__main__":
    unittest.main()
