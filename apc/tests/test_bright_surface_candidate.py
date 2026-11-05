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

    def test_wide_search_recovers_card_clipped_by_narrow_proposal_roi(self):
        image = Image.new("RGB",(560,900),(20,50,30))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((223,540,266,603),radius=5,fill=(245,243,233))
        draw.rounded_rectangle((269,540,312,603),radius=5,fill=(245,243,233))
        proposal = {"box_xyxy":[225/560,540/900,242/560,611/900],"refinement":"adaptive_no_bounded_surface"}
        self.assertEqual(recover_surface(image,proposal),proposal)
        recovered = recover_surface(image,proposal,wide_search=True)
        self.assertEqual(recovered["box_xyxy"],[223/560,540/900,267/560,604/900])
        self.assertEqual(recovered["surface_recovery"],"bright_seed_wide_v2")


if __name__ == "__main__":
    unittest.main()
