import unittest
import numpy as np
from PIL import Image,ImageDraw

from apc.perception.card_proposals import scanline_components
from apc.perception.fast_components import fast_components
from apc.perception.bright_surface_candidate import recover_surface


class FastComponentsTests(unittest.TestCase):
    def test_random_and_adversarial_masks_preserve_exact_order_bounds_area(self):
        rng = np.random.default_rng(20261106)
        masks = [np.zeros((0,0),bool),np.zeros((5,7),bool),np.ones((5,7),bool),np.eye(12,dtype=bool)]
        for density in (.01,.1,.5,.9,.99):
            masks.extend(rng.random((37,59)) < density for _ in range(40))
        for index,mask in enumerate(masks):
            with self.subTest(index=index):
                self.assertEqual(fast_components(mask),scanline_components(mask))

    def test_recovery_is_exactly_equivalent(self):
        image = Image.new("RGB",(560,900),(20,50,30))
        ImageDraw.Draw(image).rounded_rectangle((223,540,266,603),radius=5,fill=(245,243,233))
        proposal = {"box_xyxy":[225/560,540/900,242/560,611/900],"refinement":"adaptive_no_bounded_surface"}
        self.assertEqual(recover_surface(image,proposal,wide_search=True),
                         recover_surface(image,proposal,wide_search=True,fast_components=True))
