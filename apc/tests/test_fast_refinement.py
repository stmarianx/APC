from pathlib import Path
import unittest
import numpy as np
import torch
from PIL import Image,ImageDraw

from apc.perception.card_proposals import refine_card_surface
from apc.perception.fast_refinement import refine_fast,FastRefinementLocator
from apc.perception.infer_card_locator import NeuralCardLocator


class FastRefinementTests(unittest.TestCase):
    def test_adaptive_refinement_matches_on_seeded_geometry(self):
        rng = np.random.default_rng(20261108)
        image = Image.new("RGB",(160,200),(20,50,30))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((40,50,71,104),radius=5,fill=(245,243,233))
        draw.rounded_rectangle((76,50,107,104),radius=5,fill=(245,243,233))
        pixels = np.asarray(image,dtype=np.float32)
        for _ in range(100):
            x,y = rng.integers(0,100),rng.integers(0,140)
            w,h = rng.integers(8,60),rng.integers(8,60)
            proposal = {"box_xyxy":[x/160,y/200,(x+w)/160,(y+h)/200],"score":.8}
            self.assertEqual(refine_fast(image,proposal,pixels),
                refine_card_surface(image,proposal,pixels=pixels,adaptive=True))

    def test_whole_locator_matches_without_global_patch(self):
        class Model:
            def __call__(self,inputs):
                values = torch.full((len(inputs),5,23,40),-10.)
                values[:,0,10,10] = 8
                values[:,1:,10,10] = -2
                return values
        base = NeuralCardLocator.__new__(NeuralCardLocator)
        base.model = Model()
        base.refine_surfaces = base.adaptive_surfaces = True
        path = Path(__file__).resolve().parents[1]/"data/reference/capture-stability-demo-v1/frame.png"
        for tiled in (False,True):
            for dedup in (False,True):
                base.tiled_views,base.deduplicate_surfaces = tiled,dedup
                expected = base.propose(path); actual = FastRefinementLocator(base).propose(path)
                expected.pop("elapsed_ms"); actual.pop("elapsed_ms")
                self.assertEqual(actual,expected)
