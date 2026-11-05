from pathlib import Path
import unittest

import torch

from apc.perception.infer_card_locator import NeuralCardLocator
from apc.perception.trace_locator_stages import trace_frame


class LocatorTraceTests(unittest.TestCase):
    def test_wrappers_preserve_incumbent_proposals_and_separate_control(self):
        class Model:
            def __call__(self,inputs):
                values = torch.full((len(inputs),5,23,40),-10.)
                values[:,0,10,10] = 8
                values[:,0,10,11] = 7  # Adjacent peak suppressed by incumbent grid decoder.
                values[:,1:,10,10:12] = -2
                values[:,3,10,10:12] = -4  # Narrow disjoint boxes survive box-IoU suppression.
                return values
        locator = NeuralCardLocator.__new__(NeuralCardLocator)
        locator.model = Model()
        locator.refine_surfaces = True
        locator.adaptive_surfaces = True
        locator.tiled_views = True
        locator.deduplicate_surfaces = True
        image = Path(__file__).resolve().parents[1] / "data/reference/capture-stability-demo-v1/frame.png"
        before = locator.propose(image)["proposals"]
        report = trace_frame({"image":image,"targets":[{"identity":"As","box_xyxy":[.1,.1,.2,.2]}]},locator)
        after = locator.propose(image)["proposals"]
        self.assertEqual(before,after)
        counts = report["stage_proposal_counts"]
        self.assertEqual(counts["decoded_default"],5)
        self.assertGreater(counts["box_nms_decode_control"],counts["decoded_default"])
        self.assertEqual(counts["final"],len(before))
        self.assertTrue(report["box_nms_is_diagnostic_control_only"])
        self.assertEqual(set(report["targets"][0]["best_iou_by_stage"]),set(counts))


if __name__ == "__main__":
    unittest.main()
