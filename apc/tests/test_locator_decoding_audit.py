import unittest
import torch

from apc.perception.card_locator import locator_targets
from apc.perception.evaluate_locator_decoding import missed_card_diagnostics


class LocatorDecodingAuditTests(unittest.TestCase):
    def run_case(self, boxes, *, score=10, tiny_geometry=False):
        target = locator_targets(boxes, 23, 40)
        output = torch.full_like(target, -20)
        for index, (row, col) in enumerate(target[0].nonzero().tolist()):
            output[0, row, col] = score-index
            output[1:, row, col] = torch.logit(target[1:, row, col])
            if tiny_geometry:
                output[3:, row, col] = -10
        truth = [[{"box_xyxy": box} for box in boxes]]
        data = (torch.zeros(1, 3, 184, 320), target[None], truth)
        return missed_card_diagnostics(lambda _: output[None], data)

    def test_suppressed_neighbor_is_distinguished_from_score_and_geometry(self):
        report = self.run_case([[.09, .2, .12, .3], [.125, .2, .155, .3]])
        self.assertEqual(report["reason_counts"], {"grid_suppression_or_assignment": 1})
        self.assertTrue(report["uses_ground_truth_for_diagnosis_only"])

    def test_low_score_and_bad_geometry_are_reported_separately(self):
        box = [[.09, .2, .12, .3]]
        self.assertEqual(self.run_case(box, score=-2)["reason_counts"], {"low_target_cell_score": 1})
        self.assertEqual(self.run_case(box, tiny_geometry=True)["reason_counts"], {"box_geometry": 1})
        self.assertEqual(self.run_case(box)["reason_counts"], {})


if __name__ == "__main__":
    unittest.main()
