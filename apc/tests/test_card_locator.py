import unittest
import torch
from apc.perception.card_locator import CardLocator, locator_targets, locator_loss, decode_locator


class LocatorTests(unittest.TestCase):
    def test_box_suppression_retains_adjacent_cards_that_grid_peaks_drop(self):
        boxes = [[.09, .2, .12, .3], [.125, .2, .155, .3]]
        target = locator_targets(boxes, 23, 40)
        output = torch.full_like(target, -20)
        for index, (row, col) in enumerate(target[0].nonzero().tolist()):
            output[0, row, col] = 10-index
            output[1:, row, col] = torch.logit(target[1:, row, col])
        self.assertEqual(len(decode_locator(output)), 1)
        proposals = decode_locator(output, box_nms=True)
        self.assertEqual(len(proposals), 2)
        for proposal, box in zip(proposals, boxes):
            torch.testing.assert_close(torch.tensor(proposal["box_xyxy"]), torch.tensor(box))

    def test_box_suppression_removes_overlapping_duplicate_regions(self):
        output = torch.full((5, 23, 40), -20.0)
        for col, score in ((5, 10), (6, 9)):
            output[0, 5, col] = score
            output[1:, 5, col] = torch.logit(torch.tensor([.6, .75, .1, .1]))
        proposals = decode_locator(output, box_nms=True)
        self.assertEqual(len(proposals), 1)
        self.assertAlmostEqual(proposals[0]["score"], float(torch.sigmoid(torch.tensor(10.0))))
        with self.assertRaises(ValueError):
            decode_locator(output, box_nms=True, iou_threshold=0)

    def test_nonfinite_predicted_geometry_is_not_emitted(self):
        output = torch.full((5, 8, 12), -20.0)
        output[0, 3, 4] = 10
        output[1, 3, 4] = float("nan")
        self.assertEqual(decode_locator(output), [])
        self.assertEqual(decode_locator(output, box_nms=True), [])

    def test_hard_negative_mining_emphasizes_false_positives_without_dropping_positives(self):
        target = locator_targets([[.1, .1, .2, .3]], 23, 40)[None]
        raw = torch.zeros_like(target)
        raw[:, 0] = -10
        raw[0, 0, 20, 35] = 5
        baseline = raw.clone().requires_grad_()
        mined = raw.clone().requires_grad_()
        locator_loss(baseline, target).backward()
        locator_loss(mined, target, hard_negative_ratio=3).backward()
        self.assertGreater(float(mined.grad[0, 0, 20, 35]), float(baseline.grad[0, 0, 20, 35]) * 10)
        positive = target[0, 0].bool()
        torch.testing.assert_close(mined.grad[0, 0][positive], baseline.grad[0, 0][positive])
        self.assertLess(int((mined.grad[0, 0] != 0).sum()), 40)

    def test_hard_negative_mining_handles_empty_and_all_positive_frames(self):
        for present in (0, 1):
            target = torch.zeros(2, 5, 8, 12)
            target[:, 0] = present
            if present:
                target[:, 1:] = .25
            output = torch.zeros_like(target, requires_grad=True)
            loss = locator_loss(output, target, hard_negative_ratio=3)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(output.grad).all())
        for ratio in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                locator_loss(output, target, hard_negative_ratio=ratio)

    def test_training_gradient_and_empty_frame(self):
        model = CardLocator()
        result = model(torch.zeros(1, 3, 64, 96))
        self.assertEqual(result.shape, (1, 5, 8, 12))
        for boxes in ([], [[.1, .1, .2, .4]]):
            target = locator_targets(boxes, 8, 12)[None]
            loss = locator_loss(result, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward(retain_graph=True)
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_collision_and_blank_decode(self):
        with self.assertRaises(ValueError):
            locator_targets([[.1,.1,.2,.2]] * 2, 8, 12)
        self.assertEqual(decode_locator(torch.full((5,8,12), -20.0)), [])

    def test_decode_recovers_geometry_without_target_at_inference(self):
        box = [.1, .1, .2, .4]
        target = locator_targets([box], 8, 12)
        row, col = target[0].nonzero()[0].tolist()
        output = torch.full_like(target, -20)
        output[0, row, col] = 10
        output[1:, row, col] = torch.logit(target[1:, row, col])
        proposals = decode_locator(output)
        self.assertEqual(len(proposals), 1)
        torch.testing.assert_close(torch.tensor(proposals[0]["box_xyxy"]), torch.tensor(box))

    def test_relative_size_loss_pushes_collapsed_width_up(self):
        target = locator_targets([[.1,.1,.2,.4]], 8, 12)[None]
        row, col = target[0,0].nonzero()[0].tolist()
        raw = torch.zeros_like(target)
        raw[0,3,row,col] = torch.logit(torch.tensor(.01))
        standard = raw.clone().requires_grad_()
        relative = raw.clone().requires_grad_()
        locator_loss(standard, target).backward()
        locator_loss(relative, target, relative_size=True).backward()
        self.assertLess(float(relative.grad[0,3,row,col]), 0)
        self.assertGreater(abs(float(relative.grad[0,3,row,col])), abs(float(standard.grad[0,3,row,col])))
