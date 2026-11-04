import unittest
import torch
from apc.perception.card_locator import CardLocator, locator_targets, locator_loss, decode_locator


class LocatorTests(unittest.TestCase):
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
