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
