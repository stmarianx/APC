import unittest
import torch
from apc.neural.train_candidate import paired_expected_regret


class PairedRegretTests(unittest.TestCase):
    def test_gradient_prefers_higher_return_and_is_translation_invariant(self):
        values = torch.zeros(1,4,requires_grad=True)
        targets = torch.tensor([[0.,1.,2.,3.]])
        loss = paired_expected_regret(values, targets)
        self.assertAlmostEqual(float(loss.detach()), 1.5)
        loss.backward()
        self.assertLess(float(values.grad[0,3]), 0)
        self.assertGreater(float(values.grad[0,0]), 0)
        torch.testing.assert_close(loss, paired_expected_regret(values+50, targets+100))

    def test_ties_have_zero_regret_and_incomplete_groups_are_rejected(self):
        self.assertEqual(float(paired_expected_regret(torch.zeros(2,4), torch.ones(2,4))),0)
        with self.assertRaises(ValueError):
            paired_expected_regret(torch.zeros(1,3),torch.zeros(1,3))
