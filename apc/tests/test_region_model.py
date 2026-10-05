import unittest

import torch

from apc.perception.region_model import RegionRecognizer, region_supervision, recognition_loss
from apc.perception.spatial_targets import encode_annotation
from apc.tests.test_validate_dataset import annotation


class RegionModelTests(unittest.TestCase):
    def test_training_step_and_masked_identity(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        item = annotation("session", "sample", "frame.png", "a" * 64)
        for card in item["objects"]["hero_cards"] + item["objects"]["board_cards"]:
            card["visibility"] = "uncertain"
        regions, labels = region_supervision([encode_annotation(item)])
        self.assertTrue((labels["rank"] == -100).all())
        model = RegionRecognizer(crop_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        before = model.heads["class"].weight.detach().clone()
        output = model(torch.rand(1, 3, 64, 96), regions)
        loss, parts = recognition_loss(output, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["rank"].item(), 0)
        loss.backward()
        self.assertEqual(model.heads["rank"].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.heads["class"].weight))

    def test_region_bounds_and_batch_indices(self):
        model = RegionRecognizer(16)
        image = torch.rand(1, 3, 32, 32)
        for region in ([1, 0, 0, 1, 1], [.5, 0, 0, 1, 1], [0, 0, 0, 2, 1], [0, .5, 0, .4, 1]):
            with self.assertRaises(ValueError):
                model(image, torch.tensor([region], dtype=torch.float32))

    def test_grouped_sampling_preserves_interleaved_region_order(self):
        torch.set_num_threads(1)
        model = RegionRecognizer(16).eval()
        images = torch.rand(2, 3, 32, 48)
        regions = torch.tensor([[1, .1, .2, .8, .9], [0, 0, 0, 1, 1], [1, .3, .1, .7, .8]])
        with torch.no_grad():
            grouped = model(images, regions)
            singles = [model(images, region.unsqueeze(0)) for region in regions]
        for head in grouped:
            torch.testing.assert_close(grouped[head], torch.cat([item[head] for item in singles]), atol=1e-6, rtol=1e-5)
