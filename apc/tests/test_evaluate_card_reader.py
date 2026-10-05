import unittest
from unittest.mock import patch

import torch

from apc.perception.evaluate_card_reader import evaluate_reader, compare_crops


class ReaderEvaluationTests(unittest.TestCase):
    def test_paired_crop_diagnostic_counts_regression(self):
        target = {"image": {"path": "fixture", "width": 100, "height": 100}, "objects": [
            {"class_name": "hero_card", "box_xyxy": [.1, .1, .2, .3],
             "attribute_masks": {"rank": True, "suit": True}, "attributes": {"rank": "A", "suit": "s"}}]}
        rank = torch.zeros(1, 13)
        rank[0, 12] = 1
        suit = torch.zeros(1, 4)
        suit[0, 3] = 1
        with patch("apc.perception.evaluate_card_reader.load_split", return_value=[target]), \
                patch("apc.perception.evaluate_card_reader.image_tensor", return_value=torch.zeros(1, 3, 10, 10)), \
                patch("apc.perception.evaluate_card_reader.CardReader") as factory:
            factory.return_value.read.return_value = {"cards": [{"box_xyxy": [.11, .1, .21, .3], "candidate": "Qs"}]}
            factory.return_value.model.return_value = {"rank": rank, "suit": suit}
            report = compare_crops("checkpoint", "manifest")
        self.assertEqual(report["exact_correct_proposed_wrong"], 1)
        self.assertEqual(report["candidate_disagreements"], 1)
        self.assertAlmostEqual(report["mean_max_edge_shift_pixels"], 1)

    def test_false_proposals_count_against_precision(self):
        objects = [{"class_name": "hero_card", "box_xyxy": box, "box_mask": True,
                    "attribute_masks": {"rank": True, "suit": True},
                    "attributes": {"rank": "A", "suit": "s"}}
                   for box in ([.1, .1, .2, .3], [.5, .5, .6, .7])]
        target = {"image": {"path": "fixture.png"}, "objects": objects}
        cards = [{"box_xyxy": box, "rank_score": .99, "suit_score": .99, "candidate": "As"}
                 for box in ([.1, .1, .2, .3], [.8, .1, .9, .3])]
        with patch("apc.perception.evaluate_card_reader.load_split", return_value=[target]), \
                patch("apc.perception.evaluate_card_reader.CardReader") as factory:
            factory.return_value.threshold = .9
            factory.return_value.read.return_value = {"cards": cards, "elapsed_ms": 10}
            report = evaluate_reader("checkpoint", "manifest")
        self.assertEqual(report["false_positives"], 1)
        self.assertEqual(report["false_negatives"], 1)
        self.assertEqual(report["candidate_exact_recall"], .5)
        self.assertEqual(report["high_score_end_to_end_precision"], .5)
        self.assertFalse(report["full_table_readiness"])
