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
                patch("apc.perception.evaluate_card_reader.CardReader") as factory:
            factory.return_value.read.return_value = {"cards": [{"box_xyxy": [.11, .1, .21, .3], "candidate": "Qs"}]}
            factory.return_value.predict_regions.return_value = {"rank": rank, "suit": suit}
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
        cards[0]["card"] = "As"
        with patch("apc.perception.evaluate_card_reader.load_split", return_value=[target]), \
                patch("apc.perception.evaluate_card_reader.CardReader") as factory:
            factory.return_value.threshold = .9
            factory.return_value.read.return_value = {"cards": cards, "elapsed_ms": 10}
            report = evaluate_reader("checkpoint", "manifest")
        self.assertEqual(report["false_positives"], 1)
        self.assertEqual(report["false_negatives"], 1)
        self.assertEqual(report["confirmed_cards_emitted"], 1)
        self.assertEqual(report["frames_all_card_candidates_exact"], 0)
        self.assertEqual(report["frames_with_high_score_error"], 1)
        self.assertEqual(report["high_score_unmatched"], 1)
        self.assertEqual(report["high_score_wrong_matched"], 0)
        self.assertEqual(report["candidate_exact_recall"], .5)
        self.assertEqual(report["high_score_end_to_end_precision"], .5)
        self.assertFalse(report["full_table_readiness"])

    def test_partial_labels_cannot_create_perfect_frame_or_spurious_error(self):
        known = {"class_name": "hero_card", "box_mask": True, "box_xyxy": [.1,.1,.2,.3],
                 "attributes": {"rank": "A", "suit": "s"}, "attribute_masks": {"rank": True, "suit": True}}
        unknown = dict(known, box_xyxy=[.4,.1,.5,.3], attribute_masks={"rank": False, "suit": True})
        target = {"image": {"path": "fixture"}, "objects": [known, unknown]}
        proposed = {"candidate": "As", "rank_score": .99, "suit_score": .99, "box_xyxy": known["box_xyxy"]}
        with patch("apc.perception.evaluate_card_reader.load_split", return_value=[target, target]), \
                patch("apc.perception.evaluate_card_reader.CardReader") as factory:
            factory.return_value.threshold = .9
            factory.return_value.read.side_effect = [
                {"cards": [proposed], "elapsed_ms": 1},
                {"cards": [proposed, dict(proposed, box_xyxy=unknown["box_xyxy"])], "elapsed_ms": 2}]
            report = evaluate_reader("checkpoint", "manifest")
        self.assertEqual(report["frames_excluded_partial_card_labels"], 2)
        self.assertEqual(report["excluded_partial_frame_proposals"], 3)
        self.assertEqual(report["frames_all_card_candidates_exact"], 0)
        self.assertEqual(report["frames_with_high_score_error"], 0)
        self.assertIsNone(report["high_score_end_to_end_precision"])
