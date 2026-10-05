import unittest

from apc.perception.spatial_targets import encode_annotation
from apc.tests.test_validate_dataset import annotation


class SpatialTargetTests(unittest.TestCase):
    def fixture(self):
        return annotation("session", "sample", "frame.png", "a" * 64)

    def test_boxes_and_seat_attributes(self):
        result = encode_annotation(self.fixture())
        self.assertEqual(len(result["objects"]), 11)
        seat = result["objects"][1]
        self.assertEqual(seat["box_xyxy"], [0.1, 0.1, 0.2, 0.2])
        self.assertEqual(seat["attributes"]["stack_bb"], "97")
        self.assertTrue(seat["attribute_masks"]["has_dealer_button"])
        self.assertFalse(result["absence_supervised"])

    def test_uncertain_card_never_supervises_identity(self):
        item = self.fixture()
        item["objects"]["hero_cards"][0]["visibility"] = "partial"
        card = encode_annotation(item)["objects"][3]
        self.assertTrue(card["box_mask"])
        self.assertFalse(card["attribute_masks"]["rank"])
        self.assertFalse(card["attribute_masks"]["suit"])

    def test_occluded_and_unknown_masks(self):
        item = self.fixture()
        item["objects"]["hero_cards"][0].update(rank="unknown", suit="unknown", visibility="occluded")
        card = encode_annotation(item)["objects"][3]
        self.assertFalse(card["box_mask"])
        self.assertFalse(card["attribute_masks"]["rank"])

    def test_reject_unverified_audit_and_nonfinite_boxes(self):
        for modification in ("unverified", "audit", "nan"):
            item = self.fixture()
            if modification == "unverified":
                item["provenance"]["verified"] = False
            elif modification == "audit":
                item["use_policy"] = "frozen_out_of_domain_audit_only_not_training_or_gate_count"
            else:
                item["objects"]["table"]["x"] = float("nan")
            with self.assertRaises(ValueError):
                encode_annotation(item)
