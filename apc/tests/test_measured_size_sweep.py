import hashlib
import json
from pathlib import Path
import unittest

from apc.perception.diagnose_viewport_sync import summarize
from apc.perception.evaluate_browser_collection import load_collection


ROOT = Path(__file__).resolve().parents[1]/"data/development/measured-size-sweep-v1"


class MeasuredSizeSweepTests(unittest.TestCase):
    def test_all_four_originals_retained_with_resize_disagreement(self):
        report = summarize(ROOT)
        self.assertEqual(report["dimension_disagreement_pairs"],2)
        for pair in report["pairs"]:
            self.assertTrue(pair["same_dom_geometry"])
            self.assertEqual(len(pair["captures"]),2)
        self.assertFalse(report["readiness_credit"])
        first_wide = report["pairs"][1]["captures"][0]
        self.assertTrue(first_wide["bounds_check_passed"])
        previous = ROOT.parents[1]/"reference/viewport-sync-diagnostic-v1/w1440-first/frame.jpg"
        self.assertEqual(first_wide["sha256"],hashlib.sha256(previous.read_bytes()).hexdigest())
        for width in (600,1440):
            self.assertFalse((ROOT/f"w{width}-first/collection.json").exists())

    def test_second_capture_overlays_remain_draft_and_share_one_session(self):
        for width,card_size in ((600,[44,64]),(1440,[62,78])):
            folder = ROOT/f"w{width}-second"
            data,frames = load_collection(folder/"collection.json")
            self.assertEqual(data["capture_session_id"],"measured-size-sweep-v1")
            self.assertEqual(data["card_size"],card_size)
            self.assertEqual([t["identity"] for t in frames[0]["targets"]],["9s","Ts"])
            self.assertEqual(frames[0]["screenshot_mapping"],"measured_pixel_alignment_v1")
            self.assertFalse(data["readiness_credit"])
            review = json.loads((folder/"review/review.json").read_text())
            self.assertFalse(review["independently_reviewed"])
            self.assertEqual(review["frames"][0]["review_status"],"pending")
