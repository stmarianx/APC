from pathlib import Path
import unittest

from apc.perception.evaluate_browser_collection import load_collection


class FastAuditV6Tests(unittest.TestCase):
    def test_complete_native_capture_schedule_and_stability_evidence(self):
        path = Path(__file__).resolve().parents[1]/"data/reference/browser-fast-text-audit-v6/collection.json"
        data,frames = load_collection(path)
        self.assertEqual(len(frames),15)
        self.assertEqual(sum(len(f["targets"]) for f in frames),60)
        self.assertTrue(data["capture_stability_required"])
        self.assertEqual(data["use_policy"],"sealed_audit_no_training")
        self.assertEqual(len({f["image"] for f in frames}),15)
        self.assertEqual({r["capture_stability"]["before"]["fixture_scene"] for r in data["frames"][-3:]},
                         {"text","lobby","controls"})
        self.assertFalse(data["readiness_credit"])
