import hashlib
import json
from pathlib import Path
import unittest


class AuditV7RejectionTests(unittest.TestCase):
    def test_rejected_capture_retained_without_scored_collection(self):
        root = Path(__file__).resolve().parents[1]/"data/reference/browser-surface-rescue-audit-v7"
        status = json.loads((root/"status.json").read_text())
        capture = json.loads((root/"h1-preflop/capture.json").read_text())
        self.assertEqual(status["status"],"aborted_capture_rejected_no_scoring")
        self.assertEqual(status["images_scored"],0)
        self.assertEqual(status["retakes_performed"],0)
        self.assertFalse(capture["eligible_for_draft_scoring"])
        self.assertEqual(capture["rejection_reason"],"Card outside saved screenshot")
        self.assertEqual(capture["capture_stability"]["before"],capture["capture_stability"]["after"])
        self.assertEqual(hashlib.sha256((root/"h1-preflop"/capture["image"]).read_bytes()).hexdigest(),capture["sha256"])
        self.assertFalse((root/"collection.json").exists())
