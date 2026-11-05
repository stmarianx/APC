from pathlib import Path
import unittest

from apc.perception.diagnose_viewport_sync import summarize


class ViewportSyncDiagnosticTests(unittest.TestCase):
    def test_preserved_six_capture_experiment(self):
        root = Path(__file__).resolve().parents[1] / "data/reference/viewport-sync-diagnostic-v1"
        result = summarize(root)
        self.assertEqual(result["dimension_disagreement_pairs"], 3)
        self.assertEqual(len(result["pairs"]), 3)
        for pair in result["pairs"]:
            self.assertTrue(pair["same_dom_geometry"])
            self.assertEqual(len(pair["captures"]), 2)
            self.assertTrue(all(c["endpoint_geometry_equal"] for c in pair["captures"]))
        wide = result["pairs"][2]["captures"][0]
        self.assertEqual(wide["image_size"], [600, 875])
        self.assertTrue(wide["bounds_check_passed"])
        self.assertFalse(result["pixel_alignment_proven"])
        self.assertFalse(result["readiness_credit"])
        self.assertFalse(result["model_inference_performed"])
