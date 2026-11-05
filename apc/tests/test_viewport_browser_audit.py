import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apc.perception.evaluate_browser_collection import load_collection
from apc.perception.evaluate_frozen_browser_audit import ROOT, run_audit

COLLECTION = ROOT / "apc/data/reference/browser-viewport-audit-v3/collection.json"
PLAN = ROOT / "apc/perception/baselines/browser-viewport-audit-v3.plan.json"
GLYPH = ROOT / "apc/checkpoints/glyph-font-diversity-v1.pt"
LOCATOR = ROOT / "apc/checkpoints/card-locator-hardneg-v1.pt"
PREVIOUS = ROOT / "apc/data/reference/browser-capture-audit-v2/collection.json"


class ViewportBrowserAuditTests(unittest.TestCase):
    def test_per_frame_geometry_preserves_dom_viewports_and_compact_glyphs(self):
        data, frames = load_collection(COLLECTION)
        self.assertEqual([frame["viewport"] for frame in frames], json.loads(PLAN.read_text())["frame_viewports"])
        self.assertEqual(sum(len(frame["targets"]) for frame in frames), 34)
        self.assertEqual(len({t["identity"] for f in frames for t in f["targets"]}), 20)
        self.assertEqual(frames[4]["card_size"], [44, 64])
        self.assertAlmostEqual(frames[4]["targets"][0]["box_xyxy"][0], 174.333/600)
        self.assertAlmostEqual(frames[5]["targets"][0]["box_xyxy"][1], 149.073*(1425/1440)/866)
        self.assertAlmostEqual(frames[0]["targets"][0]["box_xyxy"][0], 365.385/960)
        self.assertEqual(data["use_policy"], "sealed_audit_no_training")

    def test_complete_viewport_plan_and_prior_session_disjointness_are_checked(self):
        with patch("apc.perception.evaluate_frozen_browser_audit.CardReader"), \
             patch("apc.perception.evaluate_frozen_browser_audit.evaluate", return_value={"full_table_readiness": False}):
            report = run_audit(PLAN, COLLECTION, GLYPH, LOCATOR, additional_previous_collections=(PREVIOUS,))
        self.assertTrue(report["audit_checks"]["planned_viewports_match"])
        self.assertFalse(report["full_table_readiness"])

    def test_changed_viewport_plan_fails_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            data = json.loads(PLAN.read_text())
            data["frame_viewports"][-1] = [900, 900]
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(data))
            with patch("apc.perception.evaluate_frozen_browser_audit.CardReader") as reader, \
                 self.assertRaisesRegex(ValueError, "viewport schedule"):
                run_audit(path, COLLECTION, GLYPH, LOCATOR)
            reader.assert_not_called()

    def test_invalid_per_frame_dimensions_and_unknown_mapping_are_rejected(self):
        data = json.loads(COLLECTION.read_text())
        for field, value in (("viewport", [0, 800]), ("screenshot_mapping", "guess_from_model")):
            changed = copy.deepcopy(data)
            changed["frames"][0][field] = value
            changed["frames"][0]["image"] = str((COLLECTION.parent / changed["frames"][0]["image"]).resolve())
            # Keep the same corpus root so path validation still applies.
            with patch("pathlib.Path.read_text", return_value=json.dumps(changed)), self.assertRaises(ValueError):
                load_collection(COLLECTION)


if __name__ == "__main__":
    unittest.main()
