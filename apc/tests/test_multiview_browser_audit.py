import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from apc.perception.evaluate_browser_collection import evaluate, load_collection
from apc.perception.evaluate_frozen_browser_audit import ROOT, run_audit

PLAN = ROOT / "apc/perception/baselines/browser-multiview-audit-v4.plan.json"
COLLECTION = ROOT / "apc/data/reference/browser-multiview-audit-v4/collection.json"
GLYPH = ROOT / "apc/checkpoints/glyph-font-diversity-v1.pt"
LOCATOR = ROOT / "apc/checkpoints/card-locator-hardneg-v1.pt"


class MultiviewAuditTests(unittest.TestCase):
    def test_complete_schedule_includes_negatives_and_frozen_inference_settings(self):
        data, frames = load_collection(COLLECTION)
        self.assertEqual(len(frames), 9)
        self.assertEqual(sum(len(f["targets"]) for f in frames), 27)
        self.assertEqual(sum(not f["targets"] for f in frames), 3)
        with patch("apc.perception.evaluate_frozen_browser_audit.CardReader") as reader, \
             patch("apc.perception.evaluate_frozen_browser_audit.evaluate", return_value={"full_table_readiness": False}):
            report = run_audit(PLAN, COLLECTION, GLYPH, LOCATOR)
        for option in ("adaptive_surfaces", "deduplicate_surfaces", "tiled_views"):
            self.assertTrue(reader.call_args.kwargs[option])
        self.assertTrue(report["audit_checks"]["inference_sources_match"])
        self.assertFalse(report["full_table_readiness"])

    def test_empty_frames_score_false_proposals_without_oracle_stack(self):
        data, frames = load_collection(COLLECTION)
        reader = Mock()
        reader.read.return_value = {"cards": [{"box_xyxy": [.1, .1, .2, .3], "card": None}]}
        with patch("apc.perception.evaluate_browser_collection.load_collection", return_value=(data, frames[-3:])):
            report = evaluate(COLLECTION, reader)
        self.assertEqual(report["totals"]["targets"], 0)
        self.assertEqual(report["totals"]["false_positives"], 3)
        self.assertEqual(report["totals"]["oracle_glyph_exact"], 0)
        self.assertEqual(report["empty_frames"], 3)
        self.assertEqual(report["hand_groups"], 0)
        reader.model.assert_not_called()

    def test_changed_source_or_unknown_inference_options_fail_before_model_loading(self):
        for corruption in ("source", "option", "missing_sources", "incomplete_sources"):
            plan = json.loads(PLAN.read_text())
            if corruption == "source":
                plan["inference_source_sha256"]["apc/perception/infer_cards.py"] = "0" * 64
            elif corruption == "option":
                plan["inference_options"]["confidence_threshold"] = .1
            elif corruption == "incomplete_sources":
                plan["inference_source_sha256"].pop("apc/perception/card_glyphs.py")
            else:
                plan.pop("inference_source_sha256")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                with patch("apc.perception.evaluate_frozen_browser_audit.CardReader") as reader, self.assertRaises(ValueError):
                    run_audit(path, COLLECTION, GLYPH, LOCATOR)
                reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
