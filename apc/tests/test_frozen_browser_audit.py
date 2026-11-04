import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apc.perception.evaluate_browser_collection import load_collection
from apc.perception.evaluate_frozen_browser_audit import ROOT, run_audit


PLAN = ROOT / "apc/perception/baselines/browser-capture-audit-v2.plan.json"
COLLECTION = ROOT / "apc/data/reference/browser-capture-audit-v2/collection.json"
GLYPH = ROOT / "apc/checkpoints/browser-symbol-rehearsal-v1.pt"
LOCATOR = ROOT / "apc/checkpoints/card-locator-hardneg-v1.pt"


class FrozenBrowserAuditTests(unittest.TestCase):
    def test_new_collection_retains_every_planned_frame_and_disallows_training(self):
        data, frames = load_collection(COLLECTION)
        plan = json.loads(PLAN.read_text())
        self.assertEqual(data["use_policy"], "sealed_audit_no_training")
        self.assertEqual([[r["hand_group"], r["street"]] for r in data["frames"]], plan["frame_schedule"])
        self.assertEqual(len(frames), 6)
        self.assertEqual(sum(len(frame["targets"]) for frame in frames), 34)
        self.assertEqual(len({target["identity"] for frame in frames for target in frame["targets"]}), 18)

    def test_audit_checks_frozen_weights_and_disjoint_session_before_evaluation(self):
        with patch("apc.perception.evaluate_frozen_browser_audit.CardReader") as reader, \
             patch("apc.perception.evaluate_frozen_browser_audit.evaluate", return_value={"full_table_readiness": False}):
            report = run_audit(PLAN, COLLECTION, GLYPH, LOCATOR)
        self.assertTrue(all(report["audit_checks"].values()))
        self.assertFalse(report["full_table_readiness"])
        self.assertTrue(reader.call_args.kwargs["surface_pairs"])

    def test_changed_checkpoint_plan_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = json.loads(PLAN.read_text())
            plan["recognizer_sha256"] = "0" * 64
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            with patch("apc.perception.evaluate_frozen_browser_audit.CardReader") as reader, self.assertRaisesRegex(ValueError, "frozen audit plan"):
                run_audit(path, COLLECTION, GLYPH, LOCATOR)
            reader.assert_not_called()

    def test_incomplete_schedule_or_reused_images_are_rejected(self):
        data, frames = load_collection(COLLECTION)
        altered = copy.deepcopy(data)
        altered["frames"].pop()
        with patch("apc.perception.evaluate_frozen_browser_audit.load_collection", return_value=(altered, frames[:-1])), \
             self.assertRaisesRegex(ValueError, "schedule"):
            run_audit(PLAN, COLLECTION, GLYPH, LOCATOR)
        with self.assertRaisesRegex(ValueError, "new capture session"):
            run_audit(PLAN, COLLECTION, GLYPH, LOCATOR, previous_collection=COLLECTION)
        previous = copy.deepcopy(data)
        previous["capture_session_id"] = "different-session-id"
        with patch("apc.perception.evaluate_frozen_browser_audit.load_collection", side_effect=[(data, frames), (previous, frames)]), \
             self.assertRaisesRegex(ValueError, "reuses development images"):
            run_audit(PLAN, COLLECTION, GLYPH, LOCATOR)


if __name__ == "__main__":
    unittest.main()
