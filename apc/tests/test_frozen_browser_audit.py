import copy
import hashlib
import json
import tempfile
import subprocess
import sys
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
    def test_custom_audit_cli_rejects_partial_or_ambiguous_inputs(self):
        cases = (["--plan", "missing.json"], ["--verifier-checkpoint", "missing.pt"],
                 ["--plan", "p", "--collection", "c", "--glyph-checkpoint", "g",
                  "--locator-checkpoint", "l", "--multiview-v4"])
        for args in cases:
            result = subprocess.run([sys.executable, "-m", "apc.perception.evaluate_frozen_browser_audit", *args],
                cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("error:", result.stderr)

    def test_planned_routed_reader_receives_both_frozen_experts(self):
        primary = ROOT / "apc/checkpoints/card-face-verifier-appearance-v4.pt"
        bounded = ROOT / "apc/checkpoints/card-face-verifier-v1.pt"
        plan = json.loads(PLAN.read_text())
        plan.update(verifier_sha256=hashlib.sha256(primary.read_bytes()).hexdigest(),
                    bounded_verifier_sha256=hashlib.sha256(bounded.read_bytes()).hexdigest(),
                    inference_options={"adaptive_surfaces": True, "tiled_views": True, "deduplicate_surfaces": True})
        names = ("infer_cards.py", "infer_card_locator.py", "card_proposals.py", "card_locator.py",
                 "card_glyphs.py", "card_verifier.py", "verified_card_reader.py")
        plan["inference_source_sha256"] = {"apc/perception/"+name:
            hashlib.sha256((ROOT/"apc/perception"/name).read_bytes()).hexdigest() for name in names}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"plan.json"
            path.write_text(json.dumps(plan))
            with patch("apc.perception.verified_card_reader.VerifiedCardReader") as reader, \
                 patch("apc.perception.evaluate_frozen_browser_audit.evaluate", return_value={"full_table_readiness": False}):
                reader.return_value.verifier_metadata = {"research_only": True}
                result = run_audit(path, COLLECTION, GLYPH, LOCATOR,
                    verifier_checkpoint=primary, bounded_verifier_checkpoint=bounded)
        self.assertEqual(reader.call_args.kwargs["verifier_checkpoint"], primary)
        self.assertEqual(reader.call_args.kwargs["bounded_verifier_checkpoint"], bounded)
        self.assertTrue(result["audit_checks_verifier_hashes_match"])
        self.assertFalse(result["full_table_readiness"])

    def test_wide_recovery_audit_requires_frozen_wrapper_and_stability(self):
        primary = ROOT / "apc/checkpoints/card-face-verifier-appearance-v4.pt"
        plan = json.loads(PLAN.read_text())
        plan.update(verifier_sha256=hashlib.sha256(primary.read_bytes()).hexdigest(),
                    inference_options={"adaptive_surfaces":True},wide_surface_recovery=True,
                    fast_surface_components=True,fast_locator_refinement=True)
        names = ("infer_cards.py","infer_card_locator.py","card_proposals.py","card_locator.py",
                 "card_glyphs.py","card_verifier.py","verified_card_reader.py","bright_surface_candidate.py",
                 "fast_components.py","fast_refinement.py")
        plan["inference_source_sha256"] = {"apc/perception/"+name:
            hashlib.sha256((ROOT/"apc/perception"/name).read_bytes()).hexdigest() for name in names}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"plan.json"
            path.write_text(json.dumps(plan))
            with patch("apc.perception.bright_surface_candidate.BrightSurfaceReader") as reader, \
                 patch("apc.perception.evaluate_frozen_browser_audit.evaluate",return_value={"full_table_readiness":False}):
                reader.return_value.verifier_metadata = {"research_only":True}
                result = run_audit(path,COLLECTION,GLYPH,LOCATOR,verifier_checkpoint=primary)
                self.assertTrue(reader.call_args.kwargs["wide_search"])
                self.assertTrue(reader.call_args.kwargs["fast_components"])
                self.assertTrue(reader.call_args.kwargs["fast_refinement"])
                self.assertTrue(result["frozen_wide_surface_recovery"])
            del plan["inference_source_sha256"]["apc/perception/bright_surface_candidate.py"]
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,"fingerprints"):
                run_audit(path,COLLECTION,GLYPH,LOCATOR,verifier_checkpoint=primary)
            plan["inference_source_sha256"]["apc/perception/bright_surface_candidate.py"] = hashlib.sha256(
                (ROOT/"apc/perception/bright_surface_candidate.py").read_bytes()).hexdigest()
            plan["capture_stability_required"] = True
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,"stability"):
                run_audit(path,COLLECTION,GLYPH,LOCATOR,verifier_checkpoint=primary)

    def test_unplanned_verifier_and_unpaired_bounded_expert_are_rejected(self):
        verifier = ROOT / "apc/checkpoints/card-face-verifier-v1.pt"
        with patch("apc.perception.evaluate_frozen_browser_audit.CardReader") as reader:
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                run_audit(PLAN, COLLECTION, GLYPH, LOCATOR, verifier_checkpoint=verifier)
            with self.assertRaisesRegex(ValueError, "primary verifier"):
                run_audit(PLAN, COLLECTION, GLYPH, LOCATOR, bounded_verifier_checkpoint=verifier)
            reader.assert_not_called()

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
