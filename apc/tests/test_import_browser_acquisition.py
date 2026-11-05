import copy
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import unittest

from PIL import Image

from apc.annotator.import_browser_acquisition import import_acquisition
from apc.annotator.project import AnnotationProject
from apc.capture.browser_frame import capture_browser_frame, save_browser_frame


class BrowserAcquisitionImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = AnnotationProject.create(self.root / "project", project_id="stable-import",
            source_kind="controlled_training_table",provider_id="apc-full-hand-browser",
            layout_id="heads-up",theme_id="test",max_seats=2,locale="en-US")
        self.folder = self.root / "capture"
        self.folder.mkdir()
        Image.new("RGB",(790,850),"green").save(self.folder / "frame.png")
        digest = hashlib.sha256((self.folder / "frame.png").read_bytes()).hexdigest()
        snap = {"viewport":[800,900],"scroll":[0,0],"cards":[["hero","As",100,200,62,78]]}
        self.data = {"schema_version":"apc_browser_frame_acquisition_v1", "image":"frame.png",
            "image_size":[790,850],"sha256":digest,"cards":[["hero","As",100,200]],
            "eligible_for_draft_scoring":True,"rejection_reason":None,"readiness_credit":False,
            "capture_stability":{"schema_version":"apc_capture_stability_v1",
                "image_sha256":digest,"image_size":[790,850],"before":snap,"after":copy.deepcopy(snap)}}

    def run_import(self):
        manifest = self.folder / "capture.json"
        manifest.write_text(json.dumps(self.data),encoding="utf-8")
        return import_acquisition(self.project,manifest,"session-one",0)

    def test_stable_frame_imported_as_suggestion_only_with_crop_scale(self):
        result = self.run_import()
        self.assertFalse(result["verified"])
        self.assertIsNone(self.project.load_annotation(result["sample_id"]))
        saved = self.project.load_suggestion(result["sample_id"])
        self.assertTrue(saved["review_required"])
        self.assertFalse(saved["auto_applied"])
        self.assertFalse(saved["readiness_counts_added"])
        self.assertEqual(saved["draft_card_labels"]["hero_cards"],["As"])
        box = saved["suggested_objects"]["hero_cards"][0]["box"]
        self.assertAlmostEqual(box["y"],200/(850*800/790))
        with self.assertRaisesRegex(ValueError,"preserved"):
            self.run_import()

    def test_rejects_tampering_rejection_audit_and_path_before_import(self):
        original = copy.deepcopy(self.data)
        for key, value in (("sha256","wrong"),("image_size",[800,900]),
                           ("eligible_for_draft_scoring",False),("use_policy","sealed_audit_no_training"),
                           ("image","../frame.png")):
            with self.subTest(key=key):
                self.data = copy.deepcopy(original)
                self.data[key] = value
                with self.assertRaises(ValueError):
                    self.run_import()
                self.assertEqual(len(self.project.records),0)

    def test_forged_eligibility_does_not_bypass_scroll_recheck(self):
        self.data["capture_stability"]["after"]["scroll"] = [0,80]
        with self.assertRaisesRegex(ValueError,"moved"):
            self.run_import()
        self.assertEqual(len(self.project.records),0)

    def test_geometry_outside_saved_crop_rejected(self):
        self.data["cards"][0][3] = 820
        for name in ("before","after"):
            self.data["capture_stability"][name]["cards"][0][3] = 820
        with self.assertRaisesRegex(ValueError,"outside saved"):
            self.run_import()
        self.assertEqual(len(self.project.records),0)

    def test_collector_saved_manifest_imports_without_manual_conversion(self):
        async def observe():
            return self.data["capture_stability"]["before"]
        async def screenshot():
            return (self.folder / "frame.png").read_bytes()
        result = asyncio.run(capture_browser_frame(observe,screenshot))
        manifest = save_browser_frame(result,self.root / "saved-acquisition")
        imported = import_acquisition(self.project,manifest,"session-one",0)
        suggestion = self.project.load_suggestion(imported["sample_id"])
        self.assertEqual(suggestion["draft_card_labels"]["hero_cards"],["As"])
        self.assertEqual(suggestion["capture_import_version"],"stable_browser_acquisition_v1")
        self.assertFalse(imported["verified"])

    @unittest.skipUnless(shutil.which("node"), "Node required for cross-runtime integration")
    def test_javascript_persistence_to_python_import_preserves_jpeg(self):
        jpeg = self.folder / "source.jpg"
        Image.new("RGB",(790,850),"blue").save(jpeg,format="JPEG")
        module = (Path(__file__).resolve().parents[1] / "capture/save_browser_acquisition.mjs").as_uri()
        destination = self.root / "js-saved"
        script = """
import {readFile} from 'node:fs/promises';
const [modulePath,image,folder,snapshotJSON] = process.argv.slice(1);
const {saveBrowserAcquisition} = await import(modulePath);
const snapshot = JSON.parse(snapshotJSON);
await saveBrowserAcquisition({png:await readFile(image),before:snapshot,after:structuredClone(snapshot)},folder);
"""
        subprocess.run([shutil.which("node"),"--input-type=module","-e",script,module,str(jpeg),
            str(destination),json.dumps(self.data["capture_stability"]["before"])],check=True,
            capture_output=True,text=True,timeout=30)
        manifest = destination / "capture.json"
        saved = json.loads(manifest.read_text())
        self.assertEqual(saved["image_format"],"JPEG")
        self.assertEqual((destination / "frame.jpg").read_bytes(),jpeg.read_bytes())
        imported = import_acquisition(self.project,manifest,"session-one",0)
        self.assertFalse(imported["verified"])
        suggestion = self.project.load_suggestion(imported["sample_id"])
        self.assertEqual(suggestion["draft_card_labels"]["hero_cards"],["As"])
        self.assertEqual(suggestion["image_derivation"],"decoded_rgb_lossless_png")
        self.assertEqual(suggestion["source_image_sha256"],hashlib.sha256(jpeg.read_bytes()).hexdigest())
        self.assertEqual((destination / "frame.jpg").read_bytes(),jpeg.read_bytes())


if __name__ == "__main__":
    unittest.main()
