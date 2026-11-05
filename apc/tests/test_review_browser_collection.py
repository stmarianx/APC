import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from apc.annotator.review_browser_collection import pixel_box, render_review

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/reference/capture-stability-demo-v1/collection.json"


class BrowserReviewTests(unittest.TestCase):
    def test_pixel_mapping_uses_image_axes(self):
        self.assertEqual(pixel_box([.1,.2,.3,.4], (100,200)), [10,40,30,80])

    def test_real_capture_overlay_preserves_source_and_pending_review(self):
        before = hashlib.sha256(DEMO.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            report = render_review(DEMO, output)
            self.assertFalse(report["model_used"])
            self.assertFalse(report["labels_changed"])
            self.assertFalse(report["readiness_credit"])
            row = report["frames"][0]
            self.assertEqual(row["review_status"], "pending")
            self.assertEqual(row["targets"][0]["card_xyxy"], [519,530,581,607])
            with Image.open(output / row["overlay"]) as image:
                self.assertEqual(image.size, (1265,712))
                self.assertEqual(image.getpixel((519,530)), (255,223,0))
            with self.assertRaises(FileExistsError):
                render_review(DEMO, output)
        self.assertEqual(hashlib.sha256(DEMO.read_bytes()).hexdigest(), before)

    def test_source_directory_cannot_be_output(self):
        with self.assertRaises(ValueError):
            render_review(DEMO, DEMO.parent)

    def test_geometry_revision_only_changes_documented_wide_preflop_y(self):
        folder = ROOT / "data/reference/browser-routed-audit-v5"
        original = json.loads((folder / "collection.json").read_text())
        corrected = json.loads((folder / "collection.geometry-v2.json").read_text())
        self.assertFalse(corrected.pop("annotation_revision")["independently_reviewed"])
        wide = corrected["frames"][4]
        for index, card in enumerate(wide["cards"]):
            self.assertAlmostEqual(card[3], 457*1280/1265)
            card[3] = original["frames"][4]["cards"][index][3]
        self.assertEqual(corrected, original)
