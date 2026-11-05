import json
import tempfile
import shutil
import unittest
from pathlib import Path

from apc.perception.evaluate_browser_collection import load_collection


COLLECTION = Path(__file__).resolve().parents[1] / "data/development/browser-domain-bridge-v1/collection.json"


class BrowserCollectionTests(unittest.TestCase):
    def test_measured_pixel_mapping_is_used_for_cards_and_glyphs(self):
        source = Path(__file__).resolve().parents[1]/"data/reference/capture-stability-demo-v1"
        data = json.loads((source/"collection.json").read_text())
        row = data["frames"][0]
        data["frames"] = [row]
        row["image"] = "frame.jpg"
        from PIL import Image
        with Image.open(source/"frame.png") as image:
            iw, ih = image.size
        vw, vh = row.get("viewport", data["viewport"])
        # Controlled test correspondences, not real capture approval.
        scale = min(iw/vw, ih/vh) * .9
        row["screenshot_mapping"] = "measured_pixel_alignment_v1"
        row["pixel_alignment"] = {"schema_version": "apc_pixel_alignment_v1",
            "image_sha256": row["sha256"], "image_size": [iw, ih], "viewport": [vw, vh],
            "measurement_method": "manual_visible_landmarks",
            "anchors": [{"description": str(i), "dom_xy": [x, y],
                         "pixel_xy": [x*scale+1, y*scale+1]}
                        for i, (x, y) in enumerate(((0,0),(vw,0),(0,vh),(vw,vh)))]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(source/"frame.png", root/"frame.jpg")
            path = root/"collection.json"
            path.write_text(json.dumps(data))
            _, frames = load_collection(path)
            x, y = row["cards"][0][2:]
            self.assertAlmostEqual(frames[0]["targets"][0]["box_xyxy"][0], (x*scale+1)/iw)
            self.assertAlmostEqual(frames[0]["targets"][0]["box_xyxy"][1], (y*scale+1)/ih)
            self.assertIn("rank", frames[0]["targets"][0]["regions"])
            row.pop("pixel_alignment")
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "alignment evidence"):
                load_collection(path)

    def test_native_jpeg_with_stability_evidence_loads_without_transcoding(self):
        source = Path(__file__).resolve().parents[1]/"data/reference/capture-stability-demo-v1"
        data = json.loads((source/"collection.json").read_text())
        data["frames"][0]["image"] = "frame.jpg"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(source/"frame.png",root/"frame.jpg")  # Original bytes are JPEG despite legacy suffix.
            manifest = root/"collection.json"
            manifest.write_text(json.dumps(data))
            _,frames = load_collection(manifest)
            self.assertEqual(frames[0]["image"].suffix,".jpg")
            self.assertEqual([t["identity"] for t in frames[0]["targets"]],["4h","2s"])

    def test_captured_development_collection_is_one_session_not_three(self):
        data, frames = load_collection(COLLECTION)
        self.assertTrue(data["draft_only"])
        self.assertFalse(data["readiness_credit"])
        self.assertEqual(len(frames), 6)
        self.assertEqual(len({frame["hand_group"] for frame in frames}), 3)
        self.assertEqual(sum(len(frame["targets"]) for frame in frames), 34)
        self.assertEqual(len({target["identity"] for frame in frames for target in frame["targets"]}), 19)
        # Coordinates use DOM viewport units, not the differently sized screenshot.
        first = frames[0]["targets"][0]
        self.assertAlmostEqual(first["box_xyxy"][0], 525.385 / 1280)
        self.assertAlmostEqual(first["box_xyxy"][1], 536.406 / 720)

    def test_collection_cannot_claim_readiness_or_escape_directory(self):
        for corruption in ("readiness", "path"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                data = json.loads(COLLECTION.read_text())
                if corruption == "readiness":
                    data["readiness_credit"] = True
                else:
                    data["frames"][0]["image"] = "../outside.png"
                path = Path(directory) / "collection.json"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_collection(path)


if __name__ == "__main__":
    unittest.main()
