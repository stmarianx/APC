import json
from pathlib import Path
import shutil
import tempfile
import unittest

from apc.annotator.prepare_aligned_collection import prepare
from apc.perception.evaluate_browser_collection import load_collection


DEMO = Path(__file__).resolve().parents[1]/"data/development/measured-alignment-demo-v1"


class AlignedCollectionTests(unittest.TestCase):
    def test_real_measurements_preserve_draft_status_and_card_pixels(self):
        data, frames = load_collection(DEMO/"collection.json")
        self.assertTrue(data["draft_only"])
        self.assertFalse(data["readiness_credit"])
        self.assertEqual([t["identity"] for t in frames[0]["targets"]], ["Js","Th"])
        from apc.annotator.review_browser_collection import pixel_box
        self.assertEqual(pixel_box(frames[0]["targets"][0]["box_xyxy"],(1265,712)), [519,530,581,607])

    def test_reproducible_preparation_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(DEMO/"acquisition",root/"acquisition")
            output = root/"collection.json"
            actual = prepare(root/"acquisition/capture.json", DEMO/"measurement.json",output)
            self.assertEqual(actual,json.loads((DEMO/"collection.json").read_text()))
            load_collection(output)
            with self.assertRaises(FileExistsError):
                prepare(root/"acquisition/capture.json",DEMO/"measurement.json",output)

    def test_sealed_rejected_and_tampered_sources_refused(self):
        for key,value in (("use_policy","sealed_audit_no_training"),
                          ("eligible_for_draft_scoring",False),("sha256","wrong")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shutil.copytree(DEMO/"acquisition",root/"acquisition")
                path = root/"acquisition/capture.json"
                metadata = json.loads(path.read_text())
                metadata[key]=value
                path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    prepare(path,DEMO/"measurement.json",root/"collection.json")
                self.assertFalse((root/"collection.json").exists())
