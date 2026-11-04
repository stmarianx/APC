import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from apc.annotator.project import AnnotationProject
from apc.annotator.import_practice_capture import import_capture, suggested_objects


class CaptureImportTests(unittest.TestCase):
    def metadata(self):
        return {"source":"apc_full_hand_browser","draft_only":True,"units":"BB",
                "viewport":{"width":200,"height":100},
                "public_state":{"street":"flop","pot_bb":"6","hero_cards":["As","Kh"],"board":[]},
                "regions":[{"id":"heroCards-0","box":{"x":50,"y":20,"width":20,"height":30}}]}

    def test_import_stays_unverified_and_preserves_existing_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = AnnotationProject.create(root/"project",project_id="capture-test",
                source_kind="controlled_training_table",provider_id="apc-full-hand-browser",
                layout_id="heads-up",theme_id="test",max_seats=2,locale="en-US")
            frame, metadata = root/"frame.png", root/"metadata.json"
            Image.new("RGB",(600,300),"green").save(frame)
            metadata.write_text(json.dumps(self.metadata()))
            result = import_capture(project,frame,metadata,"session-one",1000)
            self.assertFalse(result["verified"])
            self.assertIsNone(project.load_annotation(result["sample_id"]))
            saved = project.load_suggestion(result["sample_id"])
            self.assertTrue(saved["review_required"])
            self.assertFalse(saved["auto_applied"])
            self.assertEqual(saved["suggested_objects"]["hero_cards"][0]["box"]["x"],.25)
            with self.assertRaises(ValueError):
                import_capture(project,frame,metadata,"session-one",1000)

    def test_offscreen_regions_are_not_clipped_into_false_full_labels(self):
        metadata = self.metadata()
        metadata["regions"][0]["box"]["x"] = -5
        objects, skipped = suggested_objects(metadata)
        self.assertEqual(objects["hero_cards"],[])
        self.assertEqual(skipped,["heroCards-0"])
