import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image, ImageChops
from apc.annotator.project import AnnotationProject
from apc.annotator.review_capture_suggestion import render_suggestion


ROOT=Path(__file__).resolve().parents[1]/"data/development/public-label-capture-demo-v1"


class PublicLabelCaptureDemoTests(unittest.TestCase):
    def test_real_observations_and_import_remain_fingerprinted_unapproved(self):
        for key,size in (("first",[1265,712]),("second",[1265,864])):
            metadata=json.loads((ROOT/key/"capture.json").read_text())
            self.assertEqual(metadata["image_size"],size)
            self.assertEqual(hashlib.sha256((ROOT/key/metadata["image"]).read_bytes()).hexdigest(),metadata["sha256"])
            self.assertEqual(metadata["capture_stability"]["before"],metadata["capture_stability"]["after"])
            self.assertNotIn("opponent_cards",metadata["capture_stability"]["before"]["public_regions"])
        project=AnnotationProject(ROOT/"labeling-project")
        self.assertEqual(len(project.records),1)
        self.assertEqual(project.status()["verified_annotations"],0)
        record=project.records[0]
        suggestion=project.load_suggestion(record.sample_id)
        state=suggestion["suggested_visible_state"]
        self.assertEqual(state["seat_stacks_bb"],[{"seat_no":1,"stack_bb":"99.5"},{"seat_no":2,"stack_bb":"99"}])
        self.assertEqual(state["pot_bb"],"1.5")
        self.assertEqual(state["to_call_bb"],"0.5")
        self.assertEqual(state["dealer_seat"],1)
        self.assertEqual(suggestion["draft_card_labels"]["hero_cards"],["2d","6c"])
        self.assertEqual(suggestion["suggested_objects"]["action_buttons"],[])
        self.assertEqual(len(suggestion["skipped_unmeasured_public_regions"]),4)
        self.assertFalse(suggestion["readiness_counts_added"])
        self.assertIsNone(project.load_annotation(record.sample_id))
        with Image.open(ROOT/"second/frame.jpg") as original,Image.open(project.root/record.frame_path) as derivative:
            self.assertIsNone(ImageChops.difference(original.convert("RGB"),derivative.convert("RGB")).getbbox())

    def test_review_renders_five_regions_and_refuses_overwrite(self):
        project=AnnotationProject(ROOT/"labeling-project")
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"review"
            report=render_suggestion(project,project.records[0].sample_id,output)
            self.assertEqual(len(report["objects"]),5)
            self.assertEqual(report["review_status"],"pending")
            self.assertFalse(report["independent_review"])
            self.assertFalse(report["model_used"])
            with self.assertRaises(FileExistsError):
                render_suggestion(project,project.records[0].sample_id,output)

    def test_review_rejects_changed_project_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/"labeling-project",root/"project")
            project=AnnotationProject(root/"project")
            record=project.records[0]
            Image.new("RGB",(record.width,record.height),"black").save(project.root/record.frame_path)
            with self.assertRaisesRegex(ValueError,"image changed"):
                render_suggestion(project,record.sample_id,root/"review")
            self.assertFalse((root/"review").exists())
