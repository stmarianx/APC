import copy
import unittest
import json
from pathlib import Path
import shutil
import tempfile

from apc.annotator.public_table_labels import public_table_suggestions
from apc.tests.test_pixel_alignment import evidence
from apc.perception.pixel_alignment import PixelAlignment
from apc.annotator.project import AnnotationProject
from apc.annotator.import_browser_acquisition import import_acquisition


def public_fixture():
    return {"source":"owned_practice_visible_dom_v1","units":"BB","street":"preflop","to_call_bb":"0.5",
        "seats":[{"seat_no":1,"player_name":"Hero","stack_bb":"99.5","is_hero":True,"has_dealer_button":True,
                  "box_xywh":[100,500,200,100]},
                 {"seat_no":2,"player_name":"Opponent","stack_bb":"99","is_hero":False,"has_dealer_button":False,
                  "box_xywh":[100,100,200,100]}],
        "pot":{"amount_bb":"1.5","text":"1.5","box_xywh":[400,300,50,30]},
        "action_buttons":[{"action":"call","enabled":True,"text":"call","box_xywh":[400,700,80,40]}]}


class PublicTableLabelTests(unittest.TestCase):
    def test_public_labels_reach_import_suggestion_without_training_approval(self):
        source=Path(__file__).resolve().parents[1]/"data/development/measured-alignment-demo-v1"
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(source/"acquisition",root/"acquisition")
            manifest=root/"acquisition/capture.json"
            metadata=json.loads(manifest.read_text())
            public=public_fixture()
            public["seats"][0]["box_xywh"]=[400,500,200,100]
            public["seats"][1]["box_xywh"]=[400,250,200,100]
            public["pot"]["box_xywh"]=[600,400,50,30]
            for end in ("before","after"):
                metadata["capture_stability"][end]["public_regions"]=public
            manifest.write_text(json.dumps(metadata))
            project=AnnotationProject.create(root/"project",project_id="public-label-test",
                source_kind="controlled_training_table",provider_id="apc-full-hand-browser",
                layout_id="heads-up",theme_id="fixture",max_seats=2,locale="en-US")
            imported=import_acquisition(project,manifest,"fixture-session",0,pixel_alignment=source/"measurement.json")
            suggestion=project.load_suggestion(imported["sample_id"])
            self.assertEqual(suggestion["suggested_visible_state"]["pot_bb"],"1.5")
            self.assertEqual(len(suggestion["suggested_objects"]["seats"]),2)
            self.assertEqual(suggestion["skipped_unmeasured_public_regions"],["action:call"])
            self.assertFalse(suggestion["readiness_counts_added"])
            self.assertIsNone(project.load_annotation(imported["sample_id"]))

    def alignment(self):
        return PixelAlignment(evidence(),image_sha256="fixture",image_size=[600,450],viewport=[1000,800])

    def test_visible_fields_mapped_without_private_or_invented_hand_state(self):
        public=public_fixture()
        original=copy.deepcopy(public)
        objects,state,skipped=public_table_suggestions(public,self.alignment())
        self.assertEqual(public,original)
        self.assertEqual(skipped,[])
        self.assertEqual(state["pot_bb"],"1.5")
        self.assertEqual(state["hero_seat"],1)
        self.assertEqual(state["legal_actions"],["call"])
        self.assertNotIn("hand_id",state)
        self.assertNotIn("opponent_cards",state)
        self.assertEqual(objects["seats"][0]["visibility"],"uncertain")
        self.assertAlmostEqual(objects["pot"]["box"]["x"],220/600)

    def test_unmeasured_controls_are_skipped_not_extrapolated(self):
        public=public_fixture()
        public["action_buttons"][0]["box_xywh"]=[400,770,80,25]
        objects,state,skipped=public_table_suggestions(public,self.alignment())
        self.assertEqual(objects["action_buttons"],[])
        self.assertEqual(skipped,["action:call"])
        self.assertEqual(state["legal_actions"],["call"])

    def test_invalid_evidence_fails_closed(self):
        for mutate in (lambda p:p.update(units="USD"),lambda p:p["pot"].update(amount_bb="NaN"),
                       lambda p:p["seats"][1].update(seat_no=1),lambda p:p["seats"][0].update(is_hero="true"),
                       lambda p:p["action_buttons"][0].update(action="hidden")):
            public=public_fixture(); mutate(public)
            with self.assertRaises(ValueError):
                public_table_suggestions(public,self.alignment())
        with self.assertRaises(ValueError):
            public_table_suggestions(public_fixture(),None)
