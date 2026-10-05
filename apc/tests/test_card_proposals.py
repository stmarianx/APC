import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from apc.perception.card_proposals import propose_cards, match_proposals
from apc.synthetic.render_table import render_frame, THEMES, LAYOUTS


class CardProposalTests(unittest.TestCase):
    def test_pixel_proposals_find_rendered_cards_without_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            frame = render_frame(path, rng=random.Random(11), session_id="session", sequence_index=0,
                                 seats=6, layout_id="six-max", theme=THEMES[0], street="flop")
            proposals = propose_cards(path)
            targets = []
            for card in frame.annotation["objects"]["hero_cards"] + frame.annotation["objects"]["board_cards"]:
                b = card["box"]
                targets.append({"box_xyxy": [b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]]})
            report = match_proposals(proposals, targets, .8)
            self.assertEqual(report["true_positives"], 5)
            self.assertEqual(report["false_negatives"], 0)

    def test_blank_frame_has_no_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            Image.new("RGB", (640, 360), "white").save(path)
            self.assertEqual(propose_cards(path), [])

    def test_proposals_across_layouts_and_themes(self):
        with tempfile.TemporaryDirectory() as directory:
            for seats, layout in LAYOUTS:
                for theme in THEMES:
                    path = Path(directory) / "frame.png"
                    frame = render_frame(path, rng=random.Random(29), session_id="session", sequence_index=0,
                                         seats=seats, layout_id=layout, theme=theme, street="river")
                    proposals = propose_cards(path)
                    self.assertEqual(len(proposals), 7, (layout, theme["id"]))

    def test_duplicate_detections_do_not_duplicate_credit(self):
        box = {"box_xyxy": [.1, .1, .2, .3]}
        report = match_proposals([box, box], [box])
        self.assertEqual(report["true_positives"], 1)
        self.assertEqual(report["false_positives"], 1)
