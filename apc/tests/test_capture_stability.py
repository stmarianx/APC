import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from apc.perception.capture_stability import validate_capture_stability


class CaptureStabilityTests(unittest.TestCase):
    def setUp(self):
        self.cards = [["hero", "Ac", 525.0, 536.0]]
        snapshot = {"viewport": [1280, 900], "scroll": [0, 0],
                    "cards": [["hero", "Ac", 525.0, 536.0, 62, 78]]}
        self.evidence = {"schema_version": "apc_capture_stability_v1",
                         "image_sha256": "abc", "image_size": [1265, 864],
                         "before": snapshot, "after": copy.deepcopy(snapshot)}

    def validate(self, evidence):
        validate_capture_stability(evidence, image_size=(1265, 864),
            image_sha256="abc", viewport=[1280, 900], cards=self.cards)

    def test_stable_capture(self):
        self.validate(self.evidence)

    def test_scroll_change_is_rejected_even_with_same_card_boxes(self):
        self.evidence["after"]["scroll"][1] = 80
        with self.assertRaisesRegex(ValueError, "moved"):
            self.validate(self.evidence)

    def test_recovered_labels_cannot_replace_capture_geometry(self):
        self.cards[0][3] = 457
        with self.assertRaisesRegex(ValueError, "label geometry"):
            self.validate(self.evidence)

    def test_binding_and_missing_snapshot(self):
        for field, value in (("image_sha256", "other"), ("image_size", [1280, 900]),
                             ("after", None), ("before", {})):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.evidence)
                bad[field] = value
                with self.assertRaises(ValueError):
                    self.validate(bad)

    def test_nonfinite_and_boolean_geometry_rejected(self):
        for value in (float("nan"), float("inf"), True):
            bad = copy.deepcopy(self.evidence)
            bad["before"]["cards"][0][2] = value
            with self.assertRaises(ValueError):
                self.validate(bad)

    def test_negative_frame_still_requires_stable_scroll(self):
        self.cards = []
        for name in ("before", "after"):
            self.evidence[name]["cards"] = []
        self.validate(self.evidence)
        self.evidence["after"]["scroll"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "moved"):
            self.validate(self.evidence)

    def test_required_evidence_rejected_before_collection_scoring(self):
        from apc.perception.evaluate_browser_collection import load_collection
        path = Path(__file__).resolve().parents[1] / "data/reference/browser-routed-audit-v5/collection.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["capture_stability_required"] = True
        with patch("pathlib.Path.read_text", return_value=json.dumps(data)):
            with self.assertRaisesRegex(ValueError, "stability evidence"):
                load_collection(path)


if __name__ == "__main__":
    unittest.main()
