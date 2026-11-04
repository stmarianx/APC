import json
import tempfile
import unittest
from pathlib import Path

from apc.perception.evaluate_browser_collection import load_collection


COLLECTION = Path(__file__).resolve().parents[1] / "data/development/browser-domain-bridge-v1/collection.json"


class BrowserCollectionTests(unittest.TestCase):
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
