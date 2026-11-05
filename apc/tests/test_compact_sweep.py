from pathlib import Path
import tempfile
import unittest

from apc.perception.evaluate_browser_collection import load_collection
from apc.synthetic.render_compact_sweep import render_sweep

FONT = Path("C:/Windows/Fonts/segoeuib.ttf")


@unittest.skipUnless(FONT.exists(), "Explicit sweep font not available")
class CompactSweepTests(unittest.TestCase):
    def test_reproducible_geometry_and_no_capture_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = render_sweep(root/"first",FONT,seed=42)
            second = render_sweep(root/"second",FONT,seed=42)
            self.assertEqual(first,second)
            self.assertEqual(len(first["frames"]),60)
            self.assertEqual(sum(len(r["cards"]) for r in first["frames"]),96)
            self.assertEqual(sum(not r["cards"] for r in first["frames"]),12)
            self.assertEqual(first["source"],"synthetic_compact_geometry_sweep_v1")
            self.assertEqual(first["use_policy"],"synthetic_development_only")
            self.assertFalse(first["readiness_credit"])
            _,frames = load_collection(root/"first/collection.json")
            self.assertEqual(len(frames),60)
            for row in first["frames"][:48]:
                cards = row["cards"]
                self.assertEqual(cards[1][2]-cards[0][2],44+row["sweep_condition"]["gap"])
                self.assertNotEqual(cards[0][1],cards[1][1])
            with self.assertRaises(FileExistsError):
                render_sweep(root/"first",FONT,seed=42)


if __name__ == "__main__":
    unittest.main()
