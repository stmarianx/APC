import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from apc.__main__ import main


class InspectModelSelectionTests(unittest.TestCase):
    def test_saved_frame_model_selection_is_explicit_and_adaptive(self):
        with patch("sys.argv", ["apc", "inspect", "frame.png", "--neural-cards", "--adaptive-glyphs", "--glyph-checkpoint", "candidate.pt"]), \
             patch("apc.perception.infer_cards.CardReader") as reader, contextlib.redirect_stdout(io.StringIO()):
            reader.return_value.read.return_value = {"full_table_readiness": False}
            main()
        args, kwargs = reader.call_args
        self.assertEqual(args, (Path("candidate.pt"),))
        self.assertTrue(kwargs["surface_pairs"])
        self.assertTrue(kwargs["corner_pairs"])
        reader.return_value.read.assert_called_once_with(Path("frame.png"))

    def test_model_override_requires_explicit_neural_inspection(self):
        with patch("sys.argv", ["apc", "inspect", "frame.png", "--glyph-checkpoint", "candidate.pt"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
