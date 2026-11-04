import json
import unittest
from unittest.mock import patch

from apc.desktop import inspect_saved_frame


class DesktopInspectionTests(unittest.TestCase):
    def test_reader_selection_and_provenance(self):
        for filename, paired in (("glyph-v1.pt", False), ("corner-glyph-pair-v2.pt", True),
                                 ("corner-glyph-mixed-v1.pt", True)):
            with self.subTest(filename=filename), patch("apc.perception.infer_cards.CardReader") as reader:
                reader.return_value.read.return_value = {"cards": [], "full_table_readiness": False}
                result = json.loads(inspect_saved_frame("frame.png", filename))
                self.assertEqual(result["checkpoint"], filename)
                self.assertFalse(result["inspection"]["full_table_readiness"])
                self.assertEqual(reader.call_args.kwargs, {"corner_pairs": paired})
                reader.return_value.read.assert_called_once_with("frame.png")

    def test_unknown_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            inspect_saved_frame("frame.png", "../unknown.pt")
