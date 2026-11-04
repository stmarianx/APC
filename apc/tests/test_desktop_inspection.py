import json
import unittest
from unittest.mock import patch

from apc.desktop import inspect_saved_frame, checkpoint_status_text


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

    def test_neural_trial_is_explicit_and_preserves_uncertainty(self):
        with patch("apc.perception.infer_cards.CardReader") as reader:
            reader.return_value.read.return_value = {"cards": [{"card": None}], "full_table_readiness": False}
            result = json.loads(inspect_saved_frame("frame.png", "glyph-surface-pair-v1.pt"))
        self.assertTrue(reader.call_args.kwargs["corner_pairs"])
        self.assertTrue(reader.call_args.kwargs["refine_surfaces"])
        self.assertEqual(reader.call_args.kwargs["locator_checkpoint"].name, "card-locator-mixed-v1.pt")
        self.assertIsNone(result["inspection"]["cards"][0]["card"])

    def test_checkpoint_evidence_failure_is_visible(self):
        from unittest.mock import mock_open
        record = {"status": "offline_neural_candidate_unpromoted", "gates": {"visible_table_training_ready": False}}
        with patch("pathlib.Path.read_text", return_value=json.dumps(record)), \
                patch("apc.neural.train_candidate.validate_checkpoint", return_value={"valid": False, "issues": ["weights changed"]}):
            result = json.loads(checkpoint_status_text("checkpoint.json"))
        self.assertFalse(result["validation"]["valid"])
        self.assertFalse(result["gates"]["visible_table_training_ready"])
        self.assertIn("does not activate", result["notice"])

    def test_unknown_evidence_type_is_rejected(self):
        with patch("pathlib.Path.read_text", return_value='{"status": "ready"}'):
            with self.assertRaises(ValueError):
                checkpoint_status_text("checkpoint.json")
