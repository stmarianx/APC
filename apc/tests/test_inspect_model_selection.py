import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from apc.__main__ import main


class InspectModelSelectionTests(unittest.TestCase):
    def test_explicit_verifier_selection_uses_research_reader(self):
        with patch("sys.argv", ["apc", "inspect", "frame.png", "--neural-cards", "--verifier-checkpoint", "verifier.pt"]), \
             patch("apc.perception.verified_card_reader.VerifiedCardReader") as reader, contextlib.redirect_stdout(io.StringIO()):
            reader.return_value.read.return_value = {}
            main()
        self.assertEqual(reader.call_args.kwargs["verifier_checkpoint"], Path("verifier.pt"))
        with patch("sys.argv", ["apc", "inspect", "frame.png", "--verifier-checkpoint", "verifier.pt"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()

    def test_explicit_surface_routing_requires_both_verifiers_and_adaptive_mode(self):
        args = ["apc", "inspect", "frame.png", "--neural-cards", "--adaptive-surfaces",
                "--verifier-checkpoint", "ambiguous.pt", "--bounded-verifier-checkpoint", "bounded.pt"]
        with patch("sys.argv", args), patch("apc.perception.verified_card_reader.VerifiedCardReader") as reader, \
             contextlib.redirect_stdout(io.StringIO()):
            reader.return_value.read.return_value = {}
            main()
        self.assertEqual(reader.call_args.kwargs["bounded_verifier_checkpoint"], Path("bounded.pt"))
        for missing in (["apc", "inspect", "frame.png", "--bounded-verifier-checkpoint", "bounded.pt"],
                        ["apc", "inspect", "frame.png", "--neural-cards", "--verifier-checkpoint", "ambiguous.pt", "--bounded-verifier-checkpoint", "bounded.pt"]):
            with patch("sys.argv", missing), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()

    def test_tiled_inspection_is_opt_in_and_preserves_selected_weights(self):
        arguments = ["apc", "inspect", "frame.png", "--neural-cards", "--adaptive-glyphs",
                     "--adaptive-surfaces", "--deduplicate-surfaces", "--tiled-views",
                     "--glyph-checkpoint", "candidate.pt", "--locator-checkpoint", "detector.pt"]
        with patch("sys.argv", arguments), patch("apc.perception.infer_cards.CardReader") as reader, \
             contextlib.redirect_stdout(io.StringIO()):
            reader.return_value.read.return_value = {}
            main()
        for key in ("adaptive_surfaces", "deduplicate_surfaces", "tiled_views"):
            self.assertTrue(reader.call_args.kwargs[key])

    def test_tiled_inspection_requires_explicit_adaptive_mode(self):
        for options in (["--tiled-views"], ["--deduplicate-surfaces"], ["--adaptive-surfaces"]):
            with patch("sys.argv", ["apc", "inspect", "frame.png"] + options), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)

    def test_saved_frame_model_selection_is_explicit_and_adaptive(self):
        with patch("sys.argv", ["apc", "inspect", "frame.png", "--neural-cards", "--adaptive-glyphs", "--glyph-checkpoint", "candidate.pt", "--locator-checkpoint", "detector.pt"]), \
             patch("apc.perception.infer_cards.CardReader") as reader, contextlib.redirect_stdout(io.StringIO()):
            reader.return_value.read.return_value = {"full_table_readiness": False}
            main()
        args, kwargs = reader.call_args
        self.assertEqual(args, (Path("candidate.pt"),))
        self.assertTrue(kwargs["surface_pairs"])
        self.assertTrue(kwargs["corner_pairs"])
        self.assertEqual(kwargs["locator_checkpoint"], Path("detector.pt"))
        reader.return_value.read.assert_called_once_with(Path("frame.png"))

    def test_model_override_requires_explicit_neural_inspection(self):
        with patch("sys.argv", ["apc", "inspect", "frame.png", "--glyph-checkpoint", "candidate.pt"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)

    def test_detector_override_requires_explicit_neural_inspection(self):
        with patch("sys.argv", ["apc", "inspect", "frame.png", "--locator-checkpoint", "detector.pt"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
