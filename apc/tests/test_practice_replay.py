import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apc.__main__ import practice
from apc.neural.replay_buffer import APCReplayBuffer


class PracticeReplayTests(unittest.TestCase):
    def test_completed_fold_is_eligible_but_quit_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("builtins.input", return_value="quit"), contextlib.redirect_stdout(io.StringIO()):
                practice(12, replay_dir=root / "quit", session_id="session")
            self.assertFalse((root / "quit").exists())
            with patch("builtins.input", side_effect=["invalid", "fold"]), contextlib.redirect_stdout(io.StringIO()):
                practice(12, replay_dir=root / "complete", session_id="session")
            report = APCReplayBuffer(root / "complete").validate()
            self.assertTrue(report["valid"], report)

    def test_eof_creates_no_learning_record(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "recording"
            with patch("builtins.input", side_effect=EOFError), contextlib.redirect_stdout(io.StringIO()):
                practice(13, replay_dir=destination, session_id="session")
            self.assertFalse(destination.exists())
