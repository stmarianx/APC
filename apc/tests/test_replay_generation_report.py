import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apc.neural.diverse_self_play_replay import main


class GenerationReportTests(unittest.TestCase):
    def test_report_persists_failed_coverage_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            args = ["generate", directory, "--report", str(path)]
            report = {"training_eligible": False, "hands": 240}
            with patch.object(sys, "argv", args), patch(
                "apc.neural.diverse_self_play_replay.build_diverse_virtual_replay_buffer",
                return_value=report
            ) as build, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main()
                self.assertEqual(json.loads(path.read_text()), report)
                with self.assertRaises(SystemExit):
                    main()
                build.assert_called_once()
