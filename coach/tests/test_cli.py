import json
import tempfile
import unittest
from pathlib import Path

from poker_coach.cli import main


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_play_money_hand.txt"


class OfflineCliTests(unittest.TestCase):
    def test_recursive_folder_scan_exports_compact_bb_profile_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "account"
            nested.mkdir()
            sample = SAMPLE.read_text(encoding="utf-8")
            (nested / "HH-one.txt").write_text(sample, encoding="utf-8")
            (nested / "HH-two.txt").write_text(
                sample.replace("90000000001", "90000000002"),
                encoding="utf-8",
            )
            database = root / "coach.sqlite3"
            output = root / "profiles.json"

            status = main(
                [
                    str(root),
                    "--database",
                    str(database),
                    "--recursive",
                    "--profiles-only",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(status, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.0.0")
            self.assertEqual(payload["units"], "BB")
            self.assertEqual(payload["source"]["hands"], 2)
            self.assertEqual(payload["source"]["scan"]["inserted"], 2)
            self.assertIn("Hero", payload["players"])
            self.assertIn("summary", payload["players"]["Hero"])
            self.assertIn("estimates", payload["players"]["Hero"])
            self.assertIn("exploit_insights", payload["players"]["Hero"])

            rerun = main(
                [
                    str(root),
                    "--database",
                    str(database),
                    "--recursive",
                    "--profiles-only",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rerun, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"]["hands"], 2)
            self.assertEqual(payload["source"]["scan"]["skipped_files"], 2)

    def test_folder_requires_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(main([directory]), 2)

    def test_recursive_flag_rejects_file_input(self) -> None:
        self.assertEqual(main([str(SAMPLE), "--recursive"]), 2)


if __name__ == "__main__":
    unittest.main()
