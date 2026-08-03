from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from poker_coach import CoachDatabase, HandHistoryFolderScanner, split_completed_hands


ROOT = Path(__file__).resolve().parents[1]


class FolderIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = (ROOT / "examples" / "sample_play_money_hand.txt").read_text(
            encoding="utf-8"
        )

    def test_completed_split_ignores_a_hand_still_being_written(self) -> None:
        incomplete = self.sample.replace("90000000001", "90000000002").split(
            "*** SUMMARY ***", 1
        )[0]
        complete, pending = split_completed_hands(self.sample + "\n\n" + incomplete)
        self.assertEqual(len(complete), 1)
        self.assertEqual(pending, 1)

    def test_scanner_is_incremental_and_imports_completed_append(self) -> None:
        second = self.sample.replace("90000000001", "90000000002")
        incomplete, summary = second.split("*** SUMMARY ***", 1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "HH20260801.txt"
            history.write_text(self.sample + "\n\n" + incomplete, encoding="utf-8")
            with CoachDatabase(root / "coach.sqlite3") as database:
                scanner = HandHistoryFolderScanner(database)
                first = scanner.scan(root)
                self.assertEqual((first.inserted, first.incomplete_blocks), (1, 1))
                self.assertEqual(database.hand_count, 1)

                second_scan = scanner.scan(root)
                self.assertEqual((second_scan.changed_files, second_scan.skipped_files), (0, 1))

                history.write_text(
                    self.sample + "\n\n" + incomplete + "*** SUMMARY ***" + summary,
                    encoding="utf-8",
                )
                third = scanner.scan(root)
                self.assertEqual((third.inserted, third.unchanged), (1, 1))
                self.assertEqual(third.incomplete_blocks, 0)
                self.assertEqual(database.hand_count, 2)
                self.assertEqual(database.ingested_file_count, 1)

    def test_recursive_scan_and_invalid_folder_validation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "HAND.TXT").write_text(self.sample, encoding="utf-8")
            with CoachDatabase(root / "coach.sqlite3") as database:
                scanner = HandHistoryFolderScanner(database)
                self.assertEqual(scanner.scan(root).files_seen, 0)
                recursive = scanner.scan(root, recursive=True)
                self.assertEqual((recursive.files_seen, recursive.inserted), (1, 1))
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    scanner.scan(root / "missing")


if __name__ == "__main__":
    unittest.main()
