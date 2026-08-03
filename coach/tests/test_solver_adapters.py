from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from poker_coach import (
    BUNDLE_JSON_V1,
    TABULAR_CSV_V1,
    CoachDatabase,
    SolverBundleError,
    SolverBundleImporter,
    SolverExportRegistry,
    TabularSolverCSVAdapter,
)


ROOT = Path(__file__).resolve().parents[1]


class TabularSolverAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.csv_path = ROOT / "examples" / "sample_solver_export.csv"
        self.csv_source = self.csv_path.read_text(encoding="utf-8")
        self.registry = SolverExportRegistry()

    def test_multistreet_csv_groups_actions_into_valid_spots(self) -> None:
        parsed = self.registry.parse_file(self.csv_path)
        self.assertEqual(parsed.format_name, TABULAR_CSV_V1)
        self.assertEqual(len(parsed.bundle.spots), 4)
        self.assertEqual(parsed.bundle.source, "example_multistreet_solver_export")
        self.assertEqual(len(parsed.bundle.spots[0].key.board), 0)
        self.assertEqual(len(parsed.bundle.spots[-1].key.board), 5)
        self.assertEqual(len(parsed.bundle.spots[-1].key.action_history), 9)
        self.assertEqual(len(parsed.bundle.spots[-1].actions), 3)

    def test_registry_auto_detects_json_and_csv_content(self) -> None:
        csv_parsed = self.registry.parse_text(self.csv_source)
        json_source = (ROOT / "examples" / "sample_solver_bundle.json").read_text(
            encoding="utf-8"
        )
        json_parsed = self.registry.parse_text(json_source)
        self.assertEqual(csv_parsed.format_name, TABULAR_CSV_V1)
        self.assertEqual(json_parsed.format_name, BUNDLE_JSON_V1)

    def test_csv_import_is_idempotent(self) -> None:
        parsed = self.registry.parse_file(self.csv_path)
        importer = SolverBundleImporter()
        with TemporaryDirectory() as directory:
            with CoachDatabase(Path(directory) / "coach.sqlite3") as database:
                first = importer.import_into(database, parsed.bundle)
                second = importer.import_into(database, parsed.bundle)
                self.assertEqual((first.inserted, first.updated), (4, 0))
                self.assertEqual((second.inserted, second.updated), (0, 4))
                self.assertEqual(len(database.list_solutions()), 4)

    def test_missing_column_and_inconsistent_node_key_fail_explicitly(self) -> None:
        header, *rows = self.csv_source.splitlines()
        missing_header = header.replace(",ev", "")
        missing_rows = [row.rsplit(",", 1)[0] for row in rows]
        with self.assertRaisesRegex(SolverBundleError, "missing columns: ev"):
            TabularSolverCSVAdapter().parse_text(
                "\n".join((missing_header, *missing_rows))
            )

        changed = self.csv_source.replace(
            "hu_ako_a72r_flop,holdem_no_limit,2,BTN,97,6.5,Ah 7c 2d",
            "hu_ako_a72r_flop,holdem_no_limit,2,BTN,97,7.5,Ah 7c 2d",
            1,
        )
        with self.assertRaisesRegex(SolverBundleError, "changes key fields"):
            TabularSolverCSVAdapter().parse_text(changed)

    def test_frequency_validation_is_shared_with_bundle_importer(self) -> None:
        invalid = self.csv_source.replace(
            "hu_btn_open_ako_preflop,holdem_no_limit,2,BTN,99,1.5,,As Kd,,training_no_rake,chip_ev,2.5|3,raise_to:3,0.25,0.17",
            "hu_btn_open_ako_preflop,holdem_no_limit,2,BTN,99,1.5,,As Kd,,training_no_rake,chip_ev,2.5|3,raise_to:3,0.35,0.17",
        )
        with self.assertRaisesRegex(SolverBundleError, "frequencies must sum to one"):
            self.registry.parse_text(invalid, format_name=TABULAR_CSV_V1)


if __name__ == "__main__":
    unittest.main()
