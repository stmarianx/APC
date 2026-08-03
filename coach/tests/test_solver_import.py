import json
from pathlib import Path
import tempfile
import unittest

from poker_coach.solver_import import SolverBundleError, SolverBundleImporter, solved_spot_to_dict
from poker_coach.storage import CoachDatabase


ROOT = Path(__file__).resolve().parents[1]


class SolverImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.importer = SolverBundleImporter()
        self.path = ROOT / "examples" / "sample_solver_bundle.json"

    def test_parse_per_hand_spots_and_provenance(self) -> None:
        bundle = self.importer.parse_file(self.path)
        self.assertEqual(len(bundle.spots), 9)
        self.assertNotEqual(bundle.spots[0].key.fingerprint, bundle.spots[1].key.fingerprint)
        self.assertEqual(str(bundle.spots[0].key.hero_cards[0]), "Kc")
        self.assertEqual(bundle.spots[0].node_id, "btn_vs_bb_a72r_kq")
        row = solved_spot_to_dict(bundle.spots[0])
        self.assertEqual(row["source"], "example_solver_export")
        self.assertEqual(row["actions"][0]["ev_loss"], "0.09")

    def test_database_import_is_idempotent(self) -> None:
        bundle = self.importer.parse_file(self.path)
        with tempfile.TemporaryDirectory() as directory:
            with CoachDatabase(Path(directory) / "solutions.sqlite3") as database:
                first = self.importer.import_into(database, bundle)
                second = self.importer.import_into(database, bundle)
                self.assertEqual((first.inserted, first.updated), (9, 0))
                self.assertEqual((second.inserted, second.updated), (0, 9))
                self.assertEqual(database.get(bundle.spots[1].key), bundle.spots[1])

    def test_invalid_schema_frequency_and_duplicate_spot_fail(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["schema_version"] = "2.0.0"
        with self.assertRaises(SolverBundleError):
            self.importer.parse_dict(payload)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["spots"][0]["actions"][0]["frequency"] = "0.99"
        with self.assertRaises(SolverBundleError):
            self.importer.parse_dict(payload)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        duplicate = json.loads(json.dumps(payload["spots"][0]))
        duplicate["node_id"] = "another_node_id"
        payload["spots"].append(duplicate)
        with self.assertRaises(SolverBundleError):
            self.importer.parse_dict(payload)


if __name__ == "__main__":
    unittest.main()
