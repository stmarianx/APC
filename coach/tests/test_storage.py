from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import hashlib
import json

from poker_coach import ActionSolution, CoachDatabase, SolutionKey, SolvedSpot
from poker_coach.models import Card
from poker_coach.study import ReviewRating


class StorageTests(unittest.TestCase):
    def test_hand_import_is_idempotent_and_analyzable(self) -> None:
        history_path = Path(__file__).resolve().parent.parent / "examples" / "sample_play_money_hand.txt"
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "coach.sqlite3"
            with CoachDatabase(database_path) as database:
                first = database.import_file(history_path)
                second = database.import_file(history_path)
                self.assertEqual((first.inserted, first.updated), (1, 0))
                self.assertEqual((second.inserted, second.updated, second.unchanged), (0, 0, 1))
                self.assertEqual(database.hand_count, 1)
                report = database.analyze()
                self.assertEqual(report["hands"], 1)
                self.assertEqual(report["hand_reports"][0]["reconciliation_error"], "0")

    def test_solution_round_trip(self) -> None:
        key = SolutionKey(
            game="holdem_no_limit",
            players=2,
            hero_position="BTN",
            effective_stack_bb=Decimal("100"),
            pot_bb=Decimal("6.5"),
            board=(Card.parse("Ah"), Card.parse("7c"), Card.parse("2d")),
            action_history=("BTN raise", "BB call"),
            rake_model="fixture",
        )
        spot = SolvedSpot(
            key,
            (
                ActionSolution("check", Decimal("0.4"), Decimal("12")),
                ActionSolution("bet:2.2", Decimal("0.6"), Decimal("13")),
            ),
            source="fixture",
            source_version="1",
            node_id="fixture_flop_node",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "coach.sqlite3"
            with CoachDatabase(path) as database:
                database.put(spot)
            with CoachDatabase(path) as database:
                self.assertEqual(database.get(key), spot)
                self.assertEqual(database.list_solutions(), (spot,))

    def test_legacy_exact_suit_fingerprint_is_migrated_on_reopen(self) -> None:
        key = SolutionKey(
            game="holdem_no_limit",
            players=2,
            hero_position="BTN",
            effective_stack_bb=Decimal("100"),
            pot_bb=Decimal("6.5"),
            board=(Card.parse("Ah"), Card.parse("7c"), Card.parse("2d")),
            hero_cards=(Card.parse("As"), Card.parse("Kd")),
            action_history=("BTN raise_to:2.5", "BB call"),
            rake_model="fixture",
        )
        spot = SolvedSpot(
            key,
            (
                ActionSolution("check", Decimal("0.4"), Decimal("12")),
                ActionSolution("bet:0.33", Decimal("0.6"), Decimal("13")),
            ),
            source="legacy-fixture",
            source_version="1",
        )
        legacy_payload = json.dumps(
            key.canonical(), sort_keys=True, separators=(",", ":")
        )
        legacy_fingerprint = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()
        self.assertNotEqual(legacy_fingerprint, key.fingerprint)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "coach.sqlite3"
            with CoachDatabase(path) as database:
                database.put(spot)
                database.connection.execute(
                    "UPDATE solutions SET fingerprint = ? WHERE fingerprint = ?",
                    (legacy_fingerprint, key.fingerprint),
                )
                database.connection.commit()
            with CoachDatabase(path) as migrated:
                self.assertEqual(migrated.get(key), spot)
                self.assertEqual(len(migrated.list_solutions()), 1)

    def test_drill_state_and_attempts_persist_across_reopen(self) -> None:
        drill = {"drill_id": "d" * 20, "priority": "0.25", "title": "River defense"}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "coach.sqlite3"
            with CoachDatabase(path) as database:
                self.assertEqual(database.upsert_drills((drill,)), (1, 0))
                reviewed = database.review_drill("d" * 20, ReviewRating.GOOD)
                self.assertEqual(reviewed["study"]["attempts"], 1)
            with CoachDatabase(path) as database:
                rows = database.list_drills()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["study"]["last_rating"], "good")
                self.assertEqual(database.drill_attempt_count, 1)


if __name__ == "__main__":
    unittest.main()
