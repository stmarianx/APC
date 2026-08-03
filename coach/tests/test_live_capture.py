from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from poker_coach import (
    LiveTableService,
    PokerStarsLiveTailAdapter,
    SolverExportRegistry,
)


ROOT = Path(__file__).resolve().parents[1]

PREFLOP = """PokerStars Hand #91000000001: Hold'em No Limit (0.50/1 Play Money) - 2026/08/01 21:00:00 ET
Table 'Solver Tail' 2-max Seat #1 is the button
Seat 1: Hero (100 in chips)
Seat 2: Villain (100 in chips)
Hero: posts small blind 0.50
Villain: posts big blind 1
*** HOLE CARDS ***
Dealt to Hero [As Kd]
"""

HERO_RAISE = "Hero: raises 2.50 to 3\n"
PREFLOP_CLOSED = HERO_RAISE + "Villain: calls 2\n"
FLOP_DECISION = PREFLOP_CLOSED + "*** FLOP *** [Ah 7c 2d]\nVillain: checks\n"
TURN_DECISION = FLOP_DECISION + "Hero: bets 2.15\nVillain: calls 2.15\n*** TURN *** [Ah 7c 2d] [9s]\nVillain: checks\n"
RIVER_DECISION = TURN_DECISION + "Hero: bets 8.10\nVillain: calls 8.10\n*** RIVER *** [Ah 7c 2d 9s] [4h]\nVillain: checks\n"


class LiveCaptureAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solutions = SolverExportRegistry().parse_file(
            ROOT / "examples" / "sample_solver_export.csv"
        ).bundle.spots

    def test_growing_tail_emits_preflop_through_river_revisions(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "HH-live.txt"
            adapter = PokerStarsLiveTailAdapter()
            service = LiveTableService()
            session = service.create_session("Play Table Tail")

            expected = [
                (PREFLOP, 0, "preflop", "exact"),
                (PREFLOP + FLOP_DECISION, 1, "flop", "close"),
                (PREFLOP + TURN_DECISION, 2, "turn", "close"),
                (PREFLOP + RIVER_DECISION, 3, "river", "close"),
            ]
            for source, revision, street, confidence in expected:
                path.write_text(source, encoding="utf-8")
                poll = adapter.poll(path, table_id="Play Table Tail")
                self.assertEqual((poll.status, poll.changed), ("state_ready", True))
                self.assertEqual(poll.revision, revision)
                self.assertEqual(poll.payload["revision"], revision)
                result = service.update_state(
                    session["session_id"], poll.payload, self.solutions
                )
                self.assertEqual(result["state"]["street"], street)
                self.assertEqual(result["match"]["confidence"], confidence)

            unchanged = adapter.poll(path, table_id="Play Table Tail")
            self.assertFalse(unchanged.changed)
            self.assertEqual(unchanged.revision, 3)

    def test_projection_waits_for_opponent_and_closed_round(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "HH-live.txt"
            adapter = PokerStarsLiveTailAdapter()
            path.write_text(PREFLOP + HERO_RAISE, encoding="utf-8")
            waiting = adapter.poll(path, table_id="Play Table Tail")
            self.assertEqual(waiting.status, "waiting_for_player")
            self.assertEqual(waiting.next_actor, "Villain")

            path.write_text(PREFLOP + PREFLOP_CLOSED, encoding="utf-8")
            closed = adapter.poll(path, table_id="Play Table Tail")
            self.assertEqual(closed.status, "waiting_for_next_street")

    def test_completed_or_partially_written_tail_never_emits_state(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "HH-live.txt"
            adapter = PokerStarsLiveTailAdapter()
            path.write_text(PREFLOP + "Villain: calls\n", encoding="utf-8")
            partial = adapter.poll(path, table_id="Play Table Tail")
            self.assertEqual(partial.status, "pending_write")
            self.assertIsNone(partial.payload)

            path.write_text(
                PREFLOP + RIVER_DECISION + "*** SUMMARY ***\nTotal pot 26.50 | Rake 0\n",
                encoding="utf-8",
            )
            complete = adapter.poll(path, table_id="Play Table Tail")
            self.assertEqual(complete.status, "waiting_for_new_hand")
            self.assertIsNone(complete.payload)

    def test_directory_source_selects_latest_text_history(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "old.txt"
            newest = root / "new.txt"
            older.write_text(PREFLOP, encoding="utf-8")
            newest.write_text(PREFLOP.replace("91000000001", "91000000002"), encoding="utf-8")
            older.touch()
            newest.touch()
            poll = PokerStarsLiveTailAdapter().poll(
                root, table_id="Play Table Tail"
            )
            self.assertIn(poll.hand_id, {"91000000001", "91000000002"})
            self.assertEqual(poll.status, "state_ready")


if __name__ == "__main__":
    unittest.main()
