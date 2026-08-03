from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest

from poker_coach.study import InMemoryStudyStore, ReviewRating, StudyState, schedule_review


NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
DRILL = {"drill_id": "a" * 20, "priority": "0.15", "title": "Sizing drill"}


class StudyTests(unittest.TestCase):
    def test_rating_transitions_and_mastery(self) -> None:
        state = StudyState.initial(NOW)
        state = schedule_review(state, ReviewRating.GOOD, now=NOW)
        self.assertEqual((state.attempts, state.streak, state.interval_days), (1, 1, 1))
        self.assertEqual(state.due_at, NOW + timedelta(days=1))
        for index in range(3):
            state = schedule_review(state, ReviewRating.EASY, now=NOW + timedelta(days=index + 1))
        self.assertEqual(state.status, "mastered")
        self.assertGreaterEqual(state.mastery, Decimal("0.85"))

    def test_again_resets_streak_and_schedules_short_retry(self) -> None:
        learned = schedule_review(StudyState.initial(NOW), "easy", now=NOW)
        reset = schedule_review(learned, "again", now=NOW + timedelta(days=1))
        self.assertEqual(reset.streak, 0)
        self.assertEqual(reset.interval_days, 0)
        self.assertEqual(reset.due_at, NOW + timedelta(days=1, minutes=10))

    def test_in_memory_store_is_idempotent_and_tracks_attempts(self) -> None:
        store = InMemoryStudyStore()
        self.assertEqual(store.upsert_drills((DRILL,)), (1, 0))
        self.assertEqual(store.upsert_drills((DRILL,)), (0, 1))
        reviewed = store.review_drill("a" * 20, "good", now=NOW)
        self.assertEqual(reviewed["study"]["attempts"], 1)
        self.assertEqual(store.attempt_count, 1)
        self.assertEqual(len(store.list_drills(due_only=True, now=NOW)), 0)
        self.assertEqual(len(store.list_drills(due_only=True, now=NOW + timedelta(days=1))), 1)

    def test_invalid_rating_and_naive_time_fail(self) -> None:
        with self.assertRaises(ValueError):
            schedule_review(StudyState.initial(NOW), "perfect", now=NOW)
        with self.assertRaises(ValueError):
            StudyState.initial(datetime(2026, 8, 1))


if __name__ == "__main__":
    unittest.main()
