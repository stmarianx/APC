from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Iterable, Protocol


D = Decimal


class ReviewRating(str, Enum):
    AGAIN = "again"
    HARD = "hard"
    GOOD = "good"
    EASY = "easy"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class StudyState:
    attempts: int
    streak: int
    mastery: Decimal
    interval_days: int
    due_at: datetime
    last_rating: ReviewRating | None = None

    def __post_init__(self) -> None:
        if self.attempts < 0 or self.streak < 0 or self.interval_days < 0:
            raise ValueError("Study counters cannot be negative")
        if not D("0") <= self.mastery <= D("1"):
            raise ValueError("Mastery must be between zero and one")
        if self.due_at.tzinfo is None:
            raise ValueError("Study due_at must be timezone aware")

    @classmethod
    def initial(cls, now: datetime | None = None) -> "StudyState":
        return cls(0, 0, D("0"), 0, now or utc_now())

    @property
    def status(self) -> str:
        if self.mastery >= D("0.85") and self.attempts >= 4:
            return "mastered"
        if self.attempts == 0:
            return "new"
        return "learning"

    def to_dict(self) -> dict[str, object]:
        return {
            "attempts": self.attempts,
            "streak": self.streak,
            "mastery": format(self.mastery, "f"),
            "interval_days": self.interval_days,
            "due_at": self.due_at.astimezone(timezone.utc).isoformat(),
            "last_rating": None if self.last_rating is None else self.last_rating.value,
            "status": self.status,
        }


def schedule_review(
    state: StudyState,
    rating: ReviewRating | str,
    *,
    now: datetime | None = None,
) -> StudyState:
    reviewed_at = now or utc_now()
    if reviewed_at.tzinfo is None:
        raise ValueError("Review time must be timezone aware")
    try:
        result_rating = rating if isinstance(rating, ReviewRating) else ReviewRating(rating)
    except ValueError as error:
        raise ValueError("Rating must be again, hard, good, or easy") from error
    score = {
        ReviewRating.AGAIN: D("0"),
        ReviewRating.HARD: D("0.45"),
        ReviewRating.GOOD: D("0.78"),
        ReviewRating.EASY: D("1"),
    }[result_rating]
    mastery = state.mastery * D("0.55") + score * D("0.45")
    if result_rating == ReviewRating.AGAIN:
        streak = 0
        interval = 0
        due = reviewed_at + timedelta(minutes=10)
    elif result_rating == ReviewRating.HARD:
        streak = state.streak
        interval = max(1, round(max(1, state.interval_days) * 1.3))
        due = reviewed_at + timedelta(days=interval)
    elif result_rating == ReviewRating.GOOD:
        streak = state.streak + 1
        interval = 1 if state.interval_days == 0 else max(2, round(state.interval_days * 2.0))
        due = reviewed_at + timedelta(days=interval)
    else:
        streak = state.streak + 1
        interval = 4 if state.interval_days == 0 else max(4, round(state.interval_days * 2.5))
        due = reviewed_at + timedelta(days=interval)
    return StudyState(
        attempts=state.attempts + 1,
        streak=streak,
        mastery=min(D("1"), mastery),
        interval_days=interval,
        due_at=due,
        last_rating=result_rating,
    )


class StudyStore(Protocol):
    def upsert_drills(self, drills: Iterable[dict[str, object]]) -> tuple[int, int]: ...
    def list_drills(self, *, due_only: bool = False, now: datetime | None = None) -> list[dict[str, object]]: ...
    def review_drill(self, drill_id: str, rating: ReviewRating | str, *, now: datetime | None = None) -> dict[str, object]: ...


class InMemoryStudyStore:
    def __init__(self) -> None:
        self._drills: dict[str, dict[str, object]] = {}
        self._states: dict[str, StudyState] = {}
        self._attempts: list[dict[str, object]] = []
        self._lock = threading.RLock()

    def upsert_drills(self, drills: Iterable[dict[str, object]]) -> tuple[int, int]:
        inserted = updated = 0
        with self._lock:
            for drill in drills:
                drill_id = str(drill.get("drill_id", ""))
                if not drill_id:
                    raise ValueError("Drill payload requires drill_id")
                if drill_id in self._drills:
                    updated += 1
                else:
                    inserted += 1
                    self._states[drill_id] = StudyState.initial()
                self._drills[drill_id] = dict(drill)
        return inserted, updated

    def list_drills(self, *, due_only: bool = False, now: datetime | None = None) -> list[dict[str, object]]:
        current = now or utc_now()
        with self._lock:
            rows = []
            for drill_id, payload in self._drills.items():
                state = self._states[drill_id]
                if due_only and state.due_at > current:
                    continue
                row = dict(payload)
                row["study"] = state.to_dict()
                rows.append(row)
            return sorted(
                rows,
                key=lambda row: (
                    str(row["study"]["due_at"]),
                    -float(str(row.get("priority", "0"))),
                    str(row["drill_id"]),
                ),
            )

    def review_drill(
        self,
        drill_id: str,
        rating: ReviewRating | str,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if drill_id not in self._drills:
                raise KeyError(f"Unknown drill: {drill_id}")
            reviewed_at = now or utc_now()
            state = schedule_review(self._states[drill_id], rating, now=reviewed_at)
            self._states[drill_id] = state
            self._attempts.append(
                {
                    "drill_id": drill_id,
                    "rating": state.last_rating.value if state.last_rating else None,
                    "reviewed_at": reviewed_at.astimezone(timezone.utc).isoformat(),
                    "state": state.to_dict(),
                }
            )
            row = dict(self._drills[drill_id])
            row["study"] = state.to_dict()
            return row

    @property
    def attempt_count(self) -> int:
        return len(self._attempts)
