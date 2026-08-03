from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from .models import Card, HandHistory
from .pokerstars import PokerStarsParser
from .report import analyze_hands
from .solutions import ActionSolution, SolutionKey, SolutionStore, SolvedSpot
from .study import ReviewRating, StudyState, schedule_review, utc_now


@dataclass(frozen=True)
class ImportResult:
    inserted: int
    updated: int
    unchanged: int
    hand_ids: tuple[str, ...]


@dataclass(frozen=True)
class IngestedFileState:
    path: str
    size: int
    modified_ns: int
    completed_blocks: int
    last_error: str | None


class CoachDatabase(SolutionStore):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS hand_histories (
                hand_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                table_name TEXT NOT NULL,
                played_at TEXT NOT NULL,
                hero TEXT,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS solutions (
                fingerprint TEXT PRIMARY KEY,
                key_json TEXT NOT NULL,
                spot_json TEXT NOT NULL,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS drills (
                drill_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                priority TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                streak INTEGER NOT NULL DEFAULT 0,
                mastery TEXT NOT NULL DEFAULT '0',
                interval_days INTEGER NOT NULL DEFAULT 0,
                due_at TEXT NOT NULL,
                last_rating TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS drill_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                drill_id TEXT NOT NULL REFERENCES drills(drill_id) ON DELETE CASCADE,
                rating TEXT NOT NULL,
                reviewed_at TEXT NOT NULL,
                resulting_state_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ingested_files (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                modified_ns INTEGER NOT NULL,
                completed_blocks INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                scanned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.connection.commit()
        self._migrate_solution_fingerprints()

    def __enter__(self) -> "CoachDatabase":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _migrate_solution_fingerprints(self) -> None:
        """Re-index legacy exact-suit keys under the current canonical fingerprint."""

        with self._lock:
            rows = self.connection.execute(
                "SELECT fingerprint, key_json, spot_json FROM solutions ORDER BY fingerprint"
            ).fetchall()
            for row in rows:
                old_fingerprint = str(row["fingerprint"])
                key = self._key_from_json(json.loads(str(row["key_json"])))
                new_fingerprint = key.fingerprint
                if old_fingerprint == new_fingerprint:
                    continue
                collision = self.connection.execute(
                    "SELECT spot_json FROM solutions WHERE fingerprint = ?",
                    (new_fingerprint,),
                ).fetchone()
                if collision is not None:
                    if str(collision["spot_json"]) != str(row["spot_json"]):
                        raise ValueError(
                            "Suit-isomorphic solution collision contains different strategy payloads"
                        )
                    self.connection.execute(
                        "DELETE FROM solutions WHERE fingerprint = ?", (old_fingerprint,)
                    )
                else:
                    self.connection.execute(
                        """
                        UPDATE solutions SET fingerprint = ?, imported_at = CURRENT_TIMESTAMP
                        WHERE fingerprint = ?
                        """,
                        (new_fingerprint, old_fingerprint),
                    )
            self.connection.commit()

    def import_text(self, source: str) -> ImportResult:
        hands = PokerStarsParser().parse_many(source)
        inserted = updated = unchanged = 0
        with self._lock:
            for hand in hands:
                normalized_source = hand.source.replace("\r\n", "\n").replace("\r", "\n").strip()
                row = self.connection.execute(
                    "SELECT source FROM hand_histories WHERE hand_id = ?", (hand.hand_id,)
                ).fetchone()
                if row is None:
                    inserted += 1
                elif str(row["source"]) == normalized_source:
                    unchanged += 1
                    continue
                else:
                    updated += 1
                self.connection.execute(
                    """
                    INSERT INTO hand_histories(hand_id, source, table_name, played_at, hero)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(hand_id) DO UPDATE SET
                        source = excluded.source,
                        table_name = excluded.table_name,
                        played_at = excluded.played_at,
                        hero = excluded.hero,
                        imported_at = CURRENT_TIMESTAMP
                    """,
                    (hand.hand_id, normalized_source, hand.table_name, hand.played_at_raw, hand.hero),
                )
            self.connection.commit()
        return ImportResult(inserted, updated, unchanged, tuple(hand.hand_id for hand in hands))

    def import_file(self, path: str | Path) -> ImportResult:
        return self.import_text(Path(path).read_text(encoding="utf-8-sig"))

    def load_hands(self) -> tuple[HandHistory, ...]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT source FROM hand_histories ORDER BY played_at, hand_id"
            ).fetchall()
        parser = PokerStarsParser()
        return tuple(parser.parse(str(row["source"])) for row in rows)

    def analyze(self) -> dict[str, object]:
        return analyze_hands(self.load_hands())

    @property
    def hand_count(self) -> int:
        with self._lock:
            row = self.connection.execute("SELECT COUNT(*) AS count FROM hand_histories").fetchone()
        return int(row["count"])

    def ingested_file_state(self, path: str | Path) -> IngestedFileState | None:
        normalized = str(Path(path).resolve())
        with self._lock:
            row = self.connection.execute(
                """
                SELECT path, size, modified_ns, completed_blocks, last_error
                FROM ingested_files WHERE path = ?
                """,
                (normalized,),
            ).fetchone()
        if row is None:
            return None
        return IngestedFileState(
            path=str(row["path"]),
            size=int(row["size"]),
            modified_ns=int(row["modified_ns"]),
            completed_blocks=int(row["completed_blocks"]),
            last_error=None if row["last_error"] is None else str(row["last_error"]),
        )

    def record_ingested_file(
        self,
        path: str | Path,
        *,
        size: int,
        modified_ns: int,
        completed_blocks: int,
        last_error: str | None = None,
    ) -> None:
        normalized = str(Path(path).resolve())
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO ingested_files(
                    path, size, modified_ns, completed_blocks, last_error
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size = excluded.size,
                    modified_ns = excluded.modified_ns,
                    completed_blocks = excluded.completed_blocks,
                    last_error = excluded.last_error,
                    scanned_at = CURRENT_TIMESTAMP
                """,
                (normalized, size, modified_ns, completed_blocks, last_error),
            )
            self.connection.commit()

    @property
    def ingested_file_count(self) -> int:
        with self._lock:
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM ingested_files"
            ).fetchone()
        return int(row["count"])

    def put(self, spot: SolvedSpot) -> None:
        spot_data = {
            "actions": [
                {
                    "action": action.action,
                    "frequency": format(action.frequency, "f"),
                    "ev": format(action.ev, "f"),
                }
                for action in spot.actions
            ],
            "source": spot.source,
            "source_version": spot.source_version,
            "node_id": spot.node_id,
        }
        self.connection.execute(
            """
            INSERT INTO solutions(fingerprint, key_json, spot_json)
            VALUES (?, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET
                key_json = excluded.key_json,
                spot_json = excluded.spot_json,
                imported_at = CURRENT_TIMESTAMP
            """,
            (
                spot.key.fingerprint,
                json.dumps(spot.key.canonical(), sort_keys=True),
                json.dumps(spot_data, sort_keys=True),
            ),
        )
        self.connection.commit()

    def get(self, key: SolutionKey) -> SolvedSpot | None:
        row = self.connection.execute(
            "SELECT key_json, spot_json FROM solutions WHERE fingerprint = ?",
            (key.fingerprint,),
        ).fetchone()
        if row is None:
            return None
        stored_key = self._key_from_json(json.loads(str(row["key_json"])))
        if stored_key.fingerprint != key.fingerprint:
            raise ValueError("Stored solution key fingerprint mismatch")
        spot_data = json.loads(str(row["spot_json"]))
        return SolvedSpot(
            key=stored_key,
            actions=tuple(
                ActionSolution(
                    str(action["action"]),
                    Decimal(str(action["frequency"])),
                    Decimal(str(action["ev"])),
                )
                for action in spot_data["actions"]
            ),
            source=str(spot_data["source"]),
            source_version=str(spot_data.get("source_version", "")),
            node_id=str(spot_data.get("node_id", "")),
        )

    def list_solutions(self) -> tuple[SolvedSpot, ...]:
        rows = self.connection.execute(
            "SELECT key_json, spot_json FROM solutions ORDER BY fingerprint"
        ).fetchall()
        spots: list[SolvedSpot] = []
        for row in rows:
            key = self._key_from_json(json.loads(str(row["key_json"])))
            spot_data = json.loads(str(row["spot_json"]))
            spots.append(
                SolvedSpot(
                    key=key,
                    actions=tuple(
                        ActionSolution(
                            str(action["action"]),
                            Decimal(str(action["frequency"])),
                            Decimal(str(action["ev"])),
                        )
                        for action in spot_data["actions"]
                    ),
                    source=str(spot_data["source"]),
                    source_version=str(spot_data.get("source_version", "")),
                    node_id=str(spot_data.get("node_id", "")),
                )
            )
        return tuple(spots)

    def upsert_drills(self, drills: Iterable[dict[str, object]]) -> tuple[int, int]:
        inserted = updated = 0
        with self._lock:
            for raw in drills:
                drill = dict(raw)
                drill_id = str(drill.get("drill_id", ""))
                if not drill_id:
                    raise ValueError("Drill payload requires drill_id")
                exists = self.connection.execute(
                    "SELECT 1 FROM drills WHERE drill_id = ?", (drill_id,)
                ).fetchone()
                if exists:
                    updated += 1
                else:
                    inserted += 1
                initial_due = utc_now().isoformat()
                self.connection.execute(
                    """
                    INSERT INTO drills(drill_id, payload_json, priority, due_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(drill_id) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        priority = excluded.priority,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        drill_id,
                        json.dumps(drill, sort_keys=True),
                        str(drill.get("priority", "0")),
                        initial_due,
                    ),
                )
            self.connection.commit()
        return inserted, updated

    def list_drills(
        self,
        *,
        due_only: bool = False,
        now: datetime | None = None,
    ) -> list[dict[str, object]]:
        current = now or utc_now()
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT drill_id, payload_json, attempts, streak, mastery,
                       interval_days, due_at, last_rating
                FROM drills
                ORDER BY due_at, CAST(priority AS REAL) DESC, drill_id
                """
            ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            due_at = datetime.fromisoformat(str(row["due_at"]))
            if due_only and due_at > current:
                continue
            rating = None if row["last_rating"] is None else ReviewRating(str(row["last_rating"]))
            state = StudyState(
                attempts=int(row["attempts"]),
                streak=int(row["streak"]),
                mastery=Decimal(str(row["mastery"])),
                interval_days=int(row["interval_days"]),
                due_at=due_at,
                last_rating=rating,
            )
            payload = json.loads(str(row["payload_json"]))
            payload["study"] = state.to_dict()
            result.append(payload)
        return result

    def review_drill(
        self,
        drill_id: str,
        rating: ReviewRating | str,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        reviewed_at = now or utc_now()
        with self._lock:
            row = self.connection.execute(
                """
                SELECT payload_json, attempts, streak, mastery, interval_days,
                       due_at, last_rating
                FROM drills WHERE drill_id = ?
                """,
                (drill_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown drill: {drill_id}")
            current = StudyState(
                attempts=int(row["attempts"]),
                streak=int(row["streak"]),
                mastery=Decimal(str(row["mastery"])),
                interval_days=int(row["interval_days"]),
                due_at=datetime.fromisoformat(str(row["due_at"])),
                last_rating=None if row["last_rating"] is None else ReviewRating(str(row["last_rating"])),
            )
            state = schedule_review(current, rating, now=reviewed_at)
            state_json = json.dumps(state.to_dict(), sort_keys=True)
            self.connection.execute(
                """
                UPDATE drills SET attempts = ?, streak = ?, mastery = ?,
                    interval_days = ?, due_at = ?, last_rating = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE drill_id = ?
                """,
                (
                    state.attempts,
                    state.streak,
                    format(state.mastery, "f"),
                    state.interval_days,
                    state.due_at.astimezone(timezone.utc).isoformat(),
                    state.last_rating.value if state.last_rating else None,
                    drill_id,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO drill_attempts(drill_id, rating, reviewed_at, resulting_state_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    drill_id,
                    state.last_rating.value if state.last_rating else "",
                    reviewed_at.astimezone(timezone.utc).isoformat(),
                    state_json,
                ),
            )
            self.connection.commit()
            payload = json.loads(str(row["payload_json"]))
            payload["study"] = state.to_dict()
            return payload

    @property
    def drill_attempt_count(self) -> int:
        with self._lock:
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM drill_attempts"
            ).fetchone()
            return int(row["count"])

    @staticmethod
    def _key_from_json(data: dict[str, object]) -> SolutionKey:
        return SolutionKey(
            game=str(data["game"]),
            players=int(data["players"]),
            hero_position=str(data["hero_position"]),
            effective_stack_bb=Decimal(str(data["effective_stack_bb"])),
            pot_bb=Decimal(str(data["pot_bb"])),
            board=tuple(Card.parse(str(card)) for card in data["board"]),
            action_history=tuple(str(action) for action in data["action_history"]),
            rake_model=str(data["rake_model"]),
            utility_model=str(data["utility_model"]),
            allowed_sizes=tuple(Decimal(str(size)) for size in data["allowed_sizes"]),
            hero_cards=tuple(Card.parse(str(card)) for card in data.get("hero_cards", [])),
        )
