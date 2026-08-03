from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .pokerstars import PokerStarsParseError
from .storage import CoachDatabase


_HAND_START_RE = re.compile(r"(?=^PokerStars (?:Hand|Game) #)", re.MULTILINE)
_SUMMARY_RE = re.compile(r"^\*\*\* SUMMARY \*\*\*\r?$", re.MULTILINE)
DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024


def split_completed_hands(source: str) -> tuple[tuple[str, ...], int]:
    """Return complete PokerStars blocks and the number still being written.

    PokerStars appends hands to text files. A header may therefore be visible
    before its final summary. Only blocks containing the summary marker are
    eligible for import; an incomplete tail is retried after the file changes.
    """

    blocks = tuple(
        block.strip()
        for block in _HAND_START_RE.split(source)
        if block.strip().lower().startswith(("pokerstars hand #", "pokerstars game #"))
    )
    complete = tuple(block for block in blocks if _SUMMARY_RE.search(block))
    return complete, len(blocks) - len(complete)


def _read_history_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unsupported text encoding: {path}")


@dataclass(frozen=True)
class FileScanError:
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


@dataclass
class FolderScanResult:
    folder: str
    recursive: bool
    files_seen: int = 0
    changed_files: int = 0
    skipped_files: int = 0
    unstable_files: int = 0
    completed_blocks: int = 0
    incomplete_blocks: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    hand_ids: list[str] = field(default_factory=list)
    errors: list[FileScanError] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "folder": self.folder,
            "recursive": self.recursive,
            "files_seen": self.files_seen,
            "changed_files": self.changed_files,
            "skipped_files": self.skipped_files,
            "unstable_files": self.unstable_files,
            "completed_blocks": self.completed_blocks,
            "incomplete_blocks": self.incomplete_blocks,
            "inserted": self.inserted,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "hand_ids": self.hand_ids,
            "errors": [error.to_dict() for error in self.errors],
        }


class HandHistoryFolderScanner:
    def __init__(
        self,
        database: CoachDatabase,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        self.database = database
        self.max_file_bytes = max_file_bytes

    def scan(self, folder: str | Path, *, recursive: bool = False) -> FolderScanResult:
        root = Path(folder).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"Hand-history folder does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Hand-history path is not a folder: {root}")

        result = FolderScanResult(str(root), recursive)
        iterator = root.rglob("*") if recursive else root.iterdir()
        candidates = sorted(
            (path for path in iterator if path.is_file() and path.suffix.lower() == ".txt"),
            key=lambda path: str(path).lower(),
        )
        result.files_seen = len(candidates)
        for path in candidates:
            self._scan_file(path, result)
        return result

    def _scan_file(self, path: Path, result: FolderScanResult) -> None:
        before = path.stat()
        state = self.database.ingested_file_state(path)
        if (
            state is not None
            and state.size == before.st_size
            and state.modified_ns == before.st_mtime_ns
        ):
            result.skipped_files += 1
            return

        result.changed_files += 1
        if before.st_size > self.max_file_bytes:
            message = f"File exceeds {self.max_file_bytes} byte scan limit"
            result.errors.append(FileScanError(str(path), message))
            self.database.record_ingested_file(
                path,
                size=before.st_size,
                modified_ns=before.st_mtime_ns,
                completed_blocks=0,
                last_error=message,
            )
            return

        completed_count = 0
        error_messages: list[str] = []
        try:
            source = _read_history_file(path)
            blocks, incomplete = split_completed_hands(source)
            result.incomplete_blocks += incomplete
            result.completed_blocks += len(blocks)
            for block in blocks:
                try:
                    imported = self.database.import_text(block)
                except (PokerStarsParseError, ValueError) as error:
                    error_messages.append(str(error))
                    continue
                completed_count += 1
                result.inserted += imported.inserted
                result.updated += imported.updated
                result.unchanged += imported.unchanged
                result.hand_ids.extend(imported.hand_ids)
        except (OSError, UnicodeError) as error:
            error_messages.append(str(error))

        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            result.unstable_files += 1
        else:
            self.database.record_ingested_file(
                path,
                size=after.st_size,
                modified_ns=after.st_mtime_ns,
                completed_blocks=completed_count,
                last_error="; ".join(error_messages) or None,
            )
        result.errors.extend(FileScanError(str(path), message) for message in error_messages)
