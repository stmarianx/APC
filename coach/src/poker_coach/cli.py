from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pokerstars import PokerStarsParseError, PokerStarsParser
from .report import analyze_hands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze saved PokerStars text hand histories offline.")
    parser.add_argument("path", type=Path, help="Hand-history text file or folder")
    parser.add_argument("--database", type=Path, help="Persist imported hands in this SQLite database and analyze the full database")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan nested folders (folder input only)",
    )
    parser.add_argument(
        "--profiles-only",
        action="store_true",
        help="Emit a compact BB-only player-profile snapshot instead of full hand reports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this file instead of standard output",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    return parser


def build_profile_snapshot(
    report: dict[str, Any],
    *,
    source_path: Path,
    database_path: Path | None,
) -> dict[str, object]:
    """Build a compact, provenance-aware opponent map from an analysis report."""

    summaries = report.get("profile_summaries", {})
    estimates = report.get("player_profiles", {})
    insights = report.get("exploit_insights", {})
    players = {
        player: {
            "summary": summaries[player],
            "estimates": estimates.get(player, []),
            "exploit_insights": insights.get(player, []),
        }
        for player in sorted(summaries)
    }
    source: dict[str, object] = {
        "hand_history_path": str(source_path.resolve()),
        "hands": int(report.get("hands", 0)),
    }
    if database_path is not None:
        source["database"] = str(database_path.resolve())
    if "scan" in report:
        source["scan"] = report["scan"]
    if "import" in report:
        source["import"] = report["import"]
    return {
        "schema_version": "1.0.0",
        "units": "BB",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "players": players,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.path.exists():
            raise ValueError(f"Hand-history path does not exist: {args.path}")
        if args.path.is_dir():
            if args.database is None:
                raise ValueError("--database is required when scanning a hand-history folder")
            from .ingest import HandHistoryFolderScanner
            from .storage import CoachDatabase

            with CoachDatabase(args.database) as database:
                scan = HandHistoryFolderScanner(database).scan(
                    args.path, recursive=args.recursive
                )
                report = database.analyze()
                report["scan"] = scan.to_dict()
        elif args.recursive:
            raise ValueError("--recursive can only be used with a folder input")
        elif args.database:
            from .storage import CoachDatabase

            with CoachDatabase(args.database) as database:
                imported = database.import_file(args.path)
                report = database.analyze()
                report["import"] = {
                    "inserted": imported.inserted,
                    "updated": imported.updated,
                    "unchanged": imported.unchanged,
                    "hand_ids": list(imported.hand_ids),
                    "database_hands": database.hand_count,
                }
        else:
            hands = PokerStarsParser().parse_file(args.path)
            report = analyze_hands(hands)
        if args.profiles_only:
            report = build_profile_snapshot(
                report,
                source_path=args.path,
                database_path=args.database,
            )
    except (OSError, PokerStarsParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=args.indent, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
