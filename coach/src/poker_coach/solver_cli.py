from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .solver_adapters import SUPPORTED_EXPORT_FORMATS, SolverExportRegistry
from .solver_import import SolverBundleImporter
from .storage import CoachDatabase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a solver JSON bundle or tabular CSV export")
    parser.add_argument("bundle", type=Path, help="Path to solver export (.json or .csv)")
    parser.add_argument("--database", type=Path, default=Path("coach.sqlite3"))
    parser.add_argument(
        "--format",
        default="auto",
        choices=("auto",) + SUPPORTED_EXPORT_FORMATS,
        help="Export format; auto detects from extension/content",
    )
    args = parser.parse_args(argv)
    importer = SolverBundleImporter()
    try:
        parsed = SolverExportRegistry(importer).parse_file(
            args.bundle, format_name=args.format
        )
        with CoachDatabase(args.database) as database:
            result = importer.import_into(database, parsed.bundle)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    payload = result.to_dict()
    payload["format"] = parsed.format_name
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
