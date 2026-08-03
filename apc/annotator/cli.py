from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .project import AnnotationProject


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and manage an APC visible-table annotation project.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a new annotation project")
    init.add_argument("project", type=Path)
    init.add_argument("--project-id", required=True)
    init.add_argument("--source-kind", default="controlled_training_table")
    init.add_argument("--provider", required=True)
    init.add_argument("--layout", required=True)
    init.add_argument("--theme", required=True)
    init.add_argument("--locale", default="en-US")
    init.add_argument("--max-seats", type=int, required=True)

    add = sub.add_parser("import", help="Import immutable PNG frames")
    add.add_argument("project", type=Path)
    add.add_argument("frames", type=Path, nargs="+")
    add.add_argument("--session", required=True)
    add.add_argument("--timestamp-ms", type=int, default=0)

    add_folder = sub.add_parser("import-folder", help="Import an ordered capture folder as one session")
    add_folder.add_argument("project", type=Path)
    add_folder.add_argument("folder", type=Path)
    add_folder.add_argument("--session", required=True)
    add_folder.add_argument("--timestamp-ms", type=int, default=0)
    add_folder.add_argument("--interval-ms", type=int, default=100)
    add_folder.add_argument("--pattern", default="*.png")
    add_folder.add_argument("--recursive", action="store_true")
    add_folder.add_argument("--sample-every", type=int, default=1)

    template = sub.add_parser("template", help="Print or write the annotation template for a frame")
    template.add_argument("project", type=Path)
    template.add_argument("sample_id")
    template.add_argument("--output", type=Path)

    save = sub.add_parser("save", help="Validate and save a completed annotation JSON file")
    save.add_argument("project", type=Path)
    save.add_argument("sample_id")
    save.add_argument("annotation", type=Path)

    status = sub.add_parser("status", help="Report project progress")
    status.add_argument("project", type=Path)

    export = sub.add_parser("export", help="Export and validate a grouped dataset manifest")
    export.add_argument("project", type=Path)
    export.add_argument("--version", required=True)
    export.add_argument("--output", type=Path)
    return parser


def _emit(payload: object, output: Path | None = None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            project = AnnotationProject.create(
                args.project,
                project_id=args.project_id,
                source_kind=args.source_kind,
                provider_id=args.provider,
                layout_id=args.layout,
                theme_id=args.theme,
                locale=args.locale,
                max_seats=args.max_seats,
            )
            _emit(project.status())
        elif args.command == "import":
            project = AnnotationProject(args.project)
            rows = []
            for offset, frame in enumerate(args.frames):
                record, inserted = project.import_frame(
                    frame,
                    capture_session_id=args.session,
                    timestamp_ms=args.timestamp_ms + offset,
                )
                rows.append({**record.to_dict(), "inserted": inserted})
            _emit({"frames": rows, "status": project.status()})
        elif args.command == "import-folder":
            project = AnnotationProject(args.project)
            _emit(
                project.import_folder(
                    args.folder,
                    capture_session_id=args.session,
                    timestamp_ms=args.timestamp_ms,
                    interval_ms=args.interval_ms,
                    pattern=args.pattern,
                    recursive=args.recursive,
                    sample_every=args.sample_every,
                )
            )
        elif args.command == "template":
            project = AnnotationProject(args.project)
            _emit(project.annotation_template(args.sample_id), args.output)
        elif args.command == "save":
            project = AnnotationProject(args.project)
            payload = json.loads(args.annotation.read_text(encoding="utf-8"))
            path = project.save_annotation(args.sample_id, payload)
            _emit({"saved": str(path), "status": project.status()})
        elif args.command == "status":
            _emit(AnnotationProject(args.project).status())
        elif args.command == "export":
            project = AnnotationProject(args.project)
            path, report = project.export_manifest(
                dataset_version=args.version,
                output=args.output,
            )
            _emit({"manifest": str(path), "validation": report})
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
