from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from apc.annotator import AnnotationProject
from apc.tools.validate_dataset import canonical_sha256, validate_manifest


def _source_namespace(dataset_id: str) -> str:
    return f"d{canonical_sha256(dataset_id)[:12]}"


def _load_source(manifest_path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    resolved = manifest_path.expanduser().resolve()
    report = validate_manifest(resolved, require_images=True)
    if not report["valid"]:
        raise ValueError(
            f"Source manifest is invalid ({resolved}): " + "; ".join(report["errors"])
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    payload["_manifest_path"] = str(resolved)
    return payload, report


def _source_rows(source: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    manifest_path = Path(source["_manifest_path"])
    rows: list[tuple[Path, dict[str, Any]]] = []
    for relative in source["annotation_files"]:
        path = (manifest_path.parent / str(relative)).resolve()
        annotation = json.loads(path.read_text(encoding="utf-8"))
        rows.append((path, annotation))
    return sorted(
        rows,
        key=lambda row: (
            str(row[1]["capture_session_id"]),
            int(row[1]["image"]["timestamp_ms"]),
            int(row[1]["sequence_index"]),
            str(row[1]["sample_id"]),
        ),
    )


def merge_dataset_manifests(
    output: str | Path,
    manifests: Iterable[str | Path],
    *,
    dataset_id: str,
    dataset_version: str,
) -> dict[str, object]:
    manifest_paths = [Path(path) for path in manifests]
    if len(manifest_paths) < 2:
        raise ValueError("At least two source manifests are required")
    sources = [_load_source(path) for path in manifest_paths]
    source_ids = [str(payload["dataset_id"]) for payload, _ in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Source dataset_id values must be unique")
    ordered_sources = sorted(sources, key=lambda row: str(row[0]["dataset_id"]))

    project = AnnotationProject.create(
        output,
        project_id=dataset_id,
        source_kind="synthetic_render",
        provider_id="apc-dataset-merger-v1",
        layout_id="mixed",
        theme_id="mixed",
        locale="en-US",
        max_seats=10,
    )
    source_evidence: list[dict[str, object]] = []
    merged_rows: list[dict[str, object]] = []
    for source, source_report in ordered_sources:
        source_id = str(source["dataset_id"])
        namespace = _source_namespace(source_id)
        source_evidence.append(
            {
                "dataset_id": source_id,
                "dataset_version": source["dataset_version"],
                "namespace": namespace,
                "manifest_path": source["_manifest_path"],
                "annotations_sha256": source_report["computed_fingerprints"][
                    "annotations_sha256"
                ],
                "split_sha256": source_report["computed_fingerprints"]["split_sha256"],
                "frames": source_report["computed_statistics"]["verified_frames"],
            }
        )
        for annotation_path, original in _source_rows(source):
            original_session = str(original["capture_session_id"])
            merged_session = f"{namespace}-{original_session}"
            image_path = (annotation_path.parent / original["image"]["path"]).resolve()
            record, inserted = project.import_frame(
                image_path,
                capture_session_id=merged_session,
                timestamp_ms=int(original["image"]["timestamp_ms"]),
                environment=dict(original["environment"]),
            )
            if not inserted:
                raise ValueError(
                    "Identical source-frame digest occurs more than once across merged datasets"
                )
            annotation = copy.deepcopy(original)
            template = project.annotation_template(record.sample_id)
            annotation["sample_id"] = template["sample_id"]
            annotation["capture_session_id"] = template["capture_session_id"]
            annotation["sequence_index"] = template["sequence_index"]
            annotation["image"] = template["image"]
            annotation["environment"] = template["environment"]
            annotation["state"]["table_id"] = (
                f"{namespace}:{annotation['state']['table_id']}"
            )
            annotation["state"]["hand_id"] = (
                f"{namespace}:{annotation['state']['hand_id']}"
            )
            provenance = annotation["provenance"]
            source_marker = (
                f"merged_from={source_id}/{original['sample_id']}; "
                f"source_annotation_sha256={canonical_sha256(original)}"
            )
            notes = str(provenance.get("notes", ""))
            provenance["notes"] = f"{notes}{' ' if notes else ''}{source_marker}"
            project.save_annotation(record.sample_id, annotation)
            merged_rows.append(
                {
                    "source_dataset_id": source_id,
                    "source_sample_id": original["sample_id"],
                    "merged_sample_id": record.sample_id,
                    "merged_session_id": merged_session,
                }
            )

    manifest_path, validation = project.export_manifest(
        dataset_version=dataset_version
    )
    if not validation["valid"]:
        raise RuntimeError("Merged manifest failed post-export validation")
    return {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "manifest": str(manifest_path),
        "project": project.status(),
        "source_datasets": source_evidence,
        "source_evidence_sha256": canonical_sha256(source_evidence),
        "merged_rows_sha256": canonical_sha256(merged_rows),
        "validation": validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge valid APC manifests into one namespaced, group-exclusive dataset."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", default="0.1.0")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = merge_dataset_manifests(
            args.output,
            args.manifests,
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
        )
        rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.report:
            report = args.report.expanduser().resolve()
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
