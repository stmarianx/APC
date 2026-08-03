from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from apc.tools.validate_dataset import canonical_sha256, validate_annotation, validate_manifest


PROJECT_SCHEMA_VERSION = "1.0.0"
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _natural_path_key(path: Path) -> tuple[tuple[int, object], ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", path.as_posix())
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str, name: str) -> str:
    cleaned = str(value).strip()
    if not SAFE_ID.fullmatch(cleaned):
        raise ValueError(f"{name} must contain only letters, numbers, dot, underscore or hyphen")
    return cleaned


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Only PNG screenshots with a valid IHDR header are supported: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width < 320 or height < 240:
        raise ValueError("APC frames must be at least 320x240 pixels")
    return width, height


@dataclass(frozen=True)
class FrameRecord:
    sample_id: str
    capture_session_id: str
    sequence_index: int
    frame_path: str
    sha256: str
    width: int
    height: int
    timestamp_ms: int
    imported_at: str
    environment: dict[str, object] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrameRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            capture_session_id=str(payload["capture_session_id"]),
            sequence_index=int(payload["sequence_index"]),
            frame_path=str(payload["frame_path"]),
            sha256=str(payload["sha256"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            timestamp_ms=int(payload["timestamp_ms"]),
            imported_at=str(payload["imported_at"]),
            environment=(
                dict(payload["environment"])
                if isinstance(payload.get("environment"), dict)
                else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "sample_id": self.sample_id,
            "capture_session_id": self.capture_session_id,
            "sequence_index": self.sequence_index,
            "frame_path": self.frame_path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "timestamp_ms": self.timestamp_ms,
            "imported_at": self.imported_at,
        }
        if self.environment is not None:
            payload["environment"] = dict(self.environment)
        return payload


class AnnotationProject:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.project_path = self.root / "project.json"
        if not self.project_path.is_file():
            raise ValueError(f"APC annotation project is not initialized: {self.root}")
        self.config = _read_json(self.project_path)
        if self.config.get("schema_version") != PROJECT_SCHEMA_VERSION:
            raise ValueError("Unsupported APC annotation project schema")

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        project_id: str,
        source_kind: str,
        provider_id: str,
        layout_id: str,
        theme_id: str,
        locale: str,
        max_seats: int,
    ) -> "AnnotationProject":
        target = Path(root).expanduser().resolve()
        project_path = target / "project.json"
        if project_path.exists():
            raise ValueError(f"APC annotation project already exists: {target}")
        if source_kind not in {
            "controlled_training_table",
            "explicitly_permitted_virtual_table",
            "synthetic_render",
        }:
            raise ValueError("source_kind must be an allowed controlled virtual-chip source")
        if not 2 <= int(max_seats) <= 10:
            raise ValueError("max_seats must be between 2 and 10")
        config = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": _safe_id(project_id, "project_id"),
            "created_at": utc_now(),
            "environment": {
                "source_kind": source_kind,
                "provider_id": _safe_id(provider_id, "provider_id"),
                "layout_id": _safe_id(layout_id, "layout_id"),
                "theme_id": _safe_id(theme_id, "theme_id"),
                "locale": str(locale).strip(),
                "max_seats": int(max_seats),
                "virtual_chips": True,
            },
            "records": [],
        }
        target.mkdir(parents=True, exist_ok=True)
        (target / "frames").mkdir(exist_ok=True)
        (target / "annotations").mkdir(exist_ok=True)
        (target / "suggestions").mkdir(exist_ok=True)
        _write_json(project_path, config)
        return cls(target)

    @property
    def records(self) -> tuple[FrameRecord, ...]:
        return tuple(FrameRecord.from_dict(row) for row in self.config.get("records", []))

    def _persist(self) -> None:
        _write_json(self.project_path, self.config)

    def import_frame(
        self,
        source: str | Path,
        *,
        capture_session_id: str,
        timestamp_ms: int,
        environment: dict[str, object] | None = None,
    ) -> tuple[FrameRecord, bool]:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Frame does not exist: {path}")
        if path.suffix.lower() != ".png":
            raise ValueError("Only PNG screenshots are supported in APC dataset v1")
        session = _safe_id(capture_session_id, "capture_session_id")
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be nonnegative")
        frame_environment = (
            dict(self.config["environment"])
            if environment is None
            else dict(environment)
        )
        required_environment = {
            "source_kind",
            "provider_id",
            "layout_id",
            "theme_id",
            "locale",
            "max_seats",
            "virtual_chips",
        }
        if set(frame_environment) != required_environment:
            raise ValueError(
                "Frame environment must contain exactly: "
                + ", ".join(sorted(required_environment))
            )
        if frame_environment["source_kind"] not in {
            "controlled_training_table",
            "explicitly_permitted_virtual_table",
            "synthetic_render",
        } or frame_environment["virtual_chips"] is not True:
            raise ValueError("Frame environment must be an allowed virtual-chip source")
        if not 2 <= int(frame_environment["max_seats"]) <= 10:
            raise ValueError("Frame environment max_seats must be between 2 and 10")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for record in self.records:
            if record.sha256 == digest:
                return record, False
        session_records = [
            record for record in self.records if record.capture_session_id == session
        ]
        if session_records and timestamp_ms <= max(record.timestamp_ms for record in session_records):
            raise ValueError(
                "A new frame timestamp must advance within its capture session"
            )
        width, height = png_size(path)
        sequence_index = 1 + max(
            (
                record.sequence_index
                for record in self.records
                if record.capture_session_id == session
            ),
            default=-1,
        )
        sample_id = _safe_id(
            f"{session}-{sequence_index:06d}-{digest[:12]}", "sample_id"
        )
        relative = Path("frames") / session / f"{sample_id}.png"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        record = FrameRecord(
            sample_id,
            session,
            sequence_index,
            relative.as_posix(),
            digest,
            width,
            height,
            int(timestamp_ms),
            utc_now(),
            frame_environment,
        )
        self.config["records"].append(record.to_dict())
        self._persist()
        return record, True

    def import_folder(
        self,
        source: str | Path,
        *,
        capture_session_id: str,
        timestamp_ms: int = 0,
        interval_ms: int = 100,
        pattern: str = "*.png",
        recursive: bool = False,
        sample_every: int = 1,
    ) -> dict[str, object]:
        folder = Path(source).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"Capture folder does not exist: {folder}")
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be nonnegative")
        if isinstance(interval_ms, bool) or interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if isinstance(sample_every, bool) or sample_every <= 0:
            raise ValueError("sample_every must be positive")
        if not pattern.strip() or Path(pattern).name != pattern:
            raise ValueError("pattern must be a filename glob without directories")
        iterator = folder.rglob(pattern) if recursive else folder.glob(pattern)
        discovered = sorted((path for path in iterator if path.is_file()), key=_natural_path_key)
        selected = discovered[::sample_every]
        if not selected:
            raise ValueError(f"Capture folder contains no matching frames: {folder}")
        rows: list[dict[str, object]] = []
        for offset, frame in enumerate(selected):
            record, inserted = self.import_frame(
                frame,
                capture_session_id=capture_session_id,
                timestamp_ms=timestamp_ms + offset * interval_ms,
            )
            rows.append(
                {
                    "source": str(frame),
                    "sample_id": record.sample_id,
                    "sequence_index": record.sequence_index,
                    "timestamp_ms": record.timestamp_ms,
                    "sha256": record.sha256,
                    "inserted": inserted,
                }
            )
        return {
            "source_folder": str(folder),
            "capture_session_id": _safe_id(capture_session_id, "capture_session_id"),
            "pattern": pattern,
            "recursive": recursive,
            "sample_every": sample_every,
            "interval_ms": interval_ms,
            "discovered_frames": len(discovered),
            "selected_frames": len(selected),
            "inserted_frames": sum(bool(row["inserted"]) for row in rows),
            "duplicate_frames": sum(not bool(row["inserted"]) for row in rows),
            "frames": rows,
            "status": self.status(),
        }

    def annotation_template(self, sample_id: str) -> dict[str, object]:
        record = self.record(sample_id)
        return {
            "schema_version": "1.0.0",
            "sample_id": record.sample_id,
            "capture_session_id": record.capture_session_id,
            "sequence_index": record.sequence_index,
            "image": {
                "path": f"../{record.frame_path}",
                "sha256": record.sha256,
                "width": record.width,
                "height": record.height,
                "timestamp_ms": record.timestamp_ms,
            },
            "environment": dict(record.environment or self.config["environment"]),
        }

    def record(self, sample_id: str) -> FrameRecord:
        clean = _safe_id(sample_id, "sample_id")
        for record in self.records:
            if record.sample_id == clean:
                return record
        raise KeyError(f"Unknown APC sample: {clean}")

    def annotation_path(self, sample_id: str) -> Path:
        return self.root / "annotations" / f"{_safe_id(sample_id, 'sample_id')}.json"

    def suggestion_path(self, sample_id: str) -> Path:
        return self.root / "suggestions" / f"{_safe_id(sample_id, 'sample_id')}.json"

    def _validated_suggestion(
        self, sample_id: str, payload: dict[str, object]
    ) -> dict[str, object]:
        record = self.record(sample_id)
        material = dict(payload)
        supplied_fingerprint = material.pop("suggestion_sha256", None)
        if material.get("schema_version") != "1.0.0":
            raise ValueError("Unsupported APC suggestion schema")
        if material.get("kind") != "apc_perception_suggestion":
            raise ValueError("Suggestion kind must be apc_perception_suggestion")
        if material.get("sample_id") != record.sample_id:
            raise ValueError("Suggestion sample_id does not match the frame record")
        if material.get("capture_session_id") != record.capture_session_id:
            raise ValueError("Suggestion capture_session_id does not match the frame record")
        image = material.get("image")
        if not isinstance(image, dict) or image.get("sha256") != record.sha256:
            raise ValueError("Suggestion image fingerprint does not match the frame record")
        if material.get("review_required") is not True or material.get("auto_applied") is not False:
            raise ValueError("Suggestions must require review and cannot be auto-applied")
        actual_fingerprint = canonical_sha256(material)
        if supplied_fingerprint is not None and supplied_fingerprint != actual_fingerprint:
            raise ValueError("Suggestion fingerprint does not match its contents")
        material["suggestion_sha256"] = actual_fingerprint
        return material

    def save_suggestion(self, sample_id: str, payload: dict[str, object]) -> Path:
        material = self._validated_suggestion(sample_id, payload)
        path = self.suggestion_path(sample_id)
        _write_json(path, material)
        return path

    def load_suggestion(self, sample_id: str) -> dict[str, Any] | None:
        path = self.suggestion_path(sample_id)
        if not path.is_file():
            return None
        payload = _read_json(path)
        return self._validated_suggestion(sample_id, payload)

    def save_annotation(self, sample_id: str, payload: dict[str, object]) -> Path:
        record = self.record(sample_id)
        if payload.get("sample_id") != record.sample_id:
            raise ValueError("Annotation sample_id does not match the frame record")
        if payload.get("capture_session_id") != record.capture_session_id:
            raise ValueError("Annotation capture_session_id does not match the frame record")
        path = self.annotation_path(sample_id)
        issues = validate_annotation(payload, annotation_path=path, require_image=True)
        if issues:
            raise ValueError("Invalid APC annotation: " + "; ".join(issues))
        _write_json(path, payload)
        return path

    def load_annotation(self, sample_id: str) -> dict[str, Any] | None:
        path = self.annotation_path(sample_id)
        return _read_json(path) if path.is_file() else None

    @staticmethod
    def _split_sessions(session_ids: Iterable[str]) -> dict[str, list[str]]:
        sessions = sorted(
            set(session_ids),
            key=lambda session: hashlib.sha256(
                f"apc-visual-split-v1:{session}".encode("utf-8")
            ).hexdigest(),
        )
        if len(sessions) < 3:
            raise ValueError("At least three capture sessions are required to export train, validation and test splits")
        test_count = max(1, round(len(sessions) * 0.10))
        validation_count = max(1, round(len(sessions) * 0.10))
        if test_count + validation_count >= len(sessions):
            test_count = validation_count = 1
        return {
            "train": sorted(sessions[test_count + validation_count :]),
            "validation": sorted(sessions[test_count : test_count + validation_count]),
            "test": sorted(sessions[:test_count]),
        }

    def export_manifest(
        self,
        *,
        dataset_version: str,
        output: str | Path | None = None,
        require_verified: bool = True,
    ) -> tuple[Path, dict[str, object]]:
        annotations: list[dict[str, Any]] = []
        annotation_paths: list[Path] = []
        missing: list[str] = []
        for record in self.records:
            path = self.annotation_path(record.sample_id)
            if not path.is_file():
                missing.append(record.sample_id)
                continue
            annotation = _read_json(path)
            if require_verified and not annotation.get("provenance", {}).get("verified"):
                missing.append(record.sample_id)
                continue
            annotations.append(annotation)
            annotation_paths.append(path)
        if missing:
            raise ValueError(
                "All imported frames must have verified annotations before export; missing: "
                + ", ".join(missing[:10])
            )
        if not annotations:
            raise ValueError("No annotations are available for export")
        split_ids = self._split_sessions(
            str(annotation["capture_session_id"]) for annotation in annotations
        )
        splits = {
            "group_key": "capture_session_id",
            "group_exclusive": True,
            **split_ids,
        }
        session_counts: dict[str, int] = {}
        for annotation in annotations:
            session = str(annotation["capture_session_id"])
            session_counts[session] = session_counts.get(session, 0) + 1
        statistics = {
            "captured_frames": len(self.records),
            "labeled_frames": len(annotations),
            "verified_frames": sum(bool(row["provenance"]["verified"]) for row in annotations),
            "double_audited_frames": sum(bool(row["provenance"].get("reviewer")) for row in annotations),
            "capture_sessions": len(session_counts),
            "layouts": len({str(row["environment"]["layout_id"]) for row in annotations}),
            "themes": len({str(row["environment"]["theme_id"]) for row in annotations}),
            "temporal_sequence_frames": sum(count for count in session_counts.values() if count > 1),
            "controlled_visible_frames": sum(
                bool(row["provenance"]["verified"])
                and row["environment"]["source_kind"] != "synthetic_render"
                for row in annotations
            ),
            "controlled_visible_sessions": len(
                {
                    str(row["capture_session_id"])
                    for row in annotations
                    if bool(row["provenance"]["verified"])
                    and row["environment"]["source_kind"] != "synthetic_render"
                }
            ),
            "synthetic_frames": sum(
                row["environment"]["source_kind"] == "synthetic_render"
                for row in annotations
            ),
        }
        ordered = sorted(annotations, key=lambda row: str(row["sample_id"]))
        duplicate_material = sorted(
            (
                str(row["image"]["sha256"]),
                str(row["image"].get("perceptual_hash", "")),
                str(row["capture_session_id"]),
            )
            for row in annotations
        )
        output_path = (
            Path(output).expanduser().resolve()
            if output is not None
            else self.root / "dataset_manifest.json"
        )
        relative_annotations = [
            path.relative_to(output_path.parent).as_posix()
            if path.is_relative_to(output_path.parent)
            else str(path)
            for path in annotation_paths
        ]
        manifest = {
            "schema_version": "1.0.0",
            "dataset_id": self.config["project_id"],
            "dataset_version": str(dataset_version),
            "created_at": utc_now(),
            "annotation_schema": "schemas/frame_annotation.schema.json",
            "source_policy": {
                "virtual_chips_only": True,
                "allowed_source_kinds": [self.config["environment"]["source_kind"]],
                "player_identity_policy": "local_only",
            },
            "annotation_files": relative_annotations,
            "splits": splits,
            "statistics": statistics,
            "fingerprints": {
                "annotations_sha256": canonical_sha256(ordered),
                "split_sha256": canonical_sha256(splits),
                "duplicate_audit_sha256": canonical_sha256(duplicate_material),
            },
        }
        _write_json(output_path, manifest)
        report = validate_manifest(output_path, require_images=True)
        if not report["valid"]:
            raise ValueError("Exported APC manifest failed validation: " + "; ".join(report["errors"]))
        return output_path, report

    def status(self) -> dict[str, object]:
        records = self.records
        annotated = sum(self.annotation_path(record.sample_id).is_file() for record in records)
        verified = 0
        suggestions = 0
        for record in records:
            annotation = self.load_annotation(record.sample_id)
            if annotation and annotation.get("provenance", {}).get("verified"):
                verified += 1
            if self.suggestion_path(record.sample_id).is_file():
                suggestions += 1
        return {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": self.config["project_id"],
            "root": str(self.root),
            "frames": len(records),
            "annotations": annotated,
            "verified_annotations": verified,
            "model_suggestions": suggestions,
            "capture_sessions": len({record.capture_session_id for record in records}),
            "pending_annotations": len(records) - annotated,
        }
