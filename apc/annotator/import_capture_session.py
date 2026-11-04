"""Import a fingerprinted capture session without inventing frame timestamps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from apc.annotator.project import AnnotationProject, png_size
from apc.tools.validate_dataset import canonical_sha256


def import_capture_session(project: AnnotationProject, folder: str | Path, *, allow_partial: bool = False) -> dict[str, object]:
    root = Path(folder).resolve()
    raw = (root / "capture_report.json").read_bytes()
    report = json.loads(raw)
    plan = json.loads((root / "capture_plan.json").read_text(encoding="utf-8"))
    if report.get("schema_version") != "1.1.0" or report.get("timing_basis") != "monotonic_capture_start_relative_to_session":
        raise ValueError("Measured capture report v1.1.0 required; legacy folder timing is not measured")
    status = report.get("status")
    if status != "completed" and not (allow_partial and status in {"failed", "interrupted"}):
        raise ValueError("Capture must be completed, or explicitly import a failed/interrupted partial session")
    plan_body = {key: value for key, value in plan.items() if key not in {"created_at", "plan_sha256"}}
    if canonical_sha256(plan_body) != plan.get("plan_sha256") or report.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("Capture plan fingerprint mismatch")
    # Reuse the same session/region/bounds validation as the capture producer.
    from apc.capture.screen_capture import CapturePlan, CaptureRegion
    validated = CapturePlan(plan["session_id"], CaptureRegion(**plan["region"]), plan["frames"], plan["interval_ms"])
    if report.get("session_id") != validated.session_id:
        raise ValueError("Capture session mismatch")
    rows = report["frames"]
    if not rows or len(rows) != report.get("captured_frames") or canonical_sha256(rows) != report.get("frames_sha256"):
        raise ValueError("Capture frame manifest fingerprint/count mismatch or empty session")
    if len(rows) > validated.frames or (status == "completed" and len(rows) != validated.frames):
        raise ValueError("Capture frame count disagrees with plan")

    checked = []
    previous_ms, previous_end = -1, -1
    # Preflight every source before changing the annotation project.
    for index, row in enumerate(rows):
        timestamp = row["timestamp_ms"]
        start, end = row["capture_start_ns"], row["capture_end_ns"]
        if any(type(v) is not int for v in (timestamp, start, end)) or start < 0 or start < previous_end or end < start:
            raise ValueError("Invalid measured capture timing")
        expected_ms = max(start // 1_000_000, previous_ms + 1)
        if row["index"] != index or timestamp != expected_ms or row["scheduled_timestamp_ms"] != index * validated.interval_ms:
            raise ValueError("Invalid capture ordering or timestamp")
        if row.get("timestamp_resolution_adjusted") is not (timestamp != start // 1_000_000):
            raise ValueError("Invalid timestamp adjustment flag")
        name = row["filename"]
        if not isinstance(name, str) or Path(name).name != name or Path(name).suffix.lower() != ".png":
            raise ValueError("Capture filename must be a local PNG filename")
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError("Capture frame escapes its source folder")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Capture image fingerprint mismatch")
        size = png_size(path)
        if size != (validated.region.width, validated.region.height) or size != (row["width"], row["height"]):
            raise ValueError("Capture image dimensions mismatch")
        checked.append((path, row))
        previous_ms, previous_end = timestamp, end

    known = {record.sha256 for record in project.records}
    latest = max((record.timestamp_ms for record in project.records if record.capture_session_id == validated.session_id), default=-1)
    for _, row in checked:
        if row["sha256"] not in known:
            if row["timestamp_ms"] <= latest:
                raise ValueError("Capture overlaps existing session timestamps")
            latest = row["timestamp_ms"]
            known.add(row["sha256"])
    imported = []
    for path, row in checked:
        record, inserted = project.import_frame(path, capture_session_id=validated.session_id, timestamp_ms=row["timestamp_ms"])
        imported.append({"sample_id": record.sample_id, "source_timestamp_ms": row["timestamp_ms"], "record_timestamp_ms": record.timestamp_ms, "inserted": inserted})
    return {
        "capture_report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_status": status, "capture_session_id": validated.session_id,
        "inserted_frames": sum(row["inserted"] for row in imported),
        "duplicate_frames": sum(not row["inserted"] for row in imported),
        "frames": imported, "labels_created": False, "readiness_counts_added": False,
    }
