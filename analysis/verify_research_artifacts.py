from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = Path(__file__).resolve().parent
VISUAL = Path(
    r"C:\Users\st_ma\.codex\visualizations\2026\07\31\019fba2e-2fcb-7d23-8443-06624692557b\poker-gto-math-graph.html"
)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []
    manifest = json.loads((ANALYSIS / "media_manifest.json").read_text(encoding="utf-8"))
    videos = manifest["videos"]
    expected_ordinals = list(range(1, 39))
    expected_files = {int(item["ordinal"]): str(item["file"]) for item in videos}

    if manifest.get("video_count") != 38 or sorted(expected_files) != expected_ordinals:
        fail(errors, "Manifest does not contain exactly lesson ordinals 1-38")
    source_files = sorted(path.name for path in ROOT.glob("*.mp4"))
    if sorted(expected_files.values()) != source_files:
        fail(errors, "Manifest filenames do not match the 38 source MP4 files")

    transcript_paths = sorted((ANALYSIS / "transcripts").glob("*.json"))
    transcript_by_ordinal: dict[int, dict[str, object]] = {}
    total_words = 0
    total_segments = 0
    for path in transcript_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        match = re.match(r"^(\d+)", str(payload.get("file", "")))
        if not match:
            fail(errors, f"No ordinal in transcript source: {path.name}")
            continue
        ordinal = int(match.group(1))
        if ordinal in transcript_by_ordinal:
            fail(errors, f"Duplicate transcript ordinal {ordinal}")
        transcript_by_ordinal[ordinal] = payload
        if payload.get("file") != expected_files.get(ordinal):
            fail(errors, f"Transcript {ordinal} source filename differs from manifest")
        decoding = payload.get("decoding", {})
        if decoding.get("condition_on_previous_text") is not False or decoding.get("beam_size") != 1:
            fail(errors, f"Transcript {ordinal} does not use the corrected decode policy")
        segments = payload.get("segments", [])
        if not segments:
            fail(errors, f"Transcript {ordinal} has no segments")
            continue
        prior_start = -1.0
        for segment in segments:
            start = float(segment["start"])
            end = float(segment["end"])
            if start < prior_start or end < start:
                fail(errors, f"Transcript {ordinal} contains non-monotonic timestamps")
                break
            prior_start = start
        total_segments += len(segments)
        total_words += len(re.findall(r"\b\w+\b", " ".join(str(s["text"]) for s in segments)))

    if sorted(transcript_by_ordinal) != expected_ordinals or len(transcript_paths) != 38:
        fail(errors, "Canonical transcript directory is not exactly ordinals 1-38")

    index = json.loads((ANALYSIS / "transcript_evidence_index.json").read_text(encoding="utf-8"))
    if [int(record["ordinal"]) for record in index] != expected_ordinals:
        fail(errors, "Evidence index is not ordered exactly 1-38")
    if any(record.get("quality_flags") for record in index):
        fail(errors, "Evidence index still contains transcript quality flags")
    indexed_words = sum(int(record["word_count"]) for record in index)
    indexed_segments = sum(int(record["segment_count"]) for record in index)
    if indexed_words != total_words or indexed_segments != total_segments:
        fail(errors, "Evidence index totals differ from canonical transcripts")

    contact_sheets = list((ANALYSIS / "contact_sheets").glob("*.jpg"))
    galleries = list((ANALYSIS / "contact_sheet_gallery").glob("*.jpg"))
    if len(contact_sheets) != 38:
        fail(errors, f"Expected 38 contact sheets, found {len(contact_sheets)}")
    if len(galleries) != 19:
        fail(errors, f"Expected 19 gallery pages, found {len(galleries)}")

    required_reports = [
        ANALYSIS / "README.md",
        ANALYSIS / "course_video_analysis.md",
        ANALYSIS / "gto_math_foundation.md",
        ANALYSIS / "transcript_evidence_index.md",
    ]
    for path in required_reports:
        if not path.exists() or path.stat().st_size == 0:
            fail(errors, f"Missing or empty report: {path.name}")

    if not VISUAL.exists():
        fail(errors, "Interactive GTO graph is missing")
        node_count = edge_count = unique_node_count = 0
    else:
        visual_text = VISUAL.read_text(encoding="utf-8")
        node_ids = re.findall(r'\{id:"([^"]+)",label:', visual_text)
        edge_count = len(re.findall(r'\["[^"]+","[^"]+","[^"]+"\]', visual_text))
        node_count = len(node_ids)
        unique_node_count = len(set(node_ids))
        if node_count != 63 or unique_node_count != 63 or edge_count != 79:
            fail(errors, "Interactive graph does not match the verified 63-node/79-edge structure")

    result = {
        "success": not errors,
        "errors": errors,
        "source_videos": len(source_files),
        "manifest_videos": manifest.get("video_count"),
        "duration_hours": manifest.get("total_duration_hours"),
        "canonical_transcripts": len(transcript_paths),
        "transcript_words": total_words,
        "transcript_segments": total_segments,
        "evidence_records": len(index),
        "evidence_excerpts": sum(len(record.get("evidence", [])) for record in index),
        "contact_sheets": len(contact_sheets),
        "gallery_pages": len(galleries),
        "graph_nodes": node_count,
        "graph_unique_nodes": unique_node_count,
        "graph_edges": edge_count,
    }
    (ANALYSIS / "research_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
