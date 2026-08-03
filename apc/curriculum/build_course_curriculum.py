from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"
SPLIT_SALT = "apc-course-split-v1"
EMPIRICAL_TOPICS = {
    "bankroll",
    "game selection",
    "metagame",
    "player modeling",
    "table image",
    "tells and live reads",
    "tilt",
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lesson_split(video_file: str) -> str:
    bucket = int(
        hashlib.sha256(f"{SPLIT_SALT}:{video_file}".encode("utf-8")).hexdigest()[:8],
        16,
    ) % 10
    if bucket == 0:
        return "validation"
    if bucket == 1:
        return "test"
    return "train"


def _example_id(kind: str, material: object) -> str:
    return f"{kind}-{canonical_sha256(material)[:20]}"


def build_examples(
    evidence_index: Iterable[dict[str, Any]],
    math_curriculum: dict[str, Any],
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for lesson in evidence_index:
        video_file = str(lesson["file"])
        split = lesson_split(video_file)
        for evidence in lesson.get("evidence", []):
            topic = str(evidence["topic"])
            source = {
                "kind": "negreanu_course",
                "lesson_ordinal": int(lesson["ordinal"]),
                "video_file": video_file,
                "transcript_file": str(lesson["transcript_file"]),
                "start_seconds": evidence["start"],
                "end_seconds": evidence["end"],
            }
            material = {"source": source, "topic": topic, "text": evidence["text"]}
            examples.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": _example_id("course", material),
                    "split": split,
                    "source": source,
                    "task": {
                        "type": "concept_evidence_retrieval",
                        "instruction": (
                            f"Identify the timestamped course evidence for the concept "
                            f"'{topic}' in lesson {int(lesson['ordinal']):02d}."
                        ),
                    },
                    "target": {
                        "evidence_text": str(evidence["text"]),
                        "must_cite_source": True,
                    },
                    "labels": {
                        "concepts": [topic],
                        "knowledge_tier": "course_evidence",
                        "exact_strategy_target": False,
                        "verification_requirement": (
                            "requires_empirical_validation"
                            if topic in EMPIRICAL_TOPICS
                            else "requires_solver_or_math_crosscheck"
                        ),
                    },
                    "quality": {
                        "lesson_quality_flags": list(lesson.get("quality_flags", [])),
                        "mean_transcript_logprob": lesson.get("mean_avg_logprob"),
                    },
                }
            )

    for primitive in math_curriculum.get("primitives", []):
        material = {"id": primitive["id"], "formula": primitive["formula"]}
        examples.append(
            {
                "schema_version": SCHEMA_VERSION,
                "example_id": _example_id("math", material),
                "split": "train",
                "source": {
                    "kind": "verified_math",
                    "document": str(math_curriculum["source"]),
                    "primitive_id": str(primitive["id"]),
                    "verification": str(primitive["verification"]),
                },
                "task": {
                    "type": "math_primitive",
                    "instruction": str(primitive["question"]),
                },
                "target": {
                    "formula": str(primitive["formula"]),
                    "assumptions": list(primitive["assumptions"]),
                    "units": str(math_curriculum["units"]),
                },
                "labels": {
                    "concepts": list(primitive["concepts"]),
                    "knowledge_tier": "verified_math",
                    "exact_strategy_target": False,
                    "verification_requirement": "machine_or_identity_verified",
                },
                "quality": {"lesson_quality_flags": [], "mean_transcript_logprob": None},
            }
        )
    if len({example["example_id"] for example in examples}) != len(examples):
        raise ValueError("Curriculum example ids are not unique")
    return examples


def build_manifest(
    examples: list[dict[str, object]],
    *,
    evidence_path: Path,
    math_path: Path,
) -> dict[str, object]:
    split_examples = Counter(str(example["split"]) for example in examples)
    split_lessons: dict[str, set[int]] = defaultdict(set)
    concepts: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    for example in examples:
        source = example["source"]
        if source["kind"] == "negreanu_course":
            split_lessons[str(example["split"])].add(int(source["lesson_ordinal"]))
        labels = example["labels"]
        concepts.update(str(concept) for concept in labels["concepts"])
        tiers[str(labels["knowledge_tier"])] += 1
    lesson_sets = list(split_lessons.values())
    if any(lesson_sets[left] & lesson_sets[right] for left in range(len(lesson_sets)) for right in range(left + 1, len(lesson_sets))):
        raise ValueError("Course lessons cross curriculum split boundaries")
    return {
        "schema_version": SCHEMA_VERSION,
        "curriculum_id": "apc-negreanu-and-math-v1",
        "units": "BB",
        "sources": {
            "course_evidence_index": {
                "path": str(evidence_path),
                "sha256": file_sha256(evidence_path),
            },
            "verified_math": {
                "path": str(math_path),
                "sha256": file_sha256(math_path),
            },
        },
        "examples": len(examples),
        "knowledge_tiers": dict(sorted(tiers.items())),
        "concept_counts": dict(sorted(concepts.items())),
        "splits": {
            name: {
                "examples": split_examples[name],
                "course_lessons": sorted(split_lessons.get(name, set())),
            }
            for name in ("train", "validation", "test")
        },
        "exact_strategy_targets": sum(
            bool(example["labels"]["exact_strategy_target"]) for example in examples
        ),
        "curriculum_sha256": canonical_sha256(examples),
        "split_policy": {
            "group_key": "lesson_ordinal",
            "group_exclusive": True,
            "algorithm": "sha256_mod10",
            "salt": SPLIT_SALT,
            "math_primitives": "train_only",
        },
        "limitations": [
            "Course evidence is suitable for grounded explanation and concept retrieval, not exact action-frequency labels.",
            "Exact GTO action targets require a fully specified game and solver provenance.",
        ],
    }


def write_curriculum(
    evidence_path: Path,
    math_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    math = json.loads(math_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, list) or len(evidence) != 38:
        raise ValueError("Expected the verified 38-lesson evidence index")
    if math.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported math curriculum schema")
    examples = build_examples(evidence, math)
    manifest = build_manifest(examples, evidence_path=evidence_path, math_path=math_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "course_curriculum.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n" for example in examples),
        encoding="utf-8",
    )
    manifest_path = output_dir / "curriculum_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build APC's provenance-aware course and math curriculum.")
    root = Path(__file__).resolve().parents[1]
    workspace = root.parent
    parser.add_argument("--evidence", type=Path, default=workspace / "analysis" / "transcript_evidence_index.json")
    parser.add_argument("--math", type=Path, default=root / "curriculum" / "math_primitives.json")
    parser.add_argument("--output", type=Path, default=root / "curriculum" / "generated")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = write_curriculum(args.evidence.resolve(), args.math.resolve(), args.output.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
