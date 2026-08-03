from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apc.curriculum.build_course_curriculum import (
    build_examples,
    build_manifest,
    build_parser,
    write_curriculum,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "analysis" / "transcript_evidence_index.json"
MATH = ROOT / "apc" / "curriculum" / "math_primitives.json"


class CourseCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        cls.math = json.loads(MATH.read_text(encoding="utf-8"))

    def test_all_verified_lessons_and_evidence_are_preserved(self) -> None:
        examples = build_examples(self.evidence, self.math)
        course = [row for row in examples if row["source"]["kind"] == "negreanu_course"]
        math = [row for row in examples if row["source"]["kind"] == "verified_math"]

        self.assertEqual(len(course), 214)
        self.assertEqual({row["source"]["lesson_ordinal"] for row in course}, set(range(1, 39)))
        self.assertEqual(len(math), len(self.math["primitives"]))
        self.assertTrue(all(row["target"]["must_cite_source"] for row in course))
        self.assertTrue(all(not row["labels"]["exact_strategy_target"] for row in examples))

    def test_course_lessons_are_group_exclusive_across_splits(self) -> None:
        examples = build_examples(self.evidence, self.math)
        manifest = build_manifest(examples, evidence_path=EVIDENCE, math_path=MATH)
        lesson_sets = [set(manifest["splits"][name]["course_lessons"]) for name in ("train", "validation", "test")]

        self.assertTrue(all(lesson_sets))
        self.assertFalse(lesson_sets[0] & lesson_sets[1])
        self.assertFalse(lesson_sets[0] & lesson_sets[2])
        self.assertFalse(lesson_sets[1] & lesson_sets[2])
        self.assertEqual(set().union(*lesson_sets), set(range(1, 39)))
        self.assertEqual(manifest["exact_strategy_targets"], 0)

    def test_generation_is_deterministic_and_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = write_curriculum(EVIDENCE, MATH, Path(directory) / "first")
            second = write_curriculum(EVIDENCE, MATH, Path(directory) / "second")

            self.assertEqual(first["curriculum_sha256"], second["curriculum_sha256"])
            lines = (Path(directory) / "first" / "course_curriculum.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), first["examples"])
            self.assertTrue(all(json.loads(line)["schema_version"] == "1.0.0" for line in lines))

    def test_default_cli_paths_resolve_inside_workspace(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.evidence.resolve(), EVIDENCE.resolve())
        self.assertEqual(args.math.resolve(), MATH.resolve())
        self.assertEqual(args.output.resolve(), (ROOT / "apc" / "curriculum" / "generated").resolve())


if __name__ == "__main__":
    unittest.main()
