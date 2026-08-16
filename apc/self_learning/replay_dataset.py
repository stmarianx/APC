from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable


def _coach_types() -> dict[str, object]:
    try:
        from poker_coach import DecisionSolutionMatcher, PokerStarsParser, SolverExportRegistry
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[2]
        coach_source = root / "coach" / "src"
        if str(coach_source) not in sys.path:
            sys.path.insert(0, str(coach_source))
        from poker_coach import DecisionSolutionMatcher, PokerStarsParser, SolverExportRegistry
    return {
        "DecisionSolutionMatcher": DecisionSolutionMatcher,
        "PokerStarsParser": PokerStarsParser,
        "SolverExportRegistry": SolverExportRegistry,
    }


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _bb(value: Decimal) -> str:
    return format(value, "f")


def _split(group_id: str, seed: int, train: Decimal, validation: Decimal) -> str:
    draw = Decimal(int(hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest(), 16)) / Decimal(2**256)
    if draw < train:
        return "train"
    if draw < train + validation:
        return "validation"
    return "test"


def _ratios(split_ratios: tuple[Decimal, Decimal, Decimal]) -> tuple[Decimal, Decimal, Decimal]:
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must contain train, validation and test")
    if any(not row.is_finite() or row < 0 for row in split_ratios):
        raise ValueError("split ratios must be finite and non-negative")
    if sum(split_ratios, Decimal("0")) != Decimal("1"):
        raise ValueError("split ratios must sum exactly to one")
    return split_ratios


def replay_examples(
    hands: Iterable[object],
    solutions: Iterable[object],
    *,
    seed: int = 20260816,
    split_ratios: tuple[Decimal, Decimal, Decimal] = (
        Decimal("0.80"),
        Decimal("0.10"),
        Decimal("0.10"),
    ),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Create exact-match policy targets and completed-hand feedback labels."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_ratio, validation_ratio, test_ratio = _ratios(split_ratios)
    types = _coach_types()
    matcher = types["DecisionSolutionMatcher"]()
    hand_rows = tuple(hands)
    solution_rows = tuple(solutions)
    if not hand_rows:
        raise ValueError("replay dataset requires at least one completed hand")
    if not solution_rows:
        raise ValueError("replay dataset requires at least one imported solver node")

    examples: list[dict[str, object]] = []
    input_decisions = 0
    exclusions: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for hand in hand_rows:
        contexts = matcher.contexts(hand)
        input_decisions += len(contexts)
        matches = {row.context.action_index: row for row in matcher.match_hand(hand, solution_rows)}
        for context in contexts:
            match = matches.get(context.action_index)
            if match is None:
                exclusions["unmatched_solver_state"] += 1
                continue
            if match.confidence != "exact":
                exclusions[f"match_confidence_{match.confidence}"] += 1
                continue
            group_id = hashlib.sha256(f"hand:{context.hand_id}".encode("utf-8")).hexdigest()[:24]
            identity = f"{context.hand_id}:{context.action_index}:{match.solution.key.fingerprint}"
            example_id = "replay-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            if example_id in seen_ids:
                raise ValueError(f"duplicate replay example identity: {example_id}")
            seen_ids.add(example_id)
            observed_ev_loss = match.ev_loss_bb
            example = {
                "schema_version": "1.0.0",
                "example_id": example_id,
                "group_id": group_id,
                "split": _split(group_id, seed, train_ratio, validation_ratio),
                "units": "BB",
                "state": {
                    "game": context.game,
                    "players": context.players,
                    "hero_position": context.hero_position,
                    "effective_stack_bb": _bb(context.effective_stack_bb),
                    "pot_bb": _bb(context.pot_bb),
                    "board": [str(card) for card in context.board],
                    "hero_cards": [str(card) for card in context.hero_cards],
                    "action_history": list(context.action_history),
                },
                "target": {
                    "kind": "imported_solver_mixed_strategy",
                    "actions": [
                        {
                            "action_id": action.action,
                            "frequency": _bb(action.frequency),
                            "ev_bb": _bb(action.ev),
                        }
                        for action in match.solution.actions
                    ],
                    "best_ev_bb": _bb(match.solution.best_ev),
                    "gto_verified": False,
                },
                "completed_hand_feedback": {
                    "observed_action_id": match.matched_action,
                    "observed_action_covered": match.matched_action is not None,
                    "ev_loss_bb": None if observed_ev_loss is None else _bb(observed_ev_loss),
                },
                "provenance": {
                    "hand_id_sha256": hashlib.sha256(context.hand_id.encode("utf-8")).hexdigest(),
                    "action_index": context.action_index,
                    "solver_node_id": match.solution.node_id,
                    "solver_fingerprint": match.solution.key.fingerprint,
                    "solver_source": match.solution.source,
                    "solver_source_version": match.solution.source_version,
                    "match_confidence": match.confidence,
                    "card_match": match.card_match,
                },
            }
            example["example_sha256"] = _sha256_bytes(_canonical(example))
            examples.append(example)
    examples.sort(key=lambda row: str(row["example_id"]))
    diagnostics = {
        "completed_hands": len(hand_rows),
        "input_hero_decisions": input_decisions,
        "eligible_exact_examples": len(examples),
        "feedback_covered_examples": sum(
            row["completed_hand_feedback"]["observed_action_covered"] is True for row in examples
        ),
        "exclusions": dict(sorted(exclusions.items())),
        "seed": seed,
        "split_ratios": {
            "train": _bb(train_ratio),
            "validation": _bb(validation_ratio),
            "test": _bb(test_ratio),
        },
    }
    return examples, diagnostics


def build_replay_dataset(
    output: str | Path,
    hands: Iterable[object],
    solutions: Iterable[object],
    *,
    dataset_id: str,
    source_fingerprints: dict[str, str],
    seed: int = 20260816,
    split_ratios: tuple[Decimal, Decimal, Decimal] = (
        Decimal("0.80"),
        Decimal("0.10"),
        Decimal("0.10"),
    ),
) -> dict[str, object]:
    if not dataset_id or any(character in dataset_id for character in "\\/:*?\"<>|"):
        raise ValueError("dataset_id must be a non-empty portable name")
    if not source_fingerprints or any(
        not isinstance(value, str) or len(value) != 64 for value in source_fingerprints.values()
    ):
        raise ValueError("source_fingerprints must contain SHA-256 strings")
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError(f"replay dataset destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    examples, diagnostics = replay_examples(
        hands,
        solutions,
        seed=seed,
        split_ratios=split_ratios,
    )
    if not examples:
        raise ValueError("no exact solver-matched decisions are eligible for replay training")

    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        temporary_path = Path(temporary)
        examples_path = temporary_path / "examples.jsonl"
        examples_bytes = b"".join(_canonical(row) + b"\n" for row in examples)
        examples_path.write_bytes(examples_bytes)
        split_counts = Counter(str(row["split"]) for row in examples)
        group_splits: dict[str, str] = {}
        for row in examples:
            group = str(row["group_id"])
            split = str(row["split"])
            previous = group_splits.setdefault(group, split)
            if previous != split:
                raise ValueError(f"group leakage detected for {group}")
        manifest = {
            "schema_version": "1.0.0",
            "dataset_id": dataset_id,
            "dataset_kind": "solver_labeled_completed_hand_replay",
            "immutable": True,
            "units": "BB",
            "training_eligible": False,
            "training_eligibility_reason": "candidate pipeline and paired promotion evaluation are not yet implemented",
            "examples_file": "examples.jsonl",
            "examples_sha256": _sha256_bytes(examples_bytes),
            "example_count": len(examples),
            "group_count": len(group_splits),
            "split_counts": {key: split_counts.get(key, 0) for key in ("train", "validation", "test")},
            "group_exclusive": True,
            "source_fingerprints": dict(sorted(source_fingerprints.items())),
            "build": diagnostics,
            "limitations": [
                "Only exact matcher states become policy targets; unmatched and approximate decisions are excluded and counted.",
                "Imported solver targets are not labelled GTO-verified without separate verification evidence.",
                "This dataset contains structured completed-hand states, not visible-table pixels.",
            ],
        }
        fingerprint_material = dict(manifest)
        manifest["dataset_fingerprint"] = _sha256_bytes(_canonical(fingerprint_material))
        (temporary_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        validation = validate_replay_dataset(temporary_path)
        if not validation["valid"]:
            raise ValueError("generated replay dataset failed validation: " + "; ".join(validation["issues"]))
        temporary_path.replace(destination)
    return manifest


def validate_replay_dataset(root: str | Path) -> dict[str, object]:
    dataset = Path(root).resolve()
    issues: list[str] = []
    manifest_path = dataset / "manifest.json"
    examples_path = dataset / "examples.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"valid": False, "issues": [f"manifest unreadable: {error}"]}
    try:
        lines = [line for line in examples_path.read_text(encoding="utf-8").splitlines() if line]
        examples = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        return {"valid": False, "issues": [f"examples unreadable: {error}"]}
    if manifest.get("schema_version") != "1.0.0" or manifest.get("immutable") is not True:
        issues.append("manifest schema/immutability contract is invalid")
    if manifest.get("units") != "BB" or manifest.get("group_exclusive") is not True:
        issues.append("manifest must use BB and declare group-exclusive splits")
    if manifest.get("example_count") != len(examples):
        issues.append("manifest example_count does not match examples")
    if manifest.get("examples_sha256") != _sha256_file(examples_path):
        issues.append("examples file fingerprint mismatch")
    material = dict(manifest)
    observed_dataset_fingerprint = material.pop("dataset_fingerprint", None)
    if observed_dataset_fingerprint != _sha256_bytes(_canonical(material)):
        issues.append("dataset fingerprint mismatch")
    groups: dict[str, str] = {}
    ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    for index, example in enumerate(examples):
        label = f"example[{index}]"
        if example.get("schema_version") != "1.0.0" or example.get("units") != "BB":
            issues.append(f"{label} schema/BB contract is invalid")
        example_id = str(example.get("example_id", ""))
        if not example_id or example_id in ids:
            issues.append(f"{label} example_id is missing or duplicated")
        ids.add(example_id)
        split = str(example.get("split", ""))
        if split not in {"train", "validation", "test"}:
            issues.append(f"{label} split is invalid")
        split_counts[split] += 1
        group = str(example.get("group_id", ""))
        previous = groups.setdefault(group, split)
        if not group or previous != split:
            issues.append(f"{label} leaks a group across splits")
        expected_hash = example.get("example_sha256")
        material = dict(example)
        material.pop("example_sha256", None)
        if expected_hash != _sha256_bytes(_canonical(material)):
            issues.append(f"{label} fingerprint mismatch")
        target = example.get("target")
        if not isinstance(target, dict) or target.get("gto_verified") is not False:
            issues.append(f"{label} must not claim unverified GTO provenance")
            continue
        actions = target.get("actions")
        if not isinstance(actions, list) or not actions:
            issues.append(f"{label} target actions are missing")
            continue
        try:
            frequency = sum((Decimal(str(row["frequency"])) for row in actions), Decimal("0"))
        except (InvalidOperation, KeyError, TypeError):
            issues.append(f"{label} action frequency is invalid")
        else:
            if frequency != Decimal("1"):
                issues.append(f"{label} action frequencies do not sum to one")
    expected_counts = {key: split_counts.get(key, 0) for key in ("train", "validation", "test")}
    if manifest.get("split_counts") != expected_counts:
        issues.append("manifest split_counts do not match examples")
    if manifest.get("group_count") != len(groups):
        issues.append("manifest group_count does not match examples")
    return {
        "schema_version": "1.0.0",
        "valid": not issues,
        "issues": issues,
        "example_count": len(examples),
        "group_count": len(groups),
        "split_counts": expected_counts,
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
    }


def _hand_files(path: Path, recursive: bool) -> tuple[Path, ...]:
    resolved = path.resolve()
    if resolved.is_file():
        return (resolved,)
    if not resolved.is_dir():
        raise ValueError(f"hand-history path does not exist: {resolved}")
    iterator = resolved.rglob("*.txt") if recursive else resolved.glob("*.txt")
    rows = tuple(sorted((row for row in iterator if row.is_file()), key=lambda row: str(row).lower()))
    if not rows:
        raise ValueError(f"hand-history folder contains no .txt files: {resolved}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an immutable APC completed-hand replay dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("solver_export", type=Path)
    build.add_argument("hand_history", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--seed", type=int, default=20260816)
    build.add_argument("--recursive", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_replay_dataset(args.dataset)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 3
        types = _coach_types()
        solver_path = args.solver_export.resolve()
        hand_paths = _hand_files(args.hand_history, args.recursive)
        bundle = types["SolverExportRegistry"]().parse_file(solver_path).bundle
        hand_parser = types["PokerStarsParser"]()
        hands = tuple(hand for path in hand_paths for hand in hand_parser.parse_file(path))
        hand_digest = hashlib.sha256()
        for path in hand_paths:
            hand_digest.update(path.name.encode("utf-8"))
            hand_digest.update(b"\0")
            hand_digest.update(path.read_bytes())
            hand_digest.update(b"\0")
        manifest = build_replay_dataset(
            args.output,
            hands,
            bundle.spots,
            dataset_id=args.dataset_id,
            source_fingerprints={
                "solver_export_sha256": _sha256_file(solver_path),
                "hand_history_corpus_sha256": hand_digest.hexdigest(),
            },
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
