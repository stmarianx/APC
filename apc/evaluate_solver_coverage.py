from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Iterable


def _coach_types() -> dict[str, object]:
    try:
        from poker_coach import DecisionSolutionMatcher, PokerStarsParser, SolverExportRegistry, suit_isomorphic
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        coach_source = root / "coach" / "src"
        if str(coach_source) not in sys.path:
            sys.path.insert(0, str(coach_source))
        from poker_coach import DecisionSolutionMatcher, PokerStarsParser, SolverExportRegistry, suit_isomorphic
    return {
        "DecisionSolutionMatcher": DecisionSolutionMatcher,
        "PokerStarsParser": PokerStarsParser,
        "SolverExportRegistry": SolverExportRegistry,
        "suit_isomorphic": suit_isomorphic,
    }


def _ratio(numerator: int, denominator: int) -> str:
    return format(Decimal(numerator) / Decimal(denominator), "f") if denominator else "0"


def _source_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((row.resolve() for row in paths), key=lambda row: str(row).lower()):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _exclusion_reason(context: object, solutions: tuple[object, ...], suit_isomorphic: object) -> str:
    rows = [row for row in solutions if row.key.game == context.game]
    if not rows:
        return "no_game_nodes"
    rows = [row for row in rows if row.key.players == context.players]
    if not rows:
        return "no_player_count_nodes"
    rows = [row for row in rows if row.key.hero_position == context.hero_position]
    if not rows:
        return "no_position_nodes"
    rows = [
        row
        for row in rows
        if suit_isomorphic(row.key.board, row.key.hero_cards, context.board, context.hero_cards)
    ]
    if not rows:
        return "no_card_isomorphic_nodes"
    rows = [
        row
        for row in rows
        if abs(row.key.effective_stack_bb - context.effective_stack_bb) <= Decimal("2")
        and abs(row.key.pot_bb - context.pot_bb) <= Decimal("1")
    ]
    if not rows:
        return "stack_or_pot_out_of_tolerance"
    return "history_or_match_score_below_threshold"


def _slice(rows: list[dict[str, object]], field: str) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        key: {
            "decisions": len(group),
            "matched": sum(row["matched"] is True for row in group),
            "exact": sum(row["confidence"] == "exact" for row in group),
            "observed_action_covered": sum(row["observed_action_covered"] is True for row in group),
            "state_coverage": _ratio(sum(row["matched"] is True for row in group), len(group)),
            "exact_state_coverage": _ratio(sum(row["confidence"] == "exact" for row in group), len(group)),
        }
        for key, group in sorted(groups.items())
    }


def audit_solver_coverage(
    hands: Iterable[object],
    solutions: Iterable[object],
    *,
    minimum_decisions: int = 100,
    minimum_exact_coverage: Decimal = Decimal("0.80"),
) -> dict[str, object]:
    """Audit imported-solver coverage over provider-independent parsed hands."""
    if minimum_decisions <= 0:
        raise ValueError("minimum_decisions must be positive")
    if not Decimal("0") <= minimum_exact_coverage <= Decimal("1"):
        raise ValueError("minimum_exact_coverage must be between zero and one")
    types = _coach_types()
    matcher = types["DecisionSolutionMatcher"]()
    suit_isomorphic = types["suit_isomorphic"]
    hand_rows = tuple(hands)
    solution_rows = tuple(solutions)
    if not hand_rows:
        raise ValueError("coverage audit requires at least one parsed hand")
    if not solution_rows:
        raise ValueError("coverage audit requires at least one solver node")

    decisions: list[dict[str, object]] = []
    for hand in hand_rows:
        contexts = matcher.contexts(hand)
        matches = {row.context.action_index: row for row in matcher.match_hand(hand, solution_rows)}
        for context in contexts:
            match = matches.get(context.action_index)
            decisions.append(
                {
                    "hand_id": context.hand_id,
                    "action_index": context.action_index,
                    "street": context.street.name.lower(),
                    "players": context.players,
                    "hero_position": context.hero_position,
                    "pot_bb": format(context.pot_bb, "f"),
                    "effective_stack_bb": format(context.effective_stack_bb, "f"),
                    "matched": match is not None,
                    "confidence": None if match is None else match.confidence,
                    "card_match": None if match is None else match.card_match,
                    "solver_node_id": None if match is None else match.solution.node_id,
                    "observed_action": context.action.kind.value,
                    "observed_action_covered": match is not None and match.matched_action is not None,
                    "exclusion_reason": None
                    if match is not None
                    else _exclusion_reason(context, solution_rows, suit_isomorphic),
                }
            )

    total = len(decisions)
    matched = sum(row["matched"] is True for row in decisions)
    exact = sum(row["confidence"] == "exact" for row in decisions)
    action_covered = sum(row["observed_action_covered"] is True for row in decisions)
    confidence_counts = Counter(str(row["confidence"]) for row in decisions if row["confidence"])
    exclusion_counts = Counter(str(row["exclusion_reason"]) for row in decisions if row["exclusion_reason"])
    exact_coverage = Decimal(exact) / Decimal(total) if total else Decimal("0")
    gate_passed = total >= minimum_decisions and exact_coverage >= minimum_exact_coverage

    node_dimensions = {
        "games": sorted({row.key.game for row in solution_rows}),
        "player_counts": sorted({row.key.players for row in solution_rows}),
        "hero_positions": sorted({row.key.hero_position for row in solution_rows}),
        "streets": sorted({len(row.key.board) for row in solution_rows}),
        "rake_models": sorted({row.key.rake_model for row in solution_rows}),
        "utility_models": sorted({row.key.utility_model for row in solution_rows}),
    }
    return {
        "schema_version": "1.0.0",
        "evaluation_kind": "offline_decision_solver_coverage_audit",
        "audit_valid": total > 0,
        "promotion_eligible": False,
        "coverage_gate": {
            "passed": gate_passed,
            "minimum_decisions": minimum_decisions,
            "minimum_exact_coverage": format(minimum_exact_coverage, "f"),
            "observed_decisions": total,
            "observed_exact_coverage": format(exact_coverage, "f"),
        },
        "corpus": {"hands": len(hand_rows), "hero_decisions": total},
        "solver": {
            "nodes": len(solution_rows),
            "sources": sorted({f"{row.source}@{row.source_version}" for row in solution_rows}),
            "dimensions": node_dimensions,
        },
        "metrics": {
            "matched_decisions": matched,
            "unmatched_decisions": total - matched,
            "exact_matches": exact,
            "observed_actions_covered": action_covered,
            "state_coverage": _ratio(matched, total),
            "exact_state_coverage": _ratio(exact, total),
            "observed_action_coverage": _ratio(action_covered, total),
            "conditional_observed_action_coverage": _ratio(action_covered, matched),
            "confidence_counts": {key: confidence_counts.get(key, 0) for key in ("exact", "close", "approximate")},
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
        },
        "slices": {
            "street": _slice(decisions, "street"),
            "players": _slice(decisions, "players"),
            "hero_position": _slice(decisions, "hero_position"),
        },
        "decisions": decisions,
        "limitations": [
            "Coverage is measured only over the supplied completed-hand corpus and imported solver bundle.",
            "A valid audit is not a passed coverage gate; promotion additionally requires the declared sample-size and exact-coverage thresholds.",
            "This offline audit begins after hand parsing and does not measure visible-table perception coverage or latency.",
        ],
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


def evaluate_files(
    solver_export: str | Path,
    hand_history: str | Path,
    *,
    recursive: bool = False,
    minimum_decisions: int = 100,
    minimum_exact_coverage: Decimal = Decimal("0.80"),
) -> dict[str, object]:
    types = _coach_types()
    solver_path = Path(solver_export).resolve()
    hand_paths = _hand_files(Path(hand_history), recursive)
    bundle = types["SolverExportRegistry"]().parse_file(solver_path).bundle
    parser = types["PokerStarsParser"]()
    hands = tuple(hand for path in hand_paths for hand in parser.parse_file(path))
    report = audit_solver_coverage(
        hands,
        bundle.spots,
        minimum_decisions=minimum_decisions,
        minimum_exact_coverage=minimum_exact_coverage,
    )
    report["inputs"] = {
        "solver_export": str(solver_path),
        "solver_export_sha256": _source_fingerprint((solver_path,)),
        "hand_history_files": len(hand_paths),
        "hand_history_corpus_sha256": _source_fingerprint(hand_paths),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit APC solver coverage over completed-hand decisions.")
    parser.add_argument("solver_export", type=Path)
    parser.add_argument("hand_history", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--minimum-decisions", type=int, default=100)
    parser.add_argument("--minimum-exact-coverage", type=Decimal, default=Decimal("0.80"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_files(
            args.solver_export,
            args.hand_history,
            recursive=args.recursive,
            minimum_decisions=args.minimum_decisions,
            minimum_exact_coverage=args.minimum_exact_coverage,
        )
        if args.output:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["audit_valid"] else 3
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
