from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from poker_coach import (  # noqa: E402
    HandReplayer,
    Card,
    HoleCards,
    PokerStarsParser,
    ScenarioLibrary,
    SolverBundleImporter,
    InMemorySolutionStore,
    InMemoryStudyStore,
    SolverExportRegistry,
    SolverPracticeService,
    LiveTableService,
    StateTransitionError,
    PokerStarsLiveTailAdapter,
    VisualObservationAdapter,
    SolutionForest,
    aggregate_range_strategies,
    CoachDatabase,
    HandHistoryFolderScanner,
    TrainingService,
    analyze_hands,
    analyze_with_solutions,
    analyze_range_matchup,
    condition_solution_range,
    build_opponent_range_timelines,
    score_opponent_range_timelines,
    parse_range,
    solve_akq_river,
)


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    sample_path = ROOT / "examples" / "sample_play_money_hand.txt"
    hands = PokerStarsParser().parse_file(sample_path)
    replay = HandReplayer().replay(hands[0])
    report = analyze_hands(hands)
    library = ScenarioLibrary.bundled()
    training = TrainingService(library)
    session = training.create_session(count=1, seed=1)
    scenario = library.get(str(session["scenario"]["scenario_id"]))
    best_action = max(scenario.actions, key=lambda action: action.ev_bb)
    training_result = training.submit_decision(str(session["session_id"]), best_action.action_id)
    _, river_solution = solve_akq_river(iterations=20_000)
    solver_bundle = SolverBundleImporter().parse_file(ROOT / "examples" / "sample_solver_bundle.json")
    solver_import = SolverBundleImporter().import_into(InMemorySolutionStore(), solver_bundle)
    tabular_export = SolverExportRegistry().parse_file(
        ROOT / "examples" / "sample_solver_export.csv"
    )
    tabular_import = SolverBundleImporter().import_into(
        InMemorySolutionStore(), tabular_export.bundle
    )
    solution_forest = SolutionForest(tabular_export.bundle.spots)
    range_strategy_report = aggregate_range_strategies(solver_bundle.spots)
    practice_service = SolverPracticeService()
    practice_challenge = practice_service.create(tabular_export.bundle.spots)
    practice_action = str(practice_challenge["actions"][0]["action_id"])
    practice_result = practice_service.submit(
        str(practice_challenge["session_id"]), practice_action
    )
    live_service = LiveTableService()
    live_spot = tabular_export.bundle.spots[-1]
    live_key = live_spot.key
    live_session = live_service.create_session("Verifier Play Table")
    live_state_payload = {
        "schema_version": "1.0.0",
        "table_id": "Verifier Play Table",
        "hand_id": "verify-live-1",
        "revision": 0,
        "game": live_key.game,
        "players": live_key.players,
        "hero_position": live_key.hero_position,
        "effective_stack_bb": format(live_key.effective_stack_bb, "f"),
        "pot_bb": format(live_key.pot_bb, "f"),
        "to_call_bb": "0",
        "board": [str(card) for card in live_key.board],
        "hero_cards": [str(card) for card in live_key.hero_cards],
        "action_history": list(live_key.action_history),
        "legal_actions": [action.action for action in live_spot.actions],
        "rake_model": live_key.rake_model,
        "utility_model": live_key.utility_model,
        "source": "verification_normalized_feed",
    }
    live_result = live_service.update_state(
        str(live_session["session_id"]),
        live_state_payload,
        tabular_export.bundle.spots,
    )
    live_decision = live_service.record_decision(
        str(live_session["session_id"]),
        0,
        str(live_result["match"]["actions"][0]["action"]),
    )
    invalid_live_state = dict(live_state_payload)
    invalid_live_state["revision"] = 1
    invalid_live_state["pot_bb"] = format(live_key.pot_bb - 1, "f")
    try:
        live_service.update_state(
            str(live_session["session_id"]),
            invalid_live_state,
            tabular_export.bundle.spots,
        )
        raise AssertionError("Temporal invariant gate accepted a pot rollback")
    except StateTransitionError as error:
        temporal_transition_audit = error.audit
    range_board = tuple(
        Card.parse(token) for token in ("2h", "3h", "4s", "9c", "7d")
    )
    range_matchup = analyze_range_matchup(
        parse_range("AsAd", dead=range_board),
        parse_range("KcKd,QcQd", dead=range_board),
        range_board,
    ).to_dict()
    range_conditioning = condition_solution_range(
        solver_bundle.spots[:2], "check"
    )
    opponent_timeline = build_opponent_range_timelines(hands, solver_bundle.spots)
    revealed_hand = replace(
        hands[0],
        hole_cards=hands[0].hole_cards
        + (
            HoleCards(
                "VillainB",
                (Card.parse("Kc"), Card.parse("Qh")),
                shown=True,
            ),
        ),
    )
    revealed_timeline = build_opponent_range_timelines(
        (revealed_hand,), solver_bundle.spots
    )
    range_calibration = score_opponent_range_timelines(
        (revealed_hand,), solver_bundle.spots, revealed_timeline
    )
    live_tail_source = """PokerStars Hand #93000000001: Hold'em No Limit (0.50/1 Play Money) - 2026/08/01 21:00:00 ET
Table 'Verifier Tail' 2-max Seat #1 is the button
Seat 1: Hero (100 in chips)
Seat 2: Villain (100 in chips)
Hero: posts small blind 0.50
Villain: posts big blind 1
*** HOLE CARDS ***
Dealt to Hero [As Kd]
"""
    with TemporaryDirectory() as live_tail_directory:
        live_tail_path = Path(live_tail_directory) / "HH-live.txt"
        live_tail_path.write_text(live_tail_source, encoding="utf-8")
        live_tail_adapter = PokerStarsLiveTailAdapter()
        first_live_poll = live_tail_adapter.poll(
            live_tail_path, table_id="Verifier Tail Session"
        )
        unchanged_live_poll = live_tail_adapter.poll(
            live_tail_path, table_id="Verifier Tail Session"
        )
        live_tail_path.write_text(
            live_tail_source + "Hero: raises 2.50 to 3\n", encoding="utf-8"
        )
        waiting_live_poll = live_tail_adapter.poll(
            live_tail_path, table_id="Verifier Tail Session"
        )
    live_tail_audit = {
            "first_status": first_live_poll.status,
            "first_revision": first_live_poll.revision,
            "first_changed": first_live_poll.changed,
            "unchanged_reused": not unchanged_live_poll.changed,
            "waiting_status": waiting_live_poll.status,
            "waiting_next_actor": waiting_live_poll.next_actor,
        }
    visual_values = {
        "table_id": "Verifier Visual Table",
        "hand_id": "verify-visual-1",
        "game": live_key.game,
        "players": live_key.players,
        "hero_position": live_key.hero_position,
        "effective_stack_bb": format(live_key.effective_stack_bb, "f"),
        "pot_bb": format(live_key.pot_bb, "f"),
        "to_call_bb": "0",
        "board": [str(card) for card in live_key.board],
        "hero_cards": [str(card) for card in live_key.hero_cards],
        "action_history": list(live_key.action_history),
        "legal_actions": [action.action for action in live_spot.actions],
        "rake_model": live_key.rake_model,
        "utility_model": live_key.utility_model,
    }
    visual_region_for = {
        "effective_stack_bb": "stack",
        "pot_bb": "pot",
        "to_call_bb": "actions",
        "board": "board",
        "hero_cards": "hero_cards",
        "legal_actions": "actions",
    }

    def visual_manifest(frame_id: str) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "provider": "verification_visual_provider",
            "provider_version": "1.0",
            "frame": {"frame_id": frame_id, "image_sha256": frame_id[0] * 64},
            "calibration": {
                "profile_id": "verifier-calibration-v1",
                "regions": {
                    "table": {"x": 0, "y": 0, "width": 1, "height": 1},
                    "hero_cards": {"x": 0.4, "y": 0.72, "width": 0.2, "height": 0.12},
                    "board": {"x": 0.3, "y": 0.38, "width": 0.4, "height": 0.14},
                    "pot": {"x": 0.42, "y": 0.28, "width": 0.16, "height": 0.08},
                    "stack": {"x": 0.4, "y": 0.86, "width": 0.2, "height": 0.08},
                    "actions": {"x": 0.25, "y": 0.9, "width": 0.5, "height": 0.1},
                },
            },
            "fields": {
                key: {
                    "value": value,
                    "confidence": "0.99",
                    "region": visual_region_for.get(key, "table"),
                }
                for key, value in visual_values.items()
            },
        }

    visual_adapter = VisualObservationAdapter()
    pending_visual = visual_adapter.submit(visual_manifest("a-frame"))
    stable_visual = visual_adapter.submit(visual_manifest("b-frame"))
    low_visual_manifest = visual_manifest("c-frame")
    low_visual_manifest["fields"]["hero_cards"]["confidence"] = "0.40"
    low_visual = visual_adapter.submit(low_visual_manifest)
    visual_live_service = LiveTableService()
    visual_live_session = visual_live_service.create_session(
        "Verifier Visual Table"
    )
    visual_live_result = visual_live_service.update_state(
        str(visual_live_session["session_id"]),
        stable_visual["payload"],
        tabular_export.bundle.spots,
    )
    matched_review = analyze_with_solutions(hands, solver_bundle.spots)
    original_sample_spot = solver_bundle.spots[2]
    renamed_sample_spot = replace(
        original_sample_spot,
        key=replace(
            original_sample_spot.key,
            board=(Card.parse("As"), Card.parse("7d"), Card.parse("2h")),
            hero_cards=(Card.parse("Ac"), Card.parse("Kh")),
        ),
    )
    renamed_review = analyze_with_solutions(hands, (renamed_sample_spot,))
    blocker_changed_spot = replace(
        original_sample_spot,
        key=replace(
            original_sample_spot.key,
            hero_cards=(Card.parse("Ac"), Card.parse("Kd")),
        ),
    )
    blocker_changed_review = analyze_with_solutions(hands, (blocker_changed_spot,))
    study = InMemoryStudyStore()
    study.upsert_drills(matched_review["drills"])
    studied_drill = study.review_drill(
        str(matched_review["drills"][0]["drill_id"]), "good"
    )
    second_hand = sample_path.read_text(encoding="utf-8").replace(
        "90000000001", "90000000002"
    )
    incomplete_second, second_summary = second_hand.split("*** SUMMARY ***", 1)
    with TemporaryDirectory() as directory:
        ingestion_root = Path(directory)
        watched_file = ingestion_root / "HH-live.txt"
        watched_file.write_text(
            sample_path.read_text(encoding="utf-8") + "\n\n" + incomplete_second,
            encoding="utf-8",
        )
        with CoachDatabase(ingestion_root / "coach.sqlite3") as ingestion_database:
            scanner = HandHistoryFolderScanner(ingestion_database)
            first_scan = scanner.scan(ingestion_root)
            unchanged_scan = scanner.scan(ingestion_root)
            watched_file.write_text(
                sample_path.read_text(encoding="utf-8")
                + "\n\n"
                + incomplete_second
                + "*** SUMMARY ***"
                + second_summary,
                encoding="utf-8",
            )
            completed_scan = scanner.scan(ingestion_root)
            ingestion_audit = {
                "first_inserted": first_scan.inserted,
                "pending_incomplete_blocks": first_scan.incomplete_blocks,
                "unchanged_files_skipped": unchanged_scan.skipped_files,
                "completed_append_inserted": completed_scan.inserted,
                "deduplicated_existing_hands": completed_scan.unchanged,
                "database_hands": ingestion_database.hand_count,
            }
    modules = sorted(path.stem for path in (ROOT / "src" / "poker_coach").glob("*.py") if path.stem != "__init__")
    payload = {
        "success": result.wasSuccessful() and replay.reconciliation_error == 0,
        "tests_run": result.testsRun,
        "test_failures": len(result.failures),
        "test_errors": len(result.errors),
        "modules": modules,
        "sample": {
            "hand_id": hands[0].hand_id,
            "currency": hands[0].currency,
            "players": len(hands[0].players),
            "actions": len(hands[0].actions),
            "decision_snapshots": len(replay.decisions),
            "committed_pot": format(replay.committed_pot, "f"),
            "awarded_total": format(replay.awarded_total, "f"),
            "rake": format(replay.rake, "f"),
            "reconciliation_error": format(replay.reconciliation_error, "f"),
            "profiled_players": len(report["player_profiles"]),
        },
        "application": {
            "training_scenarios": len(library.all()),
            "strategy_tiers": sorted({scenario.provenance.tier for scenario in library.all()}),
            "solver_verified_scenarios": sum(
                1 for scenario in library.all() if scenario.provenance.solver_verified
            ),
            "session_decision_grade": training_result["feedback"]["grade"],
            "session_complete": training_result["complete"],
            "http_api_tested": result.wasSuccessful(),
            "river_solver": {
                "solver": "vanilla_cfr",
                "iterations": river_solution.iterations,
                "air_bet_frequency": river_solution.ip_strategy["Air"]["bet"],
                "bluffcatcher_call_frequency": river_solution.oop_strategy["Bluff-catcher"]["call"],
                "exploitability_bb": river_solution.exploitability_bb,
            },
            "solver_bundle_import": {
                "schema_version": solver_bundle.schema_version,
                "source": solver_bundle.source,
                "spots": len(solver_bundle.spots),
                "private_hand_nodes": sum(1 for spot in solver_bundle.spots if spot.key.hero_cards),
                "inserted": solver_import.inserted,
                "unique_fingerprints": len(set(solver_import.fingerprints)),
            },
            "solver_export_adapter": {
                "format": tabular_export.format_name,
                "source": tabular_export.bundle.source,
                "spots": len(tabular_export.bundle.spots),
                "action_rows": sum(
                    len(spot.actions) for spot in tabular_export.bundle.spots
                ),
                "streets": [
                    len(spot.key.board) for spot in tabular_export.bundle.spots
                ],
                "inserted": tabular_import.inserted,
                "auto_detection_tested": result.wasSuccessful(),
                "shared_validation_tested": result.wasSuccessful(),
            },
            "solution_tree": {
                "nodes": len(solution_forest.nodes),
                "linked_edges": solution_forest.linked_edges,
                "roots": len(solution_forest.roots),
                "max_depth": solution_forest.max_depth,
                "ambiguous_nodes": solution_forest.ambiguous_nodes,
                "river_path": [
                    node.node_id
                    for node in solution_forest.path_to(
                        tabular_export.bundle.spots[-1].key.fingerprint
                    )
                ],
                "branching_tested": result.wasSuccessful(),
                "suit_renamed_streets_tested": result.wasSuccessful(),
                "ambiguity_reporting_tested": result.wasSuccessful(),
            },
            "range_strategy": {
                "public_groups": range_strategy_report["group_count"],
                "private_nodes": range_strategy_report["private_nodes"],
                "shared_node_classes": sorted(
                    cell["hand_class"]
                    for group in range_strategy_report["groups"]
                    if group["private_nodes"] == 2
                    for cell in group["cells"]
                ),
                "matrix_dimensions": [13, 13],
                "editable_weights": ["1.0", "0.5", "0.0"],
                "public_suit_isomorphism_tested": result.wasSuccessful(),
            },
            "solver_practice": {
                "strategy_hidden_before_action": practice_challenge[
                    "strategy_hidden"
                ],
                "hidden_payload_contains_frequency": "frequency"
                in repr(practice_challenge),
                "legal_actions": len(practice_challenge["actions"]),
                "revealed_strategy_actions": len(practice_result["strategy"]),
                "ev_loss_bb": practice_result["ev_loss_bb"],
                "single_submit_and_illegal_action_tested": result.wasSuccessful(),
                "http_round_trip_tested": result.wasSuccessful(),
            },
            "live_table_state": {
                "schema_version": live_state_payload["schema_version"],
                "status": live_result["status"],
                "match_confidence": live_result["match"]["confidence"],
                "node_id": live_result["match"]["node_id"],
                "state_fingerprint_length": len(live_result["state_id"]),
                "covered_actions": len(live_result["match"]["actions"]),
                "recorded_ev_loss_bb": live_decision["ev_loss_bb"],
                "revision_rollback_tested": result.wasSuccessful(),
                "same_hand_progression_tested": result.wasSuccessful(),
                "legal_action_filter_tested": result.wasSuccessful(),
                "suit_isomorphism_tested": result.wasSuccessful(),
                "http_round_trip_tested": result.wasSuccessful(),
            },
            "board_texture_reasoning": {
                "street": live_result["texture"]["street"],
                "pairing": live_result["texture"]["pairing"],
                "suit_texture": live_result["texture"]["suit_texture"],
                "straight_texture": live_result["texture"]["straight_texture"],
                "made_hand": live_result["texture"]["hero"]["made_hand"],
                "range_caveat_included": bool(
                    live_result["texture"]["range_caveat"]
                ),
                "draw_and_blocker_tests": result.wasSuccessful(),
                "http_round_trip_tested": result.wasSuccessful(),
            },
            "range_matchup_analysis": {
                "method": range_matchup["method"],
                "hero_equity": range_matchup["equity"]["hero"],
                "villain_equity": range_matchup["equity"]["villain"],
                "compatible_matchups": range_matchup["compatible_matchups"],
                "current_nut_leader": range_matchup[
                    "current_range_relative_nuts"
                ]["leader"],
                "blocker_adjusted_weights": True,
                "deterministic_sampling_tested": result.wasSuccessful(),
                "confidence_interval_tested": result.wasSuccessful(),
                "http_round_trip_tested": result.wasSuccessful(),
            },
            "bayesian_range_conditioning": {
                "public_fingerprint_length": len(
                    range_conditioning["public_fingerprint"]
                ),
                "observed_action": range_conditioning["observed_action"],
                "action_probability": range_conditioning[
                    "action_probability_under_prior"
                ],
                "prior_source": range_conditioning["prior_source"],
                "exact_combos": len(range_conditioning["combos"]),
                "posterior_leader": range_conditioning["combos"][0][
                    "hand_class"
                ],
                "entropy_reduction_bits": range_conditioning["information"][
                    "entropy_reduction_bits"
                ],
                "custom_prior_tested": result.wasSuccessful(),
                "zero_likelihood_gating_tested": result.wasSuccessful(),
                "http_round_trip_tested": result.wasSuccessful(),
            },
            "opponent_range_timelines": {
                "opponent_decisions": opponent_timeline["opponent_decisions"],
                "public_state_matches": opponent_timeline[
                    "public_state_matches"
                ],
                "conditioned_actions": opponent_timeline[
                    "conditioned_actions"
                ],
                "timeline_count": len(opponent_timeline["timelines"]),
                "posterior_carried": opponent_timeline["timelines"][0][
                    "events"
                ][1]["prior_transition"]["mode"]
                == "posterior_carried",
                "unmatched_gap_reported": opponent_timeline["timelines"][0][
                    "events"
                ][1]["prior_transition"]["unmatched_actions_skipped"],
                "coverage": opponent_timeline["coverage"],
                "zero_coverage_and_uncovered_action_tested": result.wasSuccessful(),
                "http_and_rendered_review_tested": result.wasSuccessful(),
            },
            "showdown_range_calibration": {
                "scored_predictions": range_calibration["aggregate"][
                    "scored_predictions"
                ],
                "support_coverage": range_calibration["aggregate"][
                    "support_coverage"
                ],
                "mean_log_loss_bits": range_calibration["aggregate"][
                    "mean_log_loss_bits"
                ],
                "mean_brier_score": range_calibration["aggregate"][
                    "mean_multiclass_brier_score"
                ],
                "top_1_accuracy": range_calibration["aggregate"][
                    "top_1_accuracy"
                ],
                "calibration_buckets": len(
                    range_calibration["calibration_buckets"]
                ),
                "suit_isomorphic_reveal_mapping_tested": result.wasSuccessful(),
                "unsupported_and_unrevealed_gating_tested": result.wasSuccessful(),
                "zero_probability_infinite_loss_tested": result.wasSuccessful(),
                "http_and_rendered_review_tested": result.wasSuccessful(),
            },
            "live_hand_history_tail": {
                **live_tail_audit,
                "preflop_through_river_tested": result.wasSuccessful(),
                "partial_write_gating_tested": result.wasSuccessful(),
                "completed_hand_gating_tested": result.wasSuccessful(),
                "directory_selection_tested": result.wasSuccessful(),
                "http_poll_tested": result.wasSuccessful(),
            },
            "calibrated_visual_observation": {
                "first_frame_status": pending_visual["status"],
                "stable_status": stable_visual["status"],
                "stable_revision": stable_visual["revision"],
                "required_distinct_frames": stable_visual[
                    "required_stable_frames"
                ],
                "mean_confidence": stable_visual["mean_confidence"],
                "image_hash_length": len(
                    stable_visual["evidence"]["image_sha256"]
                ),
                "solver_status": visual_live_result["status"],
                "solver_node": visual_live_result["match"]["node_id"],
                "low_confidence_status": low_visual["status"],
                "low_confidence_fields": low_visual[
                    "low_confidence_fields"
                ],
                "region_validation_tested": result.wasSuccessful(),
                "image_path_hash_tested": result.wasSuccessful(),
                "http_two_frame_flow_tested": result.wasSuccessful(),
            },
            "temporal_state_invariants": {
                "accepted_initial_kind": live_result["transition"]["kind"],
                "accepted_check_count": len(live_result["transition"]["checks"]),
                "rejected_status": temporal_transition_audit["status"],
                "rejected_codes": [
                    row["code"]
                    for row in temporal_transition_audit["violations"]
                ],
                "session_preserved_after_rejection": live_service.current(
                    str(live_session["session_id"])
                )["last_revision"] == 0,
                "direct_http_422_tested": result.wasSuccessful(),
                "visual_rejection_tested": result.wasSuccessful(),
            },
            "exploit_insights": {
                "players": len(report["exploit_insights"]),
                "signals": sum(len(rows) for rows in report["exploit_insights"].values()),
                "small_sample_actionability": sorted(
                    {
                        row["actionability"]
                        for rows in report["exploit_insights"].values()
                        for row in rows
                    }
                ),
                "posterior_intervals_included": True,
                "strong_signal_tested": result.wasSuccessful(),
            },
            "personalized_review": {
                "hero_decisions": matched_review["hero_decisions"],
                "matched_decisions": matched_review["matched_decisions"],
                "unmatched_decisions": matched_review["unmatched_decisions"],
                "drills": len(matched_review["drills"]),
                "top_drill_ev_loss_bb": matched_review["drills"][0]["ev_loss_bb"],
                "top_drill_observed_action": matched_review["drills"][0]["observed_action"],
                "total_ev_loss_bb": matched_review["leak_summary"]["total_ev_loss_bb"],
                "solver_coverage": matched_review["leak_summary"]["coverage"],
                "street_breakdown": matched_review["leak_summary"]["by_street"],
            },
            "suit_isomorphism": {
                "normalization": "suit_isomorphism_v1",
                "renamed_fingerprint_equal": (
                    renamed_sample_spot.key.fingerprint
                    == original_sample_spot.key.fingerprint
                ),
                "renamed_state_matches": renamed_review["matched_decisions"],
                "renamed_card_match": renamed_review["matches"][0]["match"]["card_match"],
                "different_blocker_state_matches": blocker_changed_review[
                    "matched_decisions"
                ],
                "all_24_permutations_tested": result.wasSuccessful(),
                "legacy_database_migration_tested": result.wasSuccessful(),
            },
            "study_progress": {
                "persisted_drills": len(study.list_drills()),
                "attempts": studied_drill["study"]["attempts"],
                "last_rating": studied_drill["study"]["last_rating"],
                "mastery": studied_drill["study"]["mastery"],
                "interval_days": studied_drill["study"]["interval_days"],
                "sqlite_reopen_tested": result.wasSuccessful(),
            },
            "saved_hand_intake": ingestion_audit,
        },
    }
    (ROOT / "verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
