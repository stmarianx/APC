from __future__ import annotations

import argparse
import copy
import hashlib
import json
import mimetypes
import re
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qs, urlparse

from .pokerstars import PokerStarsParseError, PokerStarsParser
from .report import analyze_hands
from .river_solver import solve_akq_river
from .solver_import import SolverBundle, SolverBundleImporter, solved_spot_to_dict
from .solutions import InMemorySolutionStore, SolutionStore, SolvedSpot
from .matching import analyze_with_solutions
from .storage import CoachDatabase
from .study import InMemoryStudyStore, StudyStore
from .trainer import TrainingService
from .ingest import HandHistoryFolderScanner
from .solver_adapters import (
    BUNDLE_JSON_V1,
    TABULAR_CSV_V1,
    SolverExportRegistry,
)
from .solution_tree import SolutionForest
from .range_strategy import aggregate_range_strategies
from .solver_practice import SolverPracticeService
from .live_state import LiveTableService
from .state_transition import StateTransitionError
from .live_capture import PokerStarsLiveTailAdapter
from .visual_capture import VisualObservationAdapter
from .models import Card
from .ranges import parse_range
from .range_matchup import analyze_range_matchup
from .range_inference import condition_solution_range
from .range_timeline import build_opponent_range_timelines
from .range_calibration import score_opponent_range_timelines
from .range_strategy import public_node_fingerprint


MAX_REQUEST_BYTES = 5 * 1024 * 1024
SESSION_RE = re.compile(r"^/api/sessions/([0-9a-f]{32})$")
DECISION_RE = re.compile(r"^/api/sessions/([0-9a-f]{32})/decisions$")
SCENARIO_RE = re.compile(r"^/api/scenarios/([a-z0-9_-]+)$")
DRILL_ATTEMPT_RE = re.compile(r"^/api/drills/([0-9a-f]{20})/attempts$")
SOLUTION_PATH_RE = re.compile(r"^/api/solution-tree/([0-9a-f]{64})/path$")
SOLVER_PRACTICE_DECISION_RE = re.compile(
    r"^/api/solver-practice/sessions/([0-9a-f]{32})/decisions$"
)
LIVE_SESSION_RE = re.compile(r"^/api/live/sessions/([0-9a-f]{32})$")
LIVE_STATE_RE = re.compile(r"^/api/live/sessions/([0-9a-f]{32})/states$")
LIVE_DECISION_RE = re.compile(
    r"^/api/live/sessions/([0-9a-f]{32})/decisions$"
)
LIVE_CAPTURE_POLL_RE = re.compile(
    r"^/api/live/sessions/([0-9a-f]{32})/capture/polls$"
)
LIVE_VISUAL_OBSERVATION_RE = re.compile(
    r"^/api/live/sessions/([0-9a-f]{32})/visual-observations$"
)
RANGE_CONDITION_RE = re.compile(
    r"^/api/range-strategies/([0-9a-f]{64})/condition$"
)


class CoachApplication:
    def __init__(
        self,
        training: TrainingService | None = None,
        solution_store: SolutionStore | None = None,
        study_store: StudyStore | None = None,
        history_store: CoachDatabase | None = None,
    ) -> None:
        self.training = training or TrainingService()
        self.parser = PokerStarsParser()
        self.solution_store = solution_store or InMemorySolutionStore()
        self.study_store = study_store or InMemoryStudyStore()
        self.history_store = history_store
        self.solver_importer = SolverBundleImporter()
        self.solver_exports = SolverExportRegistry(self.solver_importer)
        self.solver_practice = SolverPracticeService()
        self.live_tables = LiveTableService()
        self.live_capture = PokerStarsLiveTailAdapter()
        self.visual_capture = VisualObservationAdapter()
        self._analysis_lock = threading.RLock()
        self._cached_library_report: dict[str, object] | None = None
        self._cached_library_hand_count = -1

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "application": "poker-coach-lab",
            "api_version": "1.11.0",
            "scenario_count": len(self.training.library.all()),
            "mode": "training_solver_and_saved_hand_review",
            "study_store": type(self.study_store).__name__,
            "persistent_hand_library": self.history_store is not None,
            "database_hands": 0 if self.history_store is None else self.history_store.hand_count,
            "solver_export_formats": [BUNDLE_JSON_V1, TABULAR_CSV_V1],
            "solver_practice": True,
            "live_table_states": True,
            "live_hand_history_tail": True,
            "calibrated_visual_observations": True,
            "board_texture_analysis": True,
            "range_matchup_analysis": True,
            "bayesian_range_conditioning": True,
            "opponent_range_timelines": True,
            "showdown_range_calibration": True,
            "temporal_state_invariants": True,
            "latency_bounded_strategy_routing": True,
        }

    def analyze_range_matchup(self, payload: dict[str, Any]) -> dict[str, object]:
        board_value = payload.get("board", [])
        if not isinstance(board_value, list) or not all(
            isinstance(token, str) for token in board_value
        ):
            raise ValueError("board must be an array of card strings")
        board = tuple(Card.parse(token) for token in board_value)
        hero_value = payload.get("hero_range")
        villain_value = payload.get("villain_range")
        if not isinstance(hero_value, str) or not hero_value.strip():
            raise ValueError("hero_range is required")
        if not isinstance(villain_value, str) or not villain_value.strip():
            raise ValueError("villain_range is required")
        hero_notation = hero_value.strip()
        villain_notation = villain_value.strip()
        samples = payload.get("samples", 20_000)
        seed = payload.get("seed", 1)
        max_exact = payload.get("max_exact_outcomes", 250_000)
        if isinstance(samples, bool) or not isinstance(samples, int):
            raise ValueError("samples must be an integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(max_exact, bool) or not isinstance(max_exact, int):
            raise ValueError("max_exact_outcomes must be an integer")
        hero = parse_range(hero_notation, dead=board)
        villain = parse_range(villain_notation, dead=board)
        result = analyze_range_matchup(
            hero,
            villain,
            board,
            samples=samples,
            seed=seed,
            max_exact_outcomes=max_exact,
        ).to_dict()
        result["input"] = {
            "hero_range": hero_notation,
            "villain_range": villain_notation,
        }
        return result

    def condition_range(
        self,
        public_fingerprint: str,
        observed_action: str,
        prior_weights: object = None,
    ) -> dict[str, object]:
        nodes = tuple(
            spot
            for spot in self._solutions()
            if public_node_fingerprint(spot) == public_fingerprint
        )
        if not nodes:
            raise KeyError(f"Unknown public range node: {public_fingerprint}")
        if prior_weights is not None and not isinstance(prior_weights, dict):
            raise ValueError("prior_weights must be an object keyed by exact combo")
        return condition_solution_range(
            nodes,
            observed_action,
            prior_weights=prior_weights,
        )

    def analyze_text(self, source: str) -> dict[str, object]:
        if not source.strip():
            raise ValueError("hand_history cannot be empty")
        hands = self.parser.parse_many(source)
        import_result = None
        if self.history_store is not None:
            import_result = self.history_store.import_text(source)
            report, _ = self._library_analysis(
                force=bool(import_result.inserted or import_result.updated)
            )
        else:
            report = self._analyze_hands(hands)
        if import_result is not None:
            report["import"] = {
                "inserted": import_result.inserted,
                "updated": import_result.updated,
                "unchanged": import_result.unchanged,
                "hand_ids": list(import_result.hand_ids),
                "database_hands": self.history_store.hand_count,
            }
        return report

    def scan_folder(self, folder: str, *, recursive: bool = False) -> dict[str, object]:
        if self.history_store is None:
            raise ValueError("Folder ingestion requires the persistent database mode")
        scan = HandHistoryFolderScanner(self.history_store).scan(folder, recursive=recursive)
        report, reused = self._library_analysis(force=bool(scan.inserted or scan.updated))
        report["scan"] = scan.to_dict()
        report["scan"]["database_hands"] = self.history_store.hand_count
        report["scan"]["analysis_reused"] = reused
        return report

    def library_report(self) -> dict[str, object]:
        if self.history_store is None:
            raise ValueError("Hand library requires the persistent database mode")
        report, _ = self._library_analysis()
        return report

    def _library_analysis(self, *, force: bool = False) -> tuple[dict[str, object], bool]:
        if self.history_store is None:
            raise ValueError("Hand library requires the persistent database mode")
        with self._analysis_lock:
            hand_count = self.history_store.hand_count
            if (
                not force
                and self._cached_library_report is not None
                and self._cached_library_hand_count == hand_count
            ):
                report = copy.deepcopy(self._cached_library_report)
                report["study_queue"] = self.study_store.list_drills()
                return report, True
            report = self._analyze_hands(self.history_store.load_hands())
            self._cached_library_report = copy.deepcopy(report)
            self._cached_library_hand_count = hand_count
            return report, False

    def _analyze_hands(self, hands: tuple[object, ...]) -> dict[str, object]:
        report = analyze_hands(hands)
        review = analyze_with_solutions(hands, self._solutions())
        opponent_ranges = build_opponent_range_timelines(
            hands, self._solutions()
        )
        range_calibration = score_opponent_range_timelines(
            hands, self._solutions(), opponent_ranges
        )
        self.study_store.upsert_drills(review["drills"])
        report["solution_review"] = review
        report["opponent_range_review"] = opponent_ranges
        report["range_calibration"] = range_calibration
        report["study_queue"] = self.study_store.list_drills()
        return report

    def _solutions(self) -> tuple[SolvedSpot, ...]:
        all_method = getattr(self.solution_store, "all", None)
        if callable(all_method):
            return tuple(all_method())
        list_method = getattr(self.solution_store, "list_solutions", None)
        if callable(list_method):
            return tuple(list_method())
        return ()

    def solution_tree(self) -> dict[str, object]:
        return SolutionForest(self._solutions()).to_dict()

    def range_strategies(self) -> dict[str, object]:
        return aggregate_range_strategies(self._solutions())

    def solution_path(self, fingerprint: str) -> dict[str, object]:
        forest = SolutionForest(self._solutions())
        return {
            "fingerprint": fingerprint,
            "path": [node.to_dict() for node in forest.path_to(fingerprint)],
        }

    def create_solver_practice(
        self, public_fingerprint: str | None = None
    ) -> dict[str, object]:
        return self.solver_practice.create(
            self._solutions(), public_fingerprint=public_fingerprint
        )

    def submit_solver_practice(
        self, session_id: str, action_id: str
    ) -> dict[str, object]:
        return self.solver_practice.submit(session_id, action_id)

    def sample_live_state(self) -> dict[str, object]:
        solutions = self._solutions()
        if not solutions:
            sample = files("poker_coach").joinpath("data/sample_solver_export.csv")
            solutions = self.solver_exports.parse_text(
                sample.read_text(encoding="utf-8-sig"),
                format_name=TABULAR_CSV_V1,
            ).bundle.spots
        spot = max(solutions, key=lambda row: (len(row.key.board), row.node_id))
        key = spot.key
        return {
            "schema_version": "1.0.0",
            "table_id": "Play Money Table 1",
            "hand_id": "live-demo-1",
            "revision": 0,
            "game": key.game,
            "players": key.players,
            "hero_position": key.hero_position,
            "effective_stack_bb": format(key.effective_stack_bb, "f"),
            "pot_bb": format(key.pot_bb, "f"),
            "to_call_bb": "0",
            "board": [str(card) for card in key.board],
            "hero_cards": [str(card) for card in key.hero_cards],
            "action_history": list(key.action_history),
            "legal_actions": [action.action for action in spot.actions],
            "rake_model": key.rake_model,
            "utility_model": key.utility_model,
            "source": "demo_normalized_feed",
        }

    def sample_visual_observation(self) -> dict[str, object]:
        state = self.sample_live_state()
        state["table_id"] = "Visual Play Table"
        state["hand_id"] = "visual-demo-1"
        frame_id = uuid.uuid4().hex
        values = {
            key: state[key]
            for key in (
                "table_id",
                "hand_id",
                "game",
                "players",
                "hero_position",
                "effective_stack_bb",
                "pot_bb",
                "to_call_bb",
                "board",
                "hero_cards",
                "action_history",
                "legal_actions",
                "rake_model",
                "utility_model",
            )
        }
        region_for = {
            "effective_stack_bb": "stack",
            "pot_bb": "pot",
            "to_call_bb": "actions",
            "board": "board",
            "hero_cards": "hero_cards",
            "legal_actions": "actions",
        }
        return {
            "schema_version": "1.0.0",
            "provider": "example_calibrated_visual_bridge",
            "provider_version": "1.0",
            "frame": {
                "frame_id": frame_id,
                "image_sha256": hashlib.sha256(frame_id.encode()).hexdigest(),
            },
            "calibration": {
                "profile_id": "pokerstars-example-1280x720-v1",
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
                    "region": region_for.get(key, "table"),
                }
                for key, value in values.items()
            },
        }

    def create_live_session(self, table_id: str) -> dict[str, object]:
        return self.live_tables.create_session(table_id)

    def update_live_state(
        self, session_id: str, payload: dict[str, object]
    ) -> dict[str, object]:
        return self.live_tables.update_state(session_id, payload, self._solutions())

    def record_live_decision(
        self, session_id: str, revision: int, action_id: str
    ) -> dict[str, object]:
        return self.live_tables.record_decision(session_id, revision, action_id)

    def poll_live_history(
        self, session_id: str, source_path: str
    ) -> dict[str, object]:
        current = self.live_tables.current(session_id)
        poll = self.live_capture.poll(
            source_path, table_id=str(current["table_id"])
        )
        response = poll.to_dict()
        if poll.status == "state_ready" and poll.payload is not None:
            should_update = (
                current["status"] == "awaiting_state"
                or current.get("last_revision") != poll.revision
                or current.get("state", {}).get("hand_id") != poll.hand_id
            )
            response["analysis"] = (
                self.update_live_state(session_id, poll.payload)
                if should_update
                else current
            )
        else:
            response["analysis"] = None
        return response

    def submit_visual_observation(
        self, session_id: str, payload: dict[str, object]
    ) -> dict[str, object]:
        current = self.live_tables.current(session_id)
        visual = self.visual_capture.submit(payload)
        if visual["status"] == "state_ready" and visual["payload"] is not None:
            state = visual["payload"]
            if state["table_id"] != current["table_id"]:
                raise ValueError("Visual table_id does not match the live session")
            should_update = (
                current["status"] == "awaiting_state"
                or current.get("last_revision") != visual["revision"]
                or current.get("state", {}).get("hand_id") != state["hand_id"]
            )
            try:
                visual["analysis"] = (
                    self.update_live_state(session_id, state)
                    if should_update
                    else current
                )
                visual["transition"] = visual["analysis"].get("transition")
            except StateTransitionError as error:
                visual["status"] = "invalid_transition"
                visual["changed"] = False
                visual["proposed_revision"] = visual.get("revision")
                visual["revision"] = None
                visual["payload"] = None
                visual["analysis"] = None
                visual["transition"] = error.audit
        else:
            visual["analysis"] = None
            visual["transition"] = None
        return visual

    def import_solver_bundle(self, payload: dict[str, Any]) -> dict[str, object]:
        bundle = self.solver_importer.parse_dict(payload)
        return self._import_solver_bundle(bundle, format_name=BUNDLE_JSON_V1)

    def import_solver_export(
        self, content: str, *, format_name: str = "auto"
    ) -> dict[str, object]:
        if not content.strip():
            raise ValueError("solver export content cannot be empty")
        parsed = self.solver_exports.parse_text(content, format_name=format_name)
        return self._import_solver_bundle(parsed.bundle, format_name=parsed.format_name)

    def _import_solver_bundle(
        self, bundle: SolverBundle, *, format_name: str
    ) -> dict[str, object]:
        result = self.solver_importer.import_into(self.solution_store, bundle)
        with self._analysis_lock:
            self._cached_library_report = None
            self._cached_library_hand_count = -1
        response = result.to_dict()
        response["format"] = format_name
        response["solutions"] = [solved_spot_to_dict(spot) for spot in bundle.spots]
        response["tree"] = SolutionForest(bundle.spots).to_dict()
        response["range_strategies"] = aggregate_range_strategies(bundle.spots)
        return response


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def make_handler(application: CoachApplication, *, quiet: bool = True) -> type[BaseHTTPRequestHandler]:
    class CoachRequestHandler(BaseHTTPRequestHandler):
        server_version = "PokerCoachLab/0.2"

        def log_message(self, format: str, *args: object) -> None:
            if not quiet:
                super().log_message(format, *args)

        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store" if self.path.startswith("/api/") else "no-cache")
            super().end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/health":
                    self._send_json(HTTPStatus.OK, application.health())
                    return
                if parsed.path == "/api/scenarios":
                    self._send_json(HTTPStatus.OK, {"scenarios": application.training.list_scenarios()})
                    return
                if parsed.path == "/api/drills":
                    due_only = parse_qs(parsed.query).get("due", ["0"])[0] == "1"
                    drills = application.study_store.list_drills(due_only=due_only)
                    self._send_json(HTTPStatus.OK, {"drills": drills, "due_only": due_only})
                    return
                if parsed.path == "/api/hand-history-library":
                    self._send_json(HTTPStatus.OK, application.library_report())
                    return
                if parsed.path == "/api/solution-tree":
                    self._send_json(HTTPStatus.OK, application.solution_tree())
                    return
                if parsed.path == "/api/range-strategies":
                    self._send_json(HTTPStatus.OK, application.range_strategies())
                    return
                if parsed.path == "/api/sample-hand":
                    sample = files("poker_coach").joinpath("data/sample_play_money_hand.txt")
                    self._send_json(
                        HTTPStatus.OK,
                        {"hand_history": sample.read_text(encoding="utf-8-sig")},
                    )
                    return
                if parsed.path == "/api/sample-live-state":
                    self._send_json(HTTPStatus.OK, application.sample_live_state())
                    return
                if parsed.path == "/api/sample-visual-observation":
                    self._send_json(
                        HTTPStatus.OK, application.sample_visual_observation()
                    )
                    return
                if parsed.path == "/api/sample-solver-bundle":
                    sample = files("poker_coach").joinpath("data/sample_solver_bundle.json")
                    self._send_json(HTTPStatus.OK, json.loads(sample.read_text(encoding="utf-8")))
                    return
                if parsed.path == "/api/sample-solver-export":
                    format_name = parse_qs(parsed.query).get(
                        "format", [BUNDLE_JSON_V1]
                    )[0]
                    if format_name == BUNDLE_JSON_V1:
                        sample = files("poker_coach").joinpath(
                            "data/sample_solver_bundle.json"
                        )
                    elif format_name == TABULAR_CSV_V1:
                        sample = files("poker_coach").joinpath(
                            "data/sample_solver_export.csv"
                        )
                    else:
                        raise ValueError(f"Unsupported solver export format: {format_name}")
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "format": format_name,
                            "content": sample.read_text(encoding="utf-8-sig"),
                        },
                    )
                    return
                scenario_match = SCENARIO_RE.match(parsed.path)
                if scenario_match:
                    reveal = parse_qs(parsed.query).get("reveal", ["0"])[0] == "1"
                    scenario = application.training.library.get(scenario_match.group(1))
                    self._send_json(HTTPStatus.OK, scenario.to_dict(reveal_strategy=reveal))
                    return
                session_match = SESSION_RE.match(parsed.path)
                if session_match:
                    self._send_json(
                        HTTPStatus.OK,
                        application.training.session_state(session_match.group(1)),
                    )
                    return
                solution_path_match = SOLUTION_PATH_RE.match(parsed.path)
                if solution_path_match:
                    self._send_json(
                        HTTPStatus.OK,
                        application.solution_path(solution_path_match.group(1)),
                    )
                    return
                live_session_match = LIVE_SESSION_RE.match(parsed.path)
                if live_session_match:
                    self._send_json(
                        HTTPStatus.OK,
                        application.live_tables.current(live_session_match.group(1)),
                    )
                    return
                if parsed.path.startswith("/api/"):
                    self._send_error(HTTPStatus.NOT_FOUND, "API route not found")
                    return
                self._serve_asset(parsed.path)
            except KeyError as error:
                self._send_error(HTTPStatus.NOT_FOUND, str(error).strip("'"))
            except ValueError as error:
                self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(error))

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                body = self._read_json()
                if parsed.path == "/api/sessions":
                    count = body.get("count")
                    seed = body.get("seed")
                    result = application.training.create_session(
                        count=None if count is None else int(count),
                        seed=None if seed is None else int(seed),
                    )
                    self._send_json(HTTPStatus.CREATED, result)
                    return
                if parsed.path == "/api/solver-practice/sessions":
                    public_fingerprint = str(body.get("public_fingerprint", "")).strip()
                    result = application.create_solver_practice(
                        public_fingerprint or None
                    )
                    self._send_json(HTTPStatus.CREATED, result)
                    return
                if parsed.path == "/api/live/sessions":
                    table_id = str(body.get("table_id", "")).strip()
                    result = application.create_live_session(table_id)
                    self._send_json(HTTPStatus.CREATED, result)
                    return
                range_condition_match = RANGE_CONDITION_RE.match(parsed.path)
                if range_condition_match:
                    observed_action = body.get("observed_action")
                    if not isinstance(observed_action, str) or not observed_action.strip():
                        raise ValueError("observed_action is required")
                    result = application.condition_range(
                        range_condition_match.group(1),
                        observed_action.strip(),
                        body.get("prior_weights"),
                    )
                    self._send_json(HTTPStatus.OK, result)
                    return
                live_capture_match = LIVE_CAPTURE_POLL_RE.match(parsed.path)
                if live_capture_match:
                    source_path = str(body.get("path", "")).strip()
                    if not source_path:
                        raise ValueError("path is required")
                    result = application.poll_live_history(
                        live_capture_match.group(1), source_path
                    )
                    self._send_json(HTTPStatus.OK, result)
                    return
                live_visual_match = LIVE_VISUAL_OBSERVATION_RE.match(parsed.path)
                if live_visual_match:
                    result = application.submit_visual_observation(
                        live_visual_match.group(1), body
                    )
                    self._send_json(HTTPStatus.OK, result)
                    return
                live_state_match = LIVE_STATE_RE.match(parsed.path)
                if live_state_match:
                    result = application.update_live_state(
                        live_state_match.group(1), body
                    )
                    self._send_json(HTTPStatus.OK, result)
                    return
                live_decision_match = LIVE_DECISION_RE.match(parsed.path)
                if live_decision_match:
                    action_id = str(body.get("action_id", "")).strip()
                    if not action_id:
                        raise ValueError("action_id is required")
                    revision = body.get("revision")
                    if isinstance(revision, bool) or not isinstance(revision, int):
                        raise ValueError("revision must be an integer")
                    result = application.record_live_decision(
                        live_decision_match.group(1), revision, action_id
                    )
                    self._send_json(HTTPStatus.OK, result)
                    return
                solver_practice_match = SOLVER_PRACTICE_DECISION_RE.match(
                    parsed.path
                )
                if solver_practice_match:
                    action_id = str(body.get("action_id", "")).strip()
                    if not action_id:
                        raise ValueError("action_id is required")
                    result = application.submit_solver_practice(
                        solver_practice_match.group(1), action_id
                    )
                    self._send_json(HTTPStatus.OK, result)
                    return
                decision_match = DECISION_RE.match(parsed.path)
                if decision_match:
                    action_id = str(body.get("action_id", ""))
                    if not action_id:
                        raise ValueError("action_id is required")
                    result = application.training.submit_decision(decision_match.group(1), action_id)
                    self._send_json(HTTPStatus.OK, result)
                    return
                if parsed.path == "/api/analyze-hand-history":
                    source = str(body.get("hand_history", ""))
                    self._send_json(HTTPStatus.OK, application.analyze_text(source))
                    return
                if parsed.path == "/api/scan-hand-history-folder":
                    folder = str(body.get("folder", "")).strip()
                    if not folder:
                        raise ValueError("folder is required")
                    recursive = bool(body.get("recursive", False))
                    self._send_json(
                        HTTPStatus.OK,
                        application.scan_folder(folder, recursive=recursive),
                    )
                    return
                if parsed.path == "/api/solve-river":
                    pot_bb = float(body.get("pot_bb", 10))
                    bet_bb = float(body.get("bet_bb", 10))
                    iterations = int(body.get("iterations", 50_000))
                    game, solution = solve_akq_river(
                        pot_bb=pot_bb,
                        bet_bb=bet_bb,
                        iterations=iterations,
                    )
                    result = solution.to_dict()
                    result["game"] = {
                        "name": "AKQ value-bluff river abstraction",
                        "pot_bb": game.pot_bb,
                        "bet_bb": game.bet_bb,
                        "ip_prior": {bucket.name: bucket.weight for bucket in game.ip_buckets},
                        "oop_prior": {bucket.name: bucket.weight for bucket in game.oop_buckets},
                        "assumptions": [
                            "heads-up river",
                            "IP may check or make one fixed-size bet",
                            "OOP may fold or call",
                            "equal Value/Air prior weights",
                            "zero-sum chip EV and no rake",
                        ],
                    }
                    self._send_json(HTTPStatus.OK, result)
                    return
                if parsed.path == "/api/analyze-range-matchup":
                    self._send_json(
                        HTTPStatus.OK, application.analyze_range_matchup(body)
                    )
                    return
                if parsed.path == "/api/import-solver-bundle":
                    self._send_json(HTTPStatus.OK, application.import_solver_bundle(body))
                    return
                if parsed.path == "/api/import-solver-export":
                    content = str(body.get("content", ""))
                    format_name = str(body.get("format", "auto"))
                    self._send_json(
                        HTTPStatus.OK,
                        application.import_solver_export(
                            content, format_name=format_name
                        ),
                    )
                    return
                drill_attempt_match = DRILL_ATTEMPT_RE.match(parsed.path)
                if drill_attempt_match:
                    rating = str(body.get("rating", ""))
                    if not rating:
                        raise ValueError("rating is required")
                    result = application.study_store.review_drill(
                        drill_attempt_match.group(1), rating
                    )
                    self._send_json(HTTPStatus.OK, result)
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "API route not found")
            except json.JSONDecodeError:
                self._send_error(HTTPStatus.BAD_REQUEST, "Request body must be valid JSON")
            except KeyError as error:
                self._send_error(HTTPStatus.NOT_FOUND, str(error).strip("'"))
            except StateTransitionError as error:
                self._send_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {
                        "error": str(error),
                        "status": int(HTTPStatus.UNPROCESSABLE_ENTITY),
                        "transition": error.audit,
                    },
                )
            except (ValueError, PokerStarsParseError) as error:
                self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(error))

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("Invalid Content-Length") from error
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError(f"Request body exceeds {MAX_REQUEST_BYTES} bytes")
            if length == 0:
                return {}
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Request JSON must be an object")
            return data

        def _send_json(self, status: HTTPStatus, payload: object) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json(status, {"error": message, "status": int(status)})

        def _serve_asset(self, path: str) -> None:
            asset_name = {
                "": "index.html",
                "/": "index.html",
                "/index.html": "index.html",
                "/app.js": "app.js",
                "/styles.css": "styles.css",
            }.get(path)
            if asset_name is None:
                self._send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                return
            resource = files("poker_coach").joinpath(f"web_assets/{asset_name}")
            body = resource.read_bytes()
            content_type = mimetypes.guess_type(asset_name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return CoachRequestHandler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    application: CoachApplication | None = None,
    *,
    quiet: bool = True,
) -> ThreadingHTTPServer:
    app = application or CoachApplication()
    server = ThreadingHTTPServer((host, port), make_handler(app, quiet=quiet))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Poker Coach Lab")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the app in the default browser")
    parser.add_argument(
        "--database",
        default="coach.sqlite3",
        help="SQLite file for hands, solutions, drills and study history",
    )
    args = parser.parse_args(argv)
    database = CoachDatabase(args.database)
    application = CoachApplication(
        solution_store=database,
        study_store=database,
        history_store=database,
    )
    server = create_server(args.host, args.port, application, quiet=False)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Poker Coach Lab running at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
