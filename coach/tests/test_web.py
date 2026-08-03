import copy
import json
from decimal import Decimal
from pathlib import Path
import threading
import unittest
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from poker_coach.web import CoachApplication, create_server
from poker_coach.storage import CoachDatabase


ROOT = Path(__file__).resolve().parents[1]


class WebApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = create_server("127.0.0.1", 0, CoachApplication())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, object]]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read())

    def test_health_and_static_application(self) -> None:
        status, health = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["scenario_count"], 6)
        self.assertEqual(health["api_version"], "1.11.0")
        self.assertTrue(health["board_texture_analysis"])
        self.assertTrue(health["range_matchup_analysis"])
        self.assertTrue(health["bayesian_range_conditioning"])
        self.assertTrue(health["opponent_range_timelines"])
        self.assertTrue(health["showdown_range_calibration"])
        self.assertTrue(health["temporal_state_invariants"])
        self.assertTrue(health["latency_bounded_strategy_routing"])
        with urlopen(self.base + "/", timeout=3) as response:
            html = response.read().decode("utf-8")
        self.assertIn("Poker Coach Lab", html)
        self.assertIn("River Solver Lab", html)
        self.assertIn("Import an external solution bundle", html)
        self.assertIn("Showdown range calibration", html)

    def test_session_decision_round_trip(self) -> None:
        status, session = self.request("/api/sessions", {"count": 1, "seed": 1})
        self.assertEqual(status, 201)
        action_id = session["scenario"]["actions"][0]["action_id"]
        status, result = self.request(
            f"/api/sessions/{session['session_id']}/decisions",
            {"action_id": action_id},
        )
        self.assertEqual(status, 200)
        self.assertTrue(result["complete"])
        self.assertIn("ev_loss_bb", result["feedback"])

    def test_saved_hand_analysis_endpoint(self) -> None:
        source = (ROOT / "examples" / "sample_play_money_hand.txt").read_text(encoding="utf-8")
        status, report = self.request("/api/analyze-hand-history", {"hand_history": source})
        self.assertEqual(status, 200)
        self.assertEqual(report["hands"], 1)
        self.assertEqual(report["hand_reports"][0]["reconciliation_error"], "0")

    def test_bundled_showdown_scores_range_calibration(self) -> None:
        payload = json.loads(
            (ROOT / "examples" / "sample_solver_bundle.json").read_text(
                encoding="utf-8"
            )
        )
        self.request("/api/import-solver-bundle", payload)
        _, sample = self.request("/api/sample-hand")
        status, report = self.request(
            "/api/analyze-hand-history",
            {"hand_history": sample["hand_history"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(report["hands"], 2)
        calibration = report["range_calibration"]
        self.assertEqual(calibration["aggregate"]["scored_predictions"], 1)
        self.assertEqual(calibration["aggregate"]["support_coverage"], "1")
        scored = next(
            timeline
            for timeline in calibration["timelines"]
            if timeline["status"] == "scored"
        )
        self.assertEqual(scored["revealed_cards"], ["Kc", "Qh"])
        self.assertEqual(scored["predictions"][0]["actual_combo"], "Kc Qh")

    def test_river_solver_endpoint(self) -> None:
        status, solution = self.request(
            "/api/solve-river",
            {"pot_bb": 10, "bet_bb": 10, "iterations": 20_000},
        )
        self.assertEqual(status, 200)
        self.assertEqual(solution["solver"], "vanilla_cfr")
        self.assertAlmostEqual(solution["strategy"]["ip"]["Air"]["bet"], 0.5, delta=0.04)
        self.assertLess(solution["exploitability_bb"], 0.04)

    def test_range_matchup_endpoint(self) -> None:
        status, result = self.request(
            "/api/analyze-range-matchup",
            {
                "board": ["2h", "3h", "4s", "9c", "7d"],
                "hero_range": "AsAd",
                "villain_range": "KcKd,QcQd",
                "samples": 2_000,
                "seed": 19,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["method"], "exact_enumeration")
        self.assertEqual(result["equity"]["hero"], "1")
        self.assertEqual(result["current_range_relative_nuts"]["leader"], "hero")
        self.assertEqual(result["input"]["hero_range"], "AsAd")

    def test_bayesian_range_conditioning_endpoint(self) -> None:
        payload = json.loads(
            (ROOT / "examples" / "sample_solver_bundle.json").read_text(
                encoding="utf-8"
            )
        )
        _, imported = self.request("/api/import-solver-bundle", payload)
        shared = next(
            group
            for group in imported["range_strategies"]["groups"]
            if group["private_nodes"] == 2
        )
        status, result = self.request(
            f"/api/range-strategies/{shared['public_fingerprint']}/condition",
            {
                "observed_action": "check",
                "prior_weights": {"Kc Qd": 1, "Jc Td": 1},
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            Decimal(result["action_probability_under_prior"]), Decimal("0.33")
        )
        rows = {row["hand_class"]: row for row in result["combos"]}
        self.assertGreater(
            Decimal(rows["JTo"]["posterior"]),
            Decimal(rows["KQo"]["posterior"]),
        )

    def test_solver_bundle_import_endpoint(self) -> None:
        payload = json.loads(
            (ROOT / "examples" / "sample_solver_bundle.json").read_text(encoding="utf-8")
        )
        status, result = self.request("/api/import-solver-bundle", payload)
        self.assertEqual(status, 200)
        self.assertEqual(result["spots"], 9)
        self.assertEqual(len(result["solutions"]), 9)
        self.assertEqual(result["solutions"][0]["key"]["hero_cards"], ["Kc", "Qd"])

    def test_tabular_multistreet_solver_export_endpoint(self) -> None:
        status, sample = self.request(
            "/api/sample-solver-export?format=tabular-csv-v1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(sample["format"], "tabular-csv-v1")
        self.assertIn("hu_ako_a72r94_river", sample["content"])
        status, imported = self.request(
            "/api/import-solver-export",
            {"format": sample["format"], "content": sample["content"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(imported["format"], "tabular-csv-v1")
        self.assertEqual(imported["spots"], 4)
        self.assertEqual(len(imported["solutions"]), 4)
        self.assertEqual(imported["solutions"][0]["key"]["board"], [])
        self.assertEqual(len(imported["solutions"][-1]["key"]["board"]), 5)
        self.assertEqual(imported["tree"]["linked_edges"], 3)
        self.assertEqual(imported["tree"]["max_depth"], 3)
        self.assertEqual(imported["range_strategies"]["group_count"], 4)
        river_fingerprint = imported["solutions"][-1]["fingerprint"]
        _, path = self.request(f"/api/solution-tree/{river_fingerprint}/path")
        self.assertEqual(len(path["path"]), 4)
        self.assertEqual(path["path"][-1]["node_id"], "hu_ako_a72r94_river")
        _, ranges = self.request("/api/range-strategies")
        self.assertGreaterEqual(ranges["group_count"], 4)

    def test_imported_node_solver_practice_round_trip(self) -> None:
        _, sample = self.request("/api/sample-solver-export?format=tabular-csv-v1")
        _, imported = self.request(
            "/api/import-solver-export",
            {"format": sample["format"], "content": sample["content"]},
        )
        public_fingerprint = imported["range_strategies"]["groups"][0][
            "public_fingerprint"
        ]
        status, challenge = self.request(
            "/api/solver-practice/sessions",
            {"public_fingerprint": public_fingerprint},
        )
        self.assertEqual(status, 201)
        self.assertTrue(challenge["strategy_hidden"])
        self.assertNotIn("frequency", repr(challenge))
        action_id = challenge["actions"][0]["action_id"]
        status, result = self.request(
            f"/api/solver-practice/sessions/{challenge['session_id']}/decisions",
            {"action_id": action_id},
        )
        self.assertEqual(status, 200)
        self.assertIn("ev_loss_bb", result)
        self.assertGreater(len(result["strategy"]), 1)

    def test_revisioned_live_table_state_and_decision_round_trip(self) -> None:
        _, sample_export = self.request(
            "/api/sample-solver-export?format=tabular-csv-v1"
        )
        self.request(
            "/api/import-solver-export",
            {
                "format": sample_export["format"],
                "content": sample_export["content"],
            },
        )
        _, state = self.request("/api/sample-live-state")
        status, session = self.request(
            "/api/live/sessions", {"table_id": state["table_id"]}
        )
        self.assertEqual(status, 201)
        status, result = self.request(
            f"/api/live/sessions/{session['session_id']}/states", state
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["match"]["confidence"], "exact")
        self.assertTrue(result["match"]["actions"])
        self.assertEqual(result["texture"]["street"], result["state"]["street"])
        self.assertIn("pairing", result["texture"])
        action_id = result["match"]["actions"][0]["action"]
        status, feedback = self.request(
            f"/api/live/sessions/{session['session_id']}/decisions",
            {"revision": state["revision"], "action_id": action_id},
        )
        self.assertEqual(status, 200)
        self.assertIn("ev_loss_bb", feedback)
        _, current = self.request(f"/api/live/sessions/{session['session_id']}")
        self.assertEqual(current["state_id"], result["state_id"])
        with self.assertRaises(HTTPError) as caught:
            self.request(
                f"/api/live/sessions/{session['session_id']}/states", state
            )
        self.assertEqual(caught.exception.code, 422)
        caught.exception.close()

    def test_live_hand_history_tail_poll_endpoint(self) -> None:
        _, sample_export = self.request(
            "/api/sample-solver-export?format=tabular-csv-v1"
        )
        self.request(
            "/api/import-solver-export",
            {"format": sample_export["format"], "content": sample_export["content"]},
        )
        source = """PokerStars Hand #92000000001: Hold'em No Limit (0.50/1 Play Money) - 2026/08/01 21:00:00 ET
Table 'Web Tail' 2-max Seat #1 is the button
Seat 1: Hero (100 in chips)
Seat 2: Villain (100 in chips)
Hero: posts small blind 0.50
Villain: posts big blind 1
*** HOLE CARDS ***
Dealt to Hero [As Kd]
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "HH-live.txt"
            path.write_text(source, encoding="utf-8")
            _, session = self.request(
                "/api/live/sessions", {"table_id": "Web Tail Session"}
            )
            endpoint = f"/api/live/sessions/{session['session_id']}/capture/polls"
            status, first = self.request(endpoint, {"path": str(path)})
            self.assertEqual(status, 200)
            self.assertEqual((first["status"], first["changed"]), ("state_ready", True))
            self.assertEqual(first["analysis"]["match"]["confidence"], "exact")
            _, unchanged = self.request(endpoint, {"path": str(path)})
            self.assertFalse(unchanged["changed"])
            self.assertEqual(
                unchanged["analysis"]["state_id"], first["analysis"]["state_id"]
            )
            path.write_text(
                source + "Hero: raises 2.50 to 3\n", encoding="utf-8"
            )
            _, waiting = self.request(endpoint, {"path": str(path)})
            self.assertEqual(waiting["status"], "waiting_for_player")
            self.assertEqual(waiting["next_actor"], "Villain")
            self.assertIsNone(waiting["analysis"])

    def test_two_frame_visual_observation_endpoint(self) -> None:
        _, sample_export = self.request(
            "/api/sample-solver-export?format=tabular-csv-v1"
        )
        self.request(
            "/api/import-solver-export",
            {"format": sample_export["format"], "content": sample_export["content"]},
        )
        _, first_frame = self.request("/api/sample-visual-observation")
        _, second_frame = self.request("/api/sample-visual-observation")
        self.assertNotEqual(
            first_frame["frame"]["frame_id"], second_frame["frame"]["frame_id"]
        )
        table_id = first_frame["fields"]["table_id"]["value"]
        _, session = self.request("/api/live/sessions", {"table_id": table_id})
        endpoint = (
            f"/api/live/sessions/{session['session_id']}/visual-observations"
        )
        status, pending = self.request(endpoint, first_frame)
        self.assertEqual(status, 200)
        self.assertEqual(pending["status"], "pending_stability")
        self.assertIsNone(pending["analysis"])
        status, ready = self.request(endpoint, second_frame)
        self.assertEqual(status, 200)
        self.assertEqual((ready["status"], ready["revision"]), ("state_ready", 0))
        self.assertEqual(ready["analysis"]["status"], "matched")
        self.assertEqual(ready["analysis"]["match"]["confidence"], "exact")
        low = copy.deepcopy(second_frame)
        low["frame"]["frame_id"] = "low-confidence-frame"
        low["fields"]["hero_cards"]["confidence"] = "0.40"
        _, blocked = self.request(endpoint, low)
        self.assertEqual(blocked["status"], "low_confidence")
        self.assertEqual(blocked["low_confidence_fields"], ["hero_cards"])

        invalid_first = copy.deepcopy(first_frame)
        invalid_second = copy.deepcopy(second_frame)
        for index, invalid in enumerate((invalid_first, invalid_second), start=1):
            invalid["frame"]["frame_id"] = f"invalid-transition-{index}"
            invalid["frame"]["image_sha256"] = str(index) * 64
            invalid["fields"]["pot_bb"]["value"] = "26"
        _, invalid_pending = self.request(endpoint, invalid_first)
        self.assertEqual(invalid_pending["status"], "pending_stability")
        _, rejected = self.request(endpoint, invalid_second)
        self.assertEqual(rejected["status"], "invalid_transition")
        self.assertIsNone(rejected["analysis"])
        self.assertIsNone(rejected["payload"])
        self.assertEqual(rejected["transition"]["status"], "rejected")
        self.assertIn(
            "pot_nondecreasing",
            {row["code"] for row in rejected["transition"]["violations"]},
        )
        _, current = self.request(f"/api/live/sessions/{session['session_id']}")
        self.assertEqual(current["last_revision"], 0)

    def test_direct_state_transition_error_returns_structured_422(self) -> None:
        _, sample = self.request("/api/sample-live-state")
        _, session = self.request(
            "/api/live/sessions", {"table_id": sample["table_id"]}
        )
        endpoint = f"/api/live/sessions/{session['session_id']}/states"
        self.request(endpoint, sample)
        invalid = copy.deepcopy(sample)
        invalid["revision"] = 1
        invalid["pot_bb"] = str(Decimal(sample["pot_bb"]) - Decimal("1"))
        with self.assertRaises(HTTPError) as raised:
            self.request(endpoint, invalid)
        self.assertEqual(raised.exception.code, 422)
        body = json.loads(raised.exception.read())
        self.assertEqual(body["transition"]["status"], "rejected")
        self.assertIn(
            "pot_nondecreasing",
            {row["code"] for row in body["transition"]["violations"]},
        )

    def test_imported_solution_generates_hand_review_drill(self) -> None:
        payload = json.loads(
            (ROOT / "examples" / "sample_solver_bundle.json").read_text(encoding="utf-8")
        )
        self.request("/api/import-solver-bundle", payload)
        source = (ROOT / "examples" / "sample_play_money_hand.txt").read_text(encoding="utf-8")
        _, report = self.request("/api/analyze-hand-history", {"hand_history": source})
        review = report["solution_review"]
        self.assertEqual(review["matched_decisions"], 1)
        self.assertEqual(review["drills"][0]["ev_loss_bb"], "0.15")
        self.assertEqual(report["study_queue"][0]["study"]["status"], "new")
        opponent = report["opponent_range_review"]
        self.assertEqual(opponent["public_state_matches"], 2)
        self.assertEqual(opponent["conditioned_actions"], 2)
        events = opponent["timelines"][0]["events"]
        self.assertEqual([event["street"] for event in events], ["flop", "turn"])
        self.assertEqual(events[1]["prior_transition"]["mode"], "posterior_carried")

    def test_study_rating_endpoint_updates_mastery(self) -> None:
        payload = json.loads(
            (ROOT / "examples" / "sample_solver_bundle.json").read_text(encoding="utf-8")
        )
        self.request("/api/import-solver-bundle", payload)
        source = (ROOT / "examples" / "sample_play_money_hand.txt").read_text(encoding="utf-8")
        _, report = self.request("/api/analyze-hand-history", {"hand_history": source})
        drill_id = report["study_queue"][0]["drill_id"]
        status, reviewed = self.request(
            f"/api/drills/{drill_id}/attempts", {"rating": "good"}
        )
        self.assertEqual(status, 200)
        self.assertGreater(float(reviewed["study"]["mastery"]), 0)
        _, queue = self.request("/api/drills")
        queued = next(
            drill for drill in queue["drills"] if drill["drill_id"] == drill_id
        )
        self.assertEqual(queued["study"]["last_rating"], "good")

    def test_invalid_action_is_a_structured_422(self) -> None:
        _, session = self.request("/api/sessions", {"count": 1, "seed": 1})
        with self.assertRaises(HTTPError) as caught:
            self.request(
                f"/api/sessions/{session['session_id']}/decisions",
                {"action_id": "not-legal"},
            )
        self.assertEqual(caught.exception.code, 422)
        caught.exception.close()


class PersistentHandLibraryWebTests(unittest.TestCase):
    def test_folder_scan_endpoint_persists_and_skips_unchanged_file(self) -> None:
        source = (ROOT / "examples" / "sample_play_money_hand.txt").read_text(
            encoding="utf-8"
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            histories = root / "histories"
            histories.mkdir()
            (histories / "HH.txt").write_text(source, encoding="utf-8")
            database = CoachDatabase(root / "coach.sqlite3")
            application = CoachApplication(
                solution_store=database,
                study_store=database,
                history_store=database,
            )
            server = create_server("127.0.0.1", 0, application)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def request(path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
                data = None if payload is None else json.dumps(payload).encode("utf-8")
                call = Request(
                    base + path,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="GET" if payload is None else "POST",
                )
                with urlopen(call, timeout=3) as response:
                    return json.loads(response.read())

            try:
                health = request("/api/health")
                self.assertTrue(health["persistent_hand_library"])
                first = request(
                    "/api/scan-hand-history-folder",
                    {"folder": str(histories), "recursive": False},
                )
                self.assertEqual((first["scan"]["inserted"], first["hands"]), (1, 1))
                second = request(
                    "/api/scan-hand-history-folder",
                    {"folder": str(histories), "recursive": False},
                )
                self.assertEqual(second["scan"]["skipped_files"], 1)
                self.assertTrue(second["scan"]["analysis_reused"])
                library = request("/api/hand-history-library")
                self.assertEqual(library["hands"], 1)
                pasted = request(
                    "/api/analyze-hand-history", {"hand_history": source}
                )
                self.assertEqual(pasted["import"]["unchanged"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                database.close()


if __name__ == "__main__":
    unittest.main()
