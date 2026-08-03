from pathlib import Path
import unittest

from poker_coach import LiveTableService, SolverExportRegistry, StateTransitionError


ROOT = Path(__file__).resolve().parents[1]


class LiveTableServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spots = SolverExportRegistry().parse_file(
            ROOT / "examples" / "sample_solver_export.csv"
        ).bundle.spots

    def setUp(self) -> None:
        self.service = LiveTableService()
        self.session = self.service.create_session("Play Table 1")

    @staticmethod
    def payload(spot, *, revision=0, hand_id="live-1", **changes):
        key = spot.key
        row = {
            "schema_version": "1.0.0",
            "table_id": "Play Table 1",
            "hand_id": hand_id,
            "revision": revision,
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
            "source": "test_capture_adapter",
        }
        row.update(changes)
        return row

    def update(self, payload):
        return self.service.update_state(
            self.session["session_id"], payload, self.spots
        )

    def test_exact_state_returns_provenance_backed_strategy_and_math(self) -> None:
        spot = self.spots[-1]
        result = self.update(self.payload(spot, to_call_bb="6.75"))
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["match"]["confidence"], "exact")
        self.assertEqual(result["match"]["node_id"], spot.node_id)
        self.assertEqual(len(result["match"]["actions"]), len(spot.actions))
        self.assertEqual(result["math"]["call_break_even_equity"], "0.2")
        self.assertEqual(result["state"]["source"], "test_capture_adapter")
        self.assertEqual(result["texture"]["street"], "river")
        self.assertIn("made_hand", result["texture"]["hero"])
        self.assertIn("range advantage", result["texture"]["range_caveat"])
        self.assertEqual(result["transition"]["status"], "accepted")
        self.assertEqual(result["transition"]["kind"], "initial_state")
        self.assertEqual(
            result["strategy_route"]["selection_status"], "blueprint_fallback"
        )
        self.assertEqual(
            result["strategy_route"]["fallback"]["reason"],
            "refiner_not_configured",
        )
        self.assertEqual(result["strategy_route"]["latency"]["budget_ms"], 75)

    def test_suit_renamed_state_matches_without_changing_structure(self) -> None:
        spot = self.spots[1]
        swap = {"c": "d", "d": "c", "h": "s", "s": "h"}
        renamed = lambda token: token[:-1] + swap[token[-1]]
        payload = self.payload(spot)
        payload["board"] = [renamed(token) for token in payload["board"]]
        payload["hero_cards"] = [renamed(token) for token in payload["hero_cards"]]
        result = self.update(payload)
        self.assertEqual(result["match"]["card_match"], "suit_isomorphic")

    def test_revisions_and_same_hand_progression_cannot_roll_back(self) -> None:
        self.update(self.payload(self.spots[1], revision=2))
        with self.assertRaisesRegex(ValueError, "advance beyond 2"):
            self.update(self.payload(self.spots[1], revision=2))
        with self.assertRaisesRegex(ValueError, "board cannot roll back"):
            self.update(self.payload(self.spots[0], revision=3))
        new_hand = self.update(
            self.payload(self.spots[0], revision=3, hand_id="live-2")
        )
        self.assertEqual(new_hand["state"]["street"], "preflop")

    def test_legal_action_filter_and_decision_audit(self) -> None:
        spot = self.spots[-1]
        legal = [spot.actions[0].action, spot.actions[-1].action]
        result = self.update(self.payload(spot, revision=7, legal_actions=legal))
        self.assertEqual(result["match"]["coverage"]["legal_solved_actions"], 2)
        self.assertEqual(len(result["match"]["coverage"]["omitted_actions"]), 1)
        feedback = self.service.record_decision(
            self.session["session_id"], 7, legal[0]
        )
        self.assertEqual(feedback["revision"], 7)
        self.assertIn("ev_loss_bb", feedback)
        with self.assertRaisesRegex(ValueError, "already recorded"):
            self.service.record_decision(self.session["session_id"], 7, legal[0])

    def test_stale_or_uncovered_decision_is_rejected(self) -> None:
        spot = self.spots[-1]
        self.update(self.payload(spot, revision=4, legal_actions=[]))
        with self.assertRaisesRegex(ValueError, "revision 3 is stale"):
            self.service.record_decision(
                self.session["session_id"], 3, spot.actions[0].action
            )
        with self.assertRaisesRegex(ValueError, "not covered"):
            self.service.record_decision(
                self.session["session_id"], 4, "raise_to:99"
            )

    def test_unmatched_state_is_explicit_and_cannot_be_scored(self) -> None:
        payload = self.payload(self.spots[0], pot_bb="99")
        result = self.update(payload)
        self.assertEqual(result["status"], "unmatched")
        self.assertIsNone(result["match"])
        self.assertTrue(result["warnings"])
        with self.assertRaisesRegex(ValueError, "no matched solver solution"):
            self.service.record_decision(
                self.session["session_id"], 0, "fold"
            )

    def test_malformed_state_fails_before_mutating_session(self) -> None:
        payload = self.payload(self.spots[0], hero_cards=["As", "As"])
        with self.assertRaisesRegex(ValueError, "conflicting cards"):
            self.update(payload)
        current = self.service.current(self.session["session_id"])
        self.assertEqual(current["status"], "awaiting_state")

    def test_valid_same_hand_progression_has_auditable_deltas(self) -> None:
        spot = self.spots[0]
        first = self.payload(spot)
        self.update(first)
        history = [*first["action_history"], "BB check"]
        progressed = self.update(
            self.payload(
                spot,
                revision=1,
                pot_bb="3.5",
                effective_stack_bb="99",
                action_history=history,
            )
        )
        audit = progressed["transition"]
        self.assertEqual((audit["status"], audit["kind"]), ("accepted", "same_hand_progression"))
        self.assertEqual(audit["deltas"]["actions_added"], 1)
        self.assertEqual(audit["deltas"]["pot_change_bb"], "2.0")

    def test_normalized_all_in_history_is_accepted(self) -> None:
        spot = self.spots[0]
        payload = self.payload(
            spot,
            action_history=[*spot.key.action_history, "BTN all_in:99"],
        )
        result = self.update(payload)
        self.assertEqual(result["transition"]["status"], "accepted")
        self.assertEqual(result["transition"]["kind"], "initial_state")

    def test_chip_and_hidden_card_discontinuities_are_rejected_without_mutation(self) -> None:
        spot = self.spots[0]
        initial = self.update(self.payload(spot))
        invalid_cases = (
            ({"pot_bb": "1"}, "pot_nondecreasing"),
            ({"effective_stack_bb": "101"}, "heads_up_effective_stack_nonincreasing"),
            ({"hero_cards": ["Ad", "Kd"]}, "hero_cards_immutable"),
            ({"rake_model": "different_rake"}, "rake_model_immutable"),
        )
        for changes, expected_code in invalid_cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(StateTransitionError) as raised:
                    self.update(self.payload(spot, revision=1, **changes))
                codes = {row["code"] for row in raised.exception.audit["violations"]}
                self.assertIn(expected_code, codes)
                self.assertEqual(raised.exception.audit["status"], "rejected")
                self.assertEqual(
                    self.service.current(self.session["session_id"])["state_id"],
                    initial["state_id"],
                )

    def test_unchanged_public_state_cannot_change_prices(self) -> None:
        spot = self.spots[0]
        self.update(self.payload(spot))
        with self.assertRaises(StateTransitionError) as raised:
            self.update(self.payload(spot, revision=1, to_call_bb="1"))
        self.assertIn(
            "unchanged_state_call_price",
            {row["code"] for row in raised.exception.audit["violations"]},
        )

    def test_action_tokens_are_normalized_and_revision_gaps_are_visible(self) -> None:
        spot = self.spots[0]
        with self.assertRaises(StateTransitionError) as raised:
            self.update(self.payload(spot, action_history=["BTN raises a little"]))
        self.assertEqual(
            raised.exception.audit["violations"][0]["code"],
            "normalized_action_tokens",
        )
        self.update(self.payload(spot))
        result = self.update(self.payload(spot, revision=3))
        self.assertEqual(result["transition"]["warnings"][0]["code"], "revision_gap")


if __name__ == "__main__":
    unittest.main()
