"""Revision-safe adapter for an internal virtual-chip graphical practice table."""
import threading
import time
import copy

from apc.deadline import ActionCommand
from apc.full_hand_table import HeadsUpVirtualHand


class PracticeTableSession:
    def __init__(self, seed):
        self.hand = HeadsUpVirtualHand(seed=seed, button_player=0)
        self.events = []
        self.feedback = None
        self.lock = threading.RLock()

    def snapshot(self):
        with self.lock:
            return {"state": copy.deepcopy(self.hand.observation()), "feedback": copy.deepcopy(self.feedback),
                    "opponent_policy": "check_call_practice_not_GTO",
                    "external_actuation": False}

    def act(self, action, expected_fingerprint, to_amount_bb=None):
        with self.lock:
            state = self.hand.observation()
            if expected_fingerprint != state["state_fingerprint"]:
                raise ValueError("Stale table state; refresh before acting")
            if state["next_actor"] != "Hero" or state["terminal"]:
                raise ValueError("No Hero action is available")
            command = ActionCommand(action, to_amount_bb=to_amount_bb)
            raw_ms = time.monotonic_ns() // 1_000_000
            observed = max(raw_ms, self.events[-1]["observed_monotonic_ms"] + 1) if self.events else raw_ms
            result = self.hand.step(command)
            self.events.append({"observed_monotonic_ms": observed,
                                "timestamp_resolution_adjusted": observed != raw_ms,
                                "state_fingerprint": state["state_fingerprint"],
                                "legal_action_keys": [row["action"] for row in state["action_buttons"]],
                                "chosen_action_key": action, "chosen_action": result["command"],
                                "canonical_state": state})
            while not self.hand.terminal and self.hand.observation()["next_actor"] != "Hero":
                legal = {row["action"] for row in self.hand.legal_action_buttons()}
                result = self.hand.step(ActionCommand("check" if "check" in legal else "call"))
            if self.hand.terminal:
                self.feedback = result["completed_hand_feedback"]
            return self.snapshot()

    def completed_replay(self, session_id):
        from apc.neural.self_play_replay import completed_replay
        with self.lock:
            if self.feedback is None:
                raise ValueError("An unfinished practice hand cannot become a training record")
            return completed_replay(self.hand, copy.deepcopy(self.events), copy.deepcopy(self.feedback), session_id)
