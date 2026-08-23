from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from apc.deadline import ActionCommand
from apc.full_hand_table import HeadsUpVirtualHand
from apc.neural.replay_adapter import load_replay_temporal_corpus
from apc.neural.replay_buffer import APCReplayBuffer


def _engine_fingerprint() -> str:
    source = Path(__file__).resolve().parents[1] / "full_hand_table.py"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _scripted_action(hand: HeadsUpVirtualHand) -> ActionCommand:
    buttons = {row["action"]: row for row in hand.legal_action_buttons()}
    token = hand.seed * 17 + hand.revision * 31 + (hand.next_actor or 0) * 7
    if "check" in buttons:
        if "bet" in buttons and token % 9 == 0:
            return ActionCommand("bet", to_amount_bb=buttons["bet"]["minimum_to_bb"])
        return ActionCommand("check")
    if "fold" in buttons and token % 17 == 0:
        return ActionCommand("fold")
    if "raise" in buttons and token % 13 == 0:
        return ActionCommand("raise", to_amount_bb=buttons["raise"]["minimum_to_bb"])
    return ActionCommand("call", amount_bb=buttons["call"]["amount_bb"])


def _private_safe_state(observation: dict[str, object]) -> dict[str, object]:
    state = json.loads(json.dumps(observation, separators=(",", ":")))
    state["opponent_cards"] = None
    return state


def generate_virtual_completed_replay(seed: int, *, hands_per_session: int = 3) -> dict[str, object]:
    if seed < 0 or hands_per_session <= 0:
        raise ValueError("APC self-play replay parameters are invalid")
    hand = HeadsUpVirtualHand(seed=seed, button_player=seed % 2)
    events = []
    final = None
    while not hand.terminal:
        observation = hand.observation()
        command = _scripted_action(hand)
        if observation["next_actor"] == "Hero":
            events.append({
                "observed_monotonic_ms": 1000 + hand.revision * 100,
                "state_fingerprint": observation["state_fingerprint"],
                "legal_action_keys": [str(row["action"]) for row in observation["action_buttons"]],
                "chosen_action_key": command.action,
                "chosen_action": command.payload(),
                "canonical_state": _private_safe_state(observation),
            })
        final = hand.step(command)
    if not events or final is None:
        raise RuntimeError("APC virtual hand produced no Hero decisions")
    feedback = final["completed_hand_feedback"]
    session = f"self-play-session-{seed // hands_per_session:08d}"
    return {
        "schema_version": "1.0.0",
        "model_name": "APC",
        "units": "BB",
        "source_environment": "controlled_virtual_chips",
        "session_id": session,
        "hand_id": hand.hand_id,
        "split_group_id": session,
        "source_fingerprint": _engine_fingerprint(),
        "full_hand_completed": True,
        "events": events,
        "completed_hand_feedback": {
            "full_hand_completed": True,
            "hero_reward_bb": feedback["rewards_bb"]["Hero"],
        },
        "external_actuation": False,
    }


def build_virtual_replay_buffer(
    destination: str | Path,
    *,
    hands: int,
    seed_start: int,
    hands_per_session: int = 3,
) -> dict[str, object]:
    if hands < 12:
        raise ValueError("APC continual replay build requires at least 12 completed hands")
    buffer = APCReplayBuffer(destination)
    added = 0
    for seed in range(seed_start, seed_start + hands):
        added += bool(buffer.ingest(generate_virtual_completed_replay(seed, hands_per_session=hands_per_session))["added"])
    validation = buffer.validate()
    if not validation["valid"]:
        raise RuntimeError("APC generated replay buffer failed validation")
    corpus = load_replay_temporal_corpus(buffer)
    split_hands = {
        split: len({corpus.replay_fingerprints[int(index)] for index in corpus.indices(split)})
        for split in ("train", "validation", "test")
    }
    passed = all(split_hands.values())
    return {
        "schema_version": "1.0.0",
        "model_name": "APC",
        "status": "training_eligible" if passed else "insufficient_group_split_coverage",
        "hands_requested": hands,
        "hands_added": added,
        "seed_start": seed_start,
        "seed_end": seed_start + hands - 1,
        "hands_per_session": hands_per_session,
        "buffer_validation": validation,
        "adapter_fingerprint": corpus.manifest["adapter_fingerprint"],
        "decisions": corpus.manifest["decisions"],
        "split_complete_hands": split_hands,
        "training_eligible": passed,
        "external_actuation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate APC completed-hand virtual-chip replay")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--hands", type=int, default=90)
    parser.add_argument("--seed-start", type=int, default=80000)
    parser.add_argument("--hands-per-session", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build_virtual_replay_buffer(args.destination, hands=args.hands, seed_start=args.seed_start, hands_per_session=args.hands_per_session), indent=2))


if __name__ == "__main__":
    main()
