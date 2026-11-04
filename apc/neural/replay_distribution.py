"""Inspect completed-replay target distributions without fitting weights."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from apc.neural.replay_buffer import APCReplayBuffer


def summarize(rows):
    rewards = np.asarray([float(row["reward_bb"]) for row in rows])
    counts = np.asarray([int(row["event_count"]) for row in rows])
    if not len(rows):
        return {"hands": 0}
    return {"hands": len(rows), "decisions": int(counts.sum()),
            "mean_reward_bb_hand_weighted": float(rewards.mean()),
            "mean_reward_bb_decision_weighted": float(np.average(rewards, weights=counts)),
            "mean_abs_reward_bb": float(np.abs(rewards).mean()),
            "reward_p05_bb": float(np.percentile(rewards, 5)),
            "reward_p95_bb": float(np.percentile(rewards, 95))}


def audit(root):
    buffer = APCReplayBuffer(root)
    validation = buffer.validate()
    if not validation["valid"]:
        raise ValueError("Replay integrity validation failed")
    groups = defaultdict(list)
    for entry in buffer._load()["entries"]:
        replay = json.loads((buffer.root / entry["file"]).read_text(encoding="utf-8"))
        config = replay.get("self_play_configuration", {})
        key = (entry["split"], config.get("opponent_policy", "unknown"),
               config.get("starting_stack_bb", "unknown"))
        groups[key].append(entry)
    return {"diagnostic_only": True, "model_updated": False,
            "groups": [{"split": key[0], "opponent_policy": key[1], "starting_stack_bb": key[2],
                        **summarize(rows)} for key, rows in sorted(groups.items())]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_buffer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Use a new output file")
    payload = json.dumps(audit(args.replay_buffer), indent=2)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(payload + "\n")
    print(payload)
