"""Compare frozen APC action-value models on the same held-out rollout states."""
import argparse
import json
from pathlib import Path

import torch

from apc.neural.model import load_apc_weights
from apc.neural.train_candidate import load_raised_corpus, _predict, _sliced_metrics


def compare(dataset, incumbent, candidate):
    manifest, corpus = load_raised_corpus(dataset)
    indices = corpus.indices("test")
    report = {"dataset_fingerprint": manifest["dataset_fingerprint"], "split": "test",
              "evaluation_target": "paired virtual-chip rollout returns"}
    for name, path in (("incumbent", incumbent), ("candidate", candidate)):
        record = json.loads(path.read_text(encoding="utf-8"))
        weights = record["weights"]
        model = load_apc_weights(path.parent / weights["file"], weights["weights_sha256"])
        values, _, uncertainty = _predict(model, corpus, indices, 128)
        report[name] = _sliced_metrics(corpus, indices, values, uncertainty)
        report[name]["weights_sha256"] = weights["weights_sha256"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("dataset", "incumbent", "candidate", "output"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = compare(args.dataset, args.incumbent, args.candidate)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
    for name in ("incumbent", "candidate"):
        print(name, {key: report[name][key] for key in ("decision_accuracy", "chosen_action_regret_bb", "mae_bb")})
