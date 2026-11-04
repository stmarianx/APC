"""Compare frozen APC action-value models on the same held-out rollout states."""
import argparse
import json
from pathlib import Path

import torch

from apc.neural.model import load_apc_weights
from apc.neural.train_candidate import load_raised_corpus, _predict, _sliced_metrics


def matched_conditions(left, right):
    """Report control mismatches; do not infer causality from unrelated runs."""
    mismatches = []
    for section, keys in (
        ("training", ("seed", "epochs", "batch_size", "learning_rate")),
        ("dataset", ("dataset_fingerprint",)),
        ("framework", ("name", "version", "device")),
    ):
        for key in keys:
            if key not in left.get(section, {}) or key not in right.get(section, {}) or left[section][key] != right[section][key]:
                mismatches.append(section + "." + key)
    if not left.get("architecture") or left.get("architecture") != right.get("architecture"):
        mismatches.append("architecture")
    required_training = {"seed", "epochs", "batch_size", "learning_rate"}
    for key in (set(left.get("training", {})) | set(right.get("training", {}))) - required_training - {"paired_expected_regret_weight"}:
        if left.get("training", {}).get(key) != right.get("training", {}).get(key):
            mismatches.append("training." + key)
    return {"matched": not mismatches, "mismatches": mismatches,
            "limitation": "Matching conditions alone does not establish a significant improvement"}


def compare(dataset, incumbent, candidate):
    manifest, corpus = load_raised_corpus(dataset)
    indices = corpus.indices("test")
    report = {"dataset_fingerprint": manifest["dataset_fingerprint"], "split": "test",
              "evaluation_target": "paired virtual-chip rollout returns"}
    records = {}
    for name, path in (("incumbent", incumbent), ("candidate", candidate)):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[name] = record
        weights = record["weights"]
        model = load_apc_weights(path.parent / weights["file"], weights["weights_sha256"])
        values, _, uncertainty = _predict(model, corpus, indices, 128)
        report[name] = _sliced_metrics(corpus, indices, values, uncertainty)
        report[name]["weights_sha256"] = weights["weights_sha256"]
    report["training_conditions"] = matched_conditions(records["incumbent"], records["candidate"])
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
