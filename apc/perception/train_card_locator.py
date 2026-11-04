"""Reproducible CPU neural localization experiment, evaluated without oracle boxes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from apc.perception.card_locator import CardLocator, locator_targets, locator_loss, decode_locator
from apc.perception.card_proposals import match_proposals
from apc.perception.spatial_targets import load_split


def dataset(manifest, split):
    images, labels, truths = [], [], []
    for row in load_split(manifest, split):
        cards = [obj for obj in row["objects"] if obj["class_name"] in ("hero_card", "board_card")]
        if any(not obj["box_mask"] for obj in cards):
            raise ValueError("Localization experiment requires fully boxed cards")
        with Image.open(row["image"]["path"]) as source:
            image = np.asarray(source.convert("RGB").resize((320, 184)), dtype=np.float32).copy() / 255
        images.append(torch.from_numpy(image).permute(2, 0, 1))
        labels.append(locator_targets([obj["box_xyxy"] for obj in cards], 23, 40))
        truths.append(cards)
    if not images:
        raise ValueError("Empty locator split")
    return torch.stack(images), torch.stack(labels), truths


def evaluate(model, data):
    totals = {"frames": len(data[0]), "targets": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data[0]), 8):
            outputs = model(data[0][start:start+8])
            for offset, output in enumerate(outputs):
                truth = data[2][start+offset]
                matches = match_proposals(decode_locator(output), truth)
                totals["targets"] += len(truth)
                for key in ("true_positives", "false_positives", "false_negatives"):
                    totals[key] += matches[key]
    tp, fp, fn = (totals[key] for key in ("true_positives", "false_positives", "false_negatives"))
    totals.update(precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return totals


def train(manifest, destination, epochs=10, seed=42, *, relative_size=False):
    destination = Path(destination)
    if destination.exists() or epochs < 1:
        raise ValueError("Use a new destination and positive epochs")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    datasets = {split: dataset(manifest, split) for split in ("train", "validation", "test")}
    model = CardLocator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    best, saved, history = -1, None, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(datasets["train"][0]))
        for indices in order.split(8):
            optimizer.zero_grad()
            loss = locator_loss(model(datasets["train"][0][indices]), datasets["train"][1][indices], relative_size=relative_size)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite locator loss")
            loss.backward()
            optimizer.step()
        validation = evaluate(model, datasets["validation"])
        if validation["f1"] > best:
            best = validation["f1"]
            saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
        history.append({"epoch": epoch+1, "validation": validation})
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(saved)
    report = {"model_kind": "full_frame_neural_card_locator", "architecture_version": "card-locator-v1",
              "seed": seed, "epochs": epochs, "input_size": [320,184], "threshold": .5,
              "geometry_loss": "log_size_ratio" if relative_size else "normalized_smooth_l1",
              "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "history": history, "test": evaluate(model, datasets["test"]),
              "torch_version": torch.__version__, "confidence_calibrated": False,
              "full_table_readiness": False, "limitation": "Synthetic-only diagnostic; layout memorization not excluded"}
    destination.mkdir(parents=True)
    torch.save({"state_dict": saved, "model_kind": report["model_kind"],
                "architecture_version": report["architecture_version"], "input_size": report["input_size"]},
               destination / "weights.pt")
    (destination / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--relative-size", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, args.epochs, relative_size=args.relative_size)["test"], indent=2))
