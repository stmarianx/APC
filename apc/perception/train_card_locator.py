"""Reproducible CPU neural localization experiment, evaluated without oracle boxes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
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


def augment_spatial(data):
    """Move/zoom complete frames and labels together, retaining all card faces."""
    images, labels, truths = [], [], []
    for image, cards in zip(data[0], data[2]):
        boxes = [card["box_xyxy"] for card in cards]
        if boxes:
            min_x, min_y = min(b[0] for b in boxes), min(b[1] for b in boxes)
            max_x, max_y = max(b[2] for b in boxes), max(b[3] for b in boxes)
            maximum = min(1.2, 1 / (max_x-min_x), 1 / (max_y-min_y))
            scale = .75 + float(torch.rand(())) * (maximum-.75)
            dx = -scale*min_x + float(torch.rand(())) * (1-scale*(max_x-min_x))
            dy = -scale*min_y + float(torch.rand(())) * (1-scale*(max_y-min_y))
        else:
            scale, dx, dy = 1.0, 0.0, 0.0
        transformed = []
        for card in cards:
            bounds = [min(1.0, max(0.0, scale*v + (dx if index % 2 == 0 else dy))) for index, v in enumerate(card["box_xyxy"])]
            transformed.append(dict(card, box_xyxy=bounds))
        theta = image.new_tensor([[1/scale, 0, (1-scale-2*dx)/scale],
                                  [0, 1/scale, (1-scale-2*dy)/scale]])[None]
        grid = F.affine_grid(theta, image[None].shape, align_corners=False)
        images.append(F.grid_sample(image[None], grid, align_corners=False, padding_mode="zeros")[0])
        labels.append(locator_targets([c["box_xyxy"] for c in transformed], data[1].shape[-2], data[1].shape[-1]))
        truths.append(transformed)
    return torch.stack(images), torch.stack(labels), truths


def frozen_spatial(data, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return augment_spatial(data)


def train(manifest, destination, epochs=10, seed=42, *, relative_size=False, spatial_augmentation=False, hard_negative_ratio=None):
    destination = Path(destination)
    if destination.exists() or epochs < 1:
        raise ValueError("Use a new destination and positive epochs")
    if hard_negative_ratio is not None and (type(hard_negative_ratio) is not int or hard_negative_ratio < 1):
        raise ValueError("Hard-negative ratio must be a positive integer or None")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    datasets = {split: dataset(manifest, split) for split in ("train", "validation", "test")}
    if spatial_augmentation:
        varied = frozen_spatial(datasets["validation"], seed+1)
        original = datasets["validation"]
        datasets["validation"] = (torch.cat([original[0], varied[0]]), torch.cat([original[1], varied[1]]), original[2]+varied[2])
    model = CardLocator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    best, saved, history = -1, None, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(datasets["train"][0]))
        for indices in order.split(8):
            optimizer.zero_grad()
            inputs, targets = datasets["train"][0][indices], datasets["train"][1][indices]
            if spatial_augmentation:
                inputs, targets, _ = augment_spatial((inputs, targets, [datasets["train"][2][i] for i in indices.tolist()]))
            loss = locator_loss(model(inputs), targets, relative_size=relative_size, hard_negative_ratio=hard_negative_ratio)
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
              "spatial_augmentation": "bounded_scale_translation_v1" if spatial_augmentation else None,
              "hard_negative_ratio": hard_negative_ratio,
              "hard_negative_minimum_cells_per_frame": 32 if hard_negative_ratio is not None else None,
              "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "history": history, "test": evaluate(model, datasets["test"]),
              "torch_version": torch.__version__, "confidence_calibrated": False,
              "full_table_readiness": False, "limitation": "Synthetic-only diagnostic; layout memorization not excluded"}
    if spatial_augmentation:
        report["test_frozen_spatial"] = evaluate(model, frozen_spatial(datasets["test"], seed+2))
        report["validation_scope"] = "Original validation frames plus fixed spatial variants; test does not select weights"
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
    parser.add_argument("--spatial-augmentation", action="store_true")
    parser.add_argument("--hard-negative-ratio", type=int, help="Mine strongest negative cells per frame, with at least 32 negatives")
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, args.epochs, relative_size=args.relative_size,
        spatial_augmentation=args.spatial_augmentation, hard_negative_ratio=args.hard_negative_ratio)["test"], indent=2))
