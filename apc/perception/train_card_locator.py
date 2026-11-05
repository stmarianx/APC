"""Reproducible CPU neural localization experiment, evaluated without oracle boxes."""
import argparse
import hashlib
import json
import math
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


def evaluate(model, data, *, box_nms=False, iou_threshold=.5):
    totals = {"frames": len(data[0]), "targets": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data[0]), 8):
            outputs = model(data[0][start:start+8])
            for offset, output in enumerate(outputs):
                truth = data[2][start+offset]
                matches = match_proposals(decode_locator(output, box_nms=True, iou_threshold=iou_threshold) if box_nms else decode_locator(output), truth)
                totals["targets"] += len(truth)
                for key in ("true_positives", "false_positives", "false_negatives"):
                    totals[key] += matches[key]
    tp, fp, fn = (totals[key] for key in ("true_positives", "false_positives", "false_negatives"))
    totals.update(precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return totals


def augment_spatial(data, *, scale_min=.75, scale_max=1.2, independent_axes=False):
    """Move/zoom complete frames and labels together, retaining all card faces."""
    if not all(math.isfinite(v) for v in (scale_min, scale_max)) or not 0 < scale_min <= scale_max:
        raise ValueError("Expected finite positive ordered scale bounds")
    images, labels, truths = [], [], []
    for image, cards in zip(data[0], data[2]):
        boxes = [card["box_xyxy"] for card in cards]
        if boxes:
            min_x, min_y = min(b[0] for b in boxes), min(b[1] for b in boxes)
            max_x, max_y = max(b[2] for b in boxes), max(b[3] for b in boxes)
            maximum_x, maximum_y = min(scale_max, 1/(max_x-min_x)), min(scale_max, 1/(max_y-min_y))
            if min(maximum_x, maximum_y) < scale_min:
                raise ValueError("Minimum scale cannot preserve all visible cards")
            if independent_axes:
                sx = scale_min + float(torch.rand(())) * (maximum_x-scale_min)
                sy = scale_min + float(torch.rand(())) * (maximum_y-scale_min)
            else:
                sx = sy = scale_min + float(torch.rand(())) * (min(maximum_x, maximum_y)-scale_min)
            dx = -sx*min_x + float(torch.rand(())) * (1-sx*(max_x-min_x))
            dy = -sy*min_y + float(torch.rand(())) * (1-sy*(max_y-min_y))
        else:
            sx, sy, dx, dy = 1.0, 1.0, 0.0, 0.0
        transformed = []
        for card in cards:
            bounds = [min(1.0, max(0.0, (sx*v+dx) if index % 2 == 0 else (sy*v+dy))) for index, v in enumerate(card["box_xyxy"])]
            transformed.append(dict(card, box_xyxy=bounds))
        theta = image.new_tensor([[1/sx, 0, (1-sx-2*dx)/sx],
                                  [0, 1/sy, (1-sy-2*dy)/sy]])[None]
        grid = F.affine_grid(theta, image[None].shape, align_corners=False)
        images.append(F.grid_sample(image[None], grid, align_corners=False, padding_mode="zeros")[0])
        labels.append(locator_targets([c["box_xyxy"] for c in transformed], data[1].shape[-2], data[1].shape[-1]))
        truths.append(transformed)
    return torch.stack(images), torch.stack(labels), truths


def frozen_spatial(data, seed, **options):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return augment_spatial(data, **options)


def train(manifest, destination, epochs=10, seed=42, *, relative_size=False, spatial_augmentation=False, hard_negative_ratio=None, scale_min=.75, scale_max=1.2, independent_axes=False):
    destination = Path(destination)
    if destination.exists() or epochs < 1:
        raise ValueError("Use a new destination and positive epochs")
    if hard_negative_ratio is not None and (type(hard_negative_ratio) is not int or hard_negative_ratio < 1):
        raise ValueError("Hard-negative ratio must be a positive integer or None")
    if not all(math.isfinite(v) for v in (scale_min, scale_max)) or not 0 < scale_min <= scale_max:
        raise ValueError("Expected finite positive ordered scale bounds")
    if not spatial_augmentation and (independent_axes or scale_min != .75 or scale_max != 1.2):
        raise ValueError("Scale options require spatial augmentation")
    spatial_options = dict(scale_min=scale_min, scale_max=scale_max, independent_axes=independent_axes)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    datasets = {split: dataset(manifest, split) for split in ("train", "validation", "test")}
    if spatial_augmentation:
        varied = frozen_spatial(datasets["validation"], seed+1, **spatial_options)
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
                inputs, targets, _ = augment_spatial((inputs, targets, [datasets["train"][2][i] for i in indices.tolist()]), **spatial_options)
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
              "spatial_options": spatial_options if spatial_augmentation else None,
              "hard_negative_ratio": hard_negative_ratio,
              "hard_negative_minimum_cells_per_frame": 32 if hard_negative_ratio is not None else None,
              "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "history": history, "test": evaluate(model, datasets["test"]),
              "torch_version": torch.__version__, "confidence_calibrated": False,
              "full_table_readiness": False, "limitation": "Synthetic-only diagnostic; layout memorization not excluded"}
    if spatial_augmentation:
        report["test_frozen_spatial"] = evaluate(model, frozen_spatial(datasets["test"], seed+2, **spatial_options))
        if independent_axes or scale_min != .75 or scale_max != 1.2:
            report["test_legacy_spatial"] = evaluate(model, frozen_spatial(datasets["test"], seed+2))
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
    parser.add_argument("--scale-min", type=float, default=.75)
    parser.add_argument("--scale-max", type=float, default=1.2)
    parser.add_argument("--independent-axes", action="store_true", help="Vary horizontal and vertical scale independently")
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, args.epochs, relative_size=args.relative_size,
        spatial_augmentation=args.spatial_augmentation, hard_negative_ratio=args.hard_negative_ratio,
        scale_min=args.scale_min, scale_max=args.scale_max, independent_axes=args.independent_axes)["test"], indent=2))
