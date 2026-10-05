"""Reproducible CPU region-recognition training on session-isolated manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from apc.perception.region_model import RegionRecognizer, recognition_loss, region_supervision
from apc.perception.spatial_targets import load_split


def image_tensor(target, size=512):
    # RGB conversion and normalized boxes share the same full-frame coordinate
    # system. No crop/letterbox transform silently changes the target geometry.
    with Image.open(target["image"]["path"]) as image:
        rgb = image.convert("RGB")
        scale = size / max(rgb.size)
        rgb = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))))
        array = np.asarray(rgb, dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def evaluate(model, targets):
    counts, correct, elapsed = {}, {}, []
    model.eval()
    with torch.no_grad():
        for target in targets:
            image = image_tensor(target)
            regions, labels = region_supervision([target])
            started = time.perf_counter()
            predictions = model(image, regions)
            elapsed.append((time.perf_counter() - started) * 1000)
            for name, logits in predictions.items():
                mask = labels[name] != -100
                counts[name] = counts.get(name, 0) + int(mask.sum())
                correct[name] = correct.get(name, 0) + int((logits.argmax(-1)[mask] == labels[name][mask]).sum())
    return {"frames": len(targets), "heads": {
        name: {"supervised_regions": count, "correct": correct[name],
               "accuracy": correct[name] / count if count else None}
        for name, count in counts.items()},
        "region_network_p95_ms": float(np.percentile(elapsed, 95)) if elapsed else None,
        "latency_scope": "network_only_with_ground_truth_regions_not_end_to_end",
        "full_table_readiness": False}


def train(manifest, output, *, epochs=5, seed=42, crop_size=48):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output = Path(output)
    if output.exists():
        raise ValueError("Output directory must be new; existing runs are never overwritten")
    splits = {name: load_split(manifest, name) for name in ("train", "validation", "test")}
    if any(not value for value in splits.values()):
        raise ValueError("Training requires nonempty train, validation, and test sessions")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = RegionRecognizer(crop_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    best_accuracy = -1
    best_weights = None
    for epoch in range(epochs):
        model.train()
        order = list(splits["train"])
        random.shuffle(order)
        total = 0
        for target in order:
            regions, labels = region_supervision([target])
            optimizer.zero_grad()
            loss, _ = recognition_loss(model(image_tensor(target), regions), labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        validation = evaluate(model, splits["validation"])
        # Select using validation only; test is evaluated once after selection.
        heads = validation["heads"]
        accuracy = sum(v["correct"] for v in heads.values()) / sum(v["supervised_regions"] for v in heads.values())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_loss": total / len(order), "validation": validation})
    model.load_state_dict(best_weights)
    report = {"schema_version": "1.0.0", "model_kind": "region_recognizer_requires_proposed_boxes",
              "seed": seed, "epochs": epochs, "crop_size": crop_size,
              "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "torch_version": torch.__version__, "history": history,
              "test": evaluate(model, splits["test"]), "full_table_readiness": False}
    output.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "crop_size": crop_size,
                "model_kind": report["model_kind"]}, output / "region_weights.pt")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train(args.manifest, args.output, epochs=args.epochs, seed=args.seed)
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
