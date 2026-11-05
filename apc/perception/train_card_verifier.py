"""Reproducible synthetic-only binary card-face training with session splits."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from apc.perception.card_proposals import box_iou
from apc.perception.card_verifier import ARCHITECTURE, CardFaceVerifier, proposal_tensor
from apc.perception.spatial_targets import load_split


def control_patch(rng):
    """Separate generated UI/background examples, never captured audit images."""
    w, h = rng.randint(24, 180), rng.randint(30, 140)
    image = Image.new("RGB", (w, h), rng.choice(("#10342a", "#102620", "#132b3d", "#2a2338")))
    draw = ImageDraw.Draw(image)
    mode = rng.randrange(4)
    if mode != 0:
        margin = rng.randint(0, max(1, min(w, h)//7))
        color = rng.choice(("#f4f0e3", "#99968e", "#b8e780", "#174b38"))
        draw.rounded_rectangle((margin, margin, w-margin-1, h-margin-1), radius=rng.randint(1, 8), fill=color)
        text = rng.choice(("Call", "Fold", "Raise", "100 BB", "2.5", "Check", "Player", "30", "Pot"))
        draw.text((margin+2, margin+max(2, (h-2*margin)//3)), text, fill="#202e26", font=ImageFont.load_default(size=rng.randint(10, 20)))
    return image


def patch_dataset(manifest, split, seed):
    metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if metadata.get("source_policy", {}).get("allowed_source_kinds") != ["synthetic_render"] or metadata.get("use_policy"):
        raise ValueError("Verifier training accepts only verified synthetic-render manifests")
    rng = random.Random(seed)
    tensors, labels, kinds = [], [], []
    for row in load_split(manifest, split):
        cards = [o for o in row["objects"] if o["class_name"] in ("hero_card", "board_card")]
        if any(not c["box_mask"] or c["visibility"] != "clear" for c in cards):
            raise ValueError("Verifier synthetic supervision requires complete clear card boxes")
        with Image.open(row["image"]["path"]) as image:
            def add(box, label, kind):
                tensors.append(proposal_tensor(image, box)); labels.append(label); kinds.append(kind)
            for card in cards:
                box = card["box_xyxy"]
                add(box, 1., "card_exact")
                # Detector overestimation and translation without cropping ink.
                for _ in range(2):
                    dx, dy = (box[2]-box[0])*rng.uniform(0, .18), (box[3]-box[1])*rng.uniform(0, .18)
                    add([max(0, box[0]-dx), max(0, box[1]-dy), min(1, box[2]+dx), min(1, box[3]+dy)], 1., "card_padded")
            for obj in row["objects"]:
                if obj["class_name"] in ("action_button", "pot", "seat") and obj["box_mask"]:
                    if all(box_iou(obj["box_xyxy"], c["box_xyxy"]) == 0 for c in cards):
                        add(obj["box_xyxy"], 0., "annotated_ui")
        for _ in range(12):
            patch = control_patch(rng)
            tensors.append(proposal_tensor(patch, [0, 0, 1, 1])); labels.append(0.); kinds.append("generated_ui_background")
    if not tensors or not any(labels) or all(labels):
        raise ValueError("Verifier split needs both positive and negative examples")
    return torch.stack(tensors), torch.tensor(labels), kinds


def metrics(model, data, threshold=.5):
    model.eval()
    with torch.no_grad():
        scores = torch.cat([model(batch).sigmoid() for batch in data[0].split(128)])
    actual, predicted = data[1].bool(), scores >= threshold
    tp, fp, fn, tn = (int(mask.sum()) for mask in (actual & predicted, ~actual & predicted, actual & ~predicted, ~actual & ~predicted))
    return {"samples": len(scores), "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "true_negatives": tn, "positive_recall": tp/(tp+fn), "negative_rejection": tn/(tn+fp),
            "balanced_accuracy": (tp/(tp+fn)+tn/(tn+fp))/2}


def train(manifest, destination, *, epochs=20, seed=20261112):
    destination = Path(destination)
    if destination.exists() or epochs < 1:
        raise ValueError("Use a new output directory and positive epochs")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    splits = {split: patch_dataset(manifest, split, seed+index) for index, split in enumerate(("train", "validation", "test"))}
    model = CardFaceVerifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    positive_weight = (1-splits["train"][1]).sum()/splits["train"][1].sum()
    best, selected, weights, history = -1, None, None, []
    for epoch in range(1, epochs+1):
        model.train()
        for indices in torch.randperm(len(splits["train"][0])).split(64):
            optimizer.zero_grad()
            inputs = splits["train"][0][indices]
            # Label-preserving brightness variation, train split only.
            inputs = (inputs*(.8+.4*torch.rand(len(inputs), 1, 1, 1))).clamp(0, 1)
            loss = F.binary_cross_entropy_with_logits(model(inputs), splits["train"][1][indices], pos_weight=positive_weight)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite verifier loss")
            loss.backward(); optimizer.step()
        validation = metrics(model, splits["validation"])
        if validation["balanced_accuracy"] > best:
            best, selected = validation["balanced_accuracy"], epoch
            weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
        history.append({"epoch": epoch, "validation": validation})
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(weights)
    report = {"architecture_version": ARCHITECTURE, "model_kind": "card_face_proposal_verifier", "seed": seed,
        "threshold": .5, "threshold_selection": "Fixed 0.5 before training; checkpoint uses validation balanced accuracy only",
        "selected_epoch": selected, "epochs": epochs, "history": history,
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "dataset_recipe": "exact_plus_two_padded_cards_annotated_nonoverlapping_ui_12_generated_controls_v1",
        "splits": {name: {"samples": len(data[1]), "positives": int(data[1].sum()),
            "kinds": {kind: data[2].count(kind) for kind in sorted(set(data[2]))}} for name, data in splits.items()},
        "test": metrics(model, splits["test"]), "synthetic_only": True, "audit_images_used": False,
        "confidence_calibrated": False, "full_table_readiness": False, "torch_version": str(torch.__version__)}
    destination.mkdir(parents=True)
    torch.save({"architecture_version": ARCHITECTURE, "model_kind": report["model_kind"], "state_dict": weights, "threshold": .5}, destination/"weights.pt")
    (destination/"evaluation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, epochs=args.epochs)["test"], indent=2))
