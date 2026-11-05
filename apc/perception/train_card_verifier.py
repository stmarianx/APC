"""Reproducible synthetic-only binary card-face training with session splits."""
import argparse
import hashlib
import json
import random
from functools import lru_cache
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


def control_scene_patch(rng):
    """Crops crossing multiple controls/panels, not just isolated buttons."""
    w, h = rng.randint(180, 360), rng.randint(160, 280)
    image = Image.new("RGB", (w, h), rng.choice(("#102620", "#123c30", "#15243a", "#2a2236")))
    draw = ImageDraw.Draw(image)
    columns, rows = rng.randint(2, 4), rng.randint(2, 4)
    cell_w, cell_h = w//columns, h//rows
    for row in range(rows):
        for col in range(columns):
            left, top = col*cell_w+rng.randint(1, 6), row*cell_h+rng.randint(1, 6)
            right, bottom = (col+1)*cell_w-rng.randint(2, 9), (row+1)*cell_h-rng.randint(2, 9)
            if rng.random() < .2:
                continue
            draw.rounded_rectangle((left, top, right, bottom), radius=rng.randint(1, 7),
                fill=rng.choice(("#eee9dd", "#9e9b92", "#cae791", "#244b39")))
            font = ImageFont.load_default(size=rng.randint(10, 24))
            for line in range(rng.randint(1, 3)):
                draw.text((left+3, top+3+line*18), rng.choice(("Fold", "Call", "Stack", "100 BB", "Player", "Check", "Raise", "Pot", "2.5 BB")),
                          font=font, fill="#23322b")
    crop_w, crop_h = rng.randint(30, w), rng.randint(30, h)
    left, top = rng.randint(0, w-crop_w), rng.randint(0, h-crop_h)
    return image.crop((left, top, left+crop_w, top+crop_h))


@lru_cache(maxsize=128)
def _face_font(path, size):
    return ImageFont.truetype(path, size=size)


def generated_card_patch(rng, fonts):
    """Known synthetic rank/suit faces with independent shape/appearance."""
    h = rng.randint(64, 100)
    w = rng.randint(40, min(76, round(h*.95)))
    image = Image.new("RGB", (w, h), rng.choice(("#102620", "#183b34", "#112b40", "#282638")))
    draw = ImageDraw.Draw(image)
    margin = rng.choice((0, 0, 1, 2))
    face = rng.choice(("#f5f3e9", "#f8f7f1", "#eeeadc", "#aaa79e", "#8b8983"))
    draw.rounded_rectangle((margin, margin, w-1-margin, h-1-margin), radius=rng.randint(2, 8), fill=face,
                          outline=rng.choice((face, face, "#535650")))
    font = _face_font(str(rng.choice(fonts)), rng.randint(14, min(28, round(h*.35))))
    rank, suit = rng.choice("23456789TJQKA"), rng.choice(("♠", "♥", "♦", "♣"))
    ink = rng.choice(("#131d19", "#d0313e", "#276bc5", "#228454", "#242424"))
    centered = rng.choice((True, False))
    for text, top in ((rank, round(h*rng.uniform(.08, .15))), (suit, round(h*rng.uniform(.47, .56)))):
        x = (w-draw.textlength(text, font=font))/2 if centered else rng.randint(4, 8)
        draw.text((x, top), text, font=font, fill=ink, anchor="lt")
    return image


def patch_dataset(manifest, split, seed, *, negative_scenes=0, positive_aspect_variants=0, generated_positive_faces=0, positive_fonts=()):
    if type(negative_scenes) is not int or not 0 <= negative_scenes <= 64:
        raise ValueError("Expected 0 through 64 composite negatives per frame")
    if type(positive_aspect_variants) is not int or not 0 <= positive_aspect_variants <= 8:
        raise ValueError("Expected 0 through 8 positive aspect variants per card")
    if type(generated_positive_faces) is not int or not 0 <= generated_positive_faces <= 32 or (generated_positive_faces and not positive_fonts):
        raise ValueError("Generated positive faces need 0 through 32 examples and explicit fonts")
    if generated_positive_faces:
        from apc.synthetic.render_table import validate_symbol_font
        for font in positive_fonts:
            validate_symbol_font(font, (14, 28))
    metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if metadata.get("source_policy", {}).get("allowed_source_kinds") != ["synthetic_render"] or metadata.get("use_policy"):
        raise ValueError("Verifier training accepts only verified synthetic-render manifests")
    rng = random.Random(seed)
    scene_rng = random.Random(seed+10_000_000)
    aspect_rng = random.Random(seed+20_000_000)
    face_rng = random.Random(seed+30_000_000)
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
                for _ in range(positive_aspect_variants):
                    coords = [round(v*d) for v, d in zip(box, image.size*2)]
                    crop = image.crop(coords)
                    size = (max(1, round(crop.width*aspect_rng.uniform(.8, 1.25))),
                            max(1, round(crop.height*aspect_rng.uniform(.8, 1.25))))
                    crop = crop.resize(size, Image.Resampling.BILINEAR)
                    tensors.append(proposal_tensor(crop, [0, 0, 1, 1])); labels.append(1.); kinds.append("card_aspect_variant")
            for obj in row["objects"]:
                if obj["class_name"] in ("action_button", "pot", "seat") and obj["box_mask"]:
                    if all(box_iou(obj["box_xyxy"], c["box_xyxy"]) == 0 for c in cards):
                        add(obj["box_xyxy"], 0., "annotated_ui")
        for _ in range(12):
            patch = control_patch(rng)
            tensors.append(proposal_tensor(patch, [0, 0, 1, 1])); labels.append(0.); kinds.append("generated_ui_background")
        for _ in range(negative_scenes):
            patch = control_scene_patch(scene_rng)
            tensors.append(proposal_tensor(patch, [0, 0, 1, 1])); labels.append(0.); kinds.append("generated_composite_ui")
        for _ in range(generated_positive_faces):
            patch = generated_card_patch(face_rng, positive_fonts)
            tensors.append(proposal_tensor(patch, [0, 0, 1, 1])); labels.append(1.); kinds.append("generated_card_face")
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


def train(manifest, destination, *, epochs=20, seed=20261112, negative_scenes=0, positive_aspect_variants=0, generated_positive_faces=0, positive_fonts=()):
    destination = Path(destination)
    if destination.exists() or epochs < 1:
        raise ValueError("Use a new output directory and positive epochs")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    splits = {split: patch_dataset(manifest, split, seed+index, negative_scenes=negative_scenes,
        positive_aspect_variants=positive_aspect_variants, generated_positive_faces=generated_positive_faces,
        positive_fonts=positive_fonts) for index, split in enumerate(("train", "validation", "test"))}
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
        "extra_composite_negatives_per_frame": negative_scenes,
        "extra_positive_aspect_variants_per_card": positive_aspect_variants,
        "generated_positive_faces_per_frame": generated_positive_faces,
        "positive_font_sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in positive_fonts},
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
    parser.add_argument("--negative-scenes", type=int, default=0, help="Additional composite UI crops per source frame")
    parser.add_argument("--positive-aspect-variants", type=int, default=0, help="Complete-card aspect/scale variation per source card")
    parser.add_argument("--generated-positive-faces", type=int, default=0)
    parser.add_argument("--positive-font", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, epochs=args.epochs, negative_scenes=args.negative_scenes,
        positive_aspect_variants=args.positive_aspect_variants, generated_positive_faces=args.generated_positive_faces,
        positive_fonts=tuple(args.positive_font))["test"], indent=2))
