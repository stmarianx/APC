"""Train neural card-token recognition with session-isolated supervision."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F

from apc.perception.card_glyphs import CardGlyphNetwork, glyph_tensor
from apc.perception.region_model import RANKS, SUITS
from apc.perception.spatial_targets import load_split


def token_dataset(manifest, split, *, glyph_pair=False):
    tokens, ranks, suits = [], [], []
    for target in load_split(manifest, split):
        with Image.open(target["image"]["path"]) as image:
            for obj in target["objects"]:
                if obj["class_name"] not in ("hero_card", "board_card"):
                    continue
                if not obj["attribute_masks"].get("rank") or not obj["attribute_masks"].get("suit"):
                    continue
                if glyph_pair:
                    regions = obj.get("recognition_regions", {})
                    if any(not regions.get(key, {}).get("supervised") for key in ("rank", "suit")):
                        raise ValueError("Glyph-pair training requires explicit verified rank and suit boxes")
                    coords = [round(value * dimension) for value, dimension in zip(obj["box_xyxy"], (image.width, image.height) * 2)]
                    background = np.median(np.asarray(image.convert("RGB").crop(coords)).reshape(-1, 3), axis=0)
                    parts = []
                    for key in ("rank", "suit"):
                        box = regions[key]["box"]
                        xyxy = [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]]
                        parts.append(glyph_tensor(image, xyxy, border_margin=False, background_rgb=background))
                    token, present = torch.cat([part[0] for part in parts]), all(part[1] for part in parts)
                else:
                    token, present = glyph_tensor(image, obj["box_xyxy"])
                if not present:
                    raise ValueError("Verified card has no extractable ink; cannot silently drop supervision")
                tokens.append(token)
                ranks.append(RANKS.index(obj["attributes"]["rank"]))
                suits.append(SUITS.index(obj["attributes"]["suit"]))
    if not tokens:
        raise ValueError("Split has no known card tokens")
    return torch.stack(tokens), torch.tensor(ranks), torch.tensor(suits)


def metrics(model, dataset):
    model.eval()
    with torch.no_grad():
        output = model(dataset[0])
        rank_correct = output["rank"].argmax(1) == dataset[1]
        suit_correct = output["suit"].argmax(1) == dataset[2]
    return {"cards": len(dataset[0]), "rank_correct": int(rank_correct.sum()),
            "suit_correct": int(suit_correct.sum()), "exact_correct": int((rank_correct & suit_correct).sum())}


def augment_tokens(tokens):
    """Training-only color/scale/translation variation; never changes labels."""
    if tokens.ndim != 4 or tokens.shape[1] not in (3, 6) or tokens.shape[2:] != (32, 32):
        raise ValueError("Expected RGB glyph tokens or paired RGB glyph tokens")
    shape = tokens.shape
    parts = tokens.reshape(-1, 3, 32, 32)
    size = int(torch.randint(20, 33, ()).item())
    parts = F.interpolate(F.interpolate(parts, size=(size, size), mode="bilinear", align_corners=False),
                          size=(32, 32), mode="bilinear", align_corners=False)
    gray = parts.amax(dim=1, keepdim=True).expand_as(parts)
    use_gray = torch.rand(len(parts), 1, 1, 1, device=parts.device) < .5
    parts = torch.where(use_gray, gray, parts)
    parts = (parts * (.75 + .5 * torch.rand(len(parts), 1, 1, 1, device=parts.device))).clamp(0, 1)
    dx, dy = torch.randint(0, 5, (2,)).tolist()
    parts = F.pad(parts, (2, 2, 2, 2))[:, :, dy:dy+32, dx:dx+32]
    return parts.reshape(shape)


def audit_source_isolation(manifests):
    """Reject duplicated manifests/images, including cross-source split leakage."""
    seen_manifests, seen_images, seen_sessions, sources = set(), {}, {}, []
    for manifest in manifests:
        raw = Path(manifest).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen_manifests:
            raise ValueError("Duplicate training source manifest")
        seen_manifests.add(digest)
        metadata = json.loads(raw)
        synthetic = metadata.get("source_policy", {}).get("allowed_source_kinds") == ["synthetic_render"]
        namespace = metadata["dataset_id"] if synthetic else "controlled_capture"
        counts = {}
        for split in ("train", "validation", "test"):
            frames = load_split(manifest, split)
            counts[split] = len(frames)
            for frame in frames:
                session = (namespace, frame["capture_session_id"])
                if session in seen_sessions and seen_sessions[session] != split:
                    raise ValueError("Capture session crosses source splits")
                seen_sessions[session] = split
                fingerprint = frame["image"]["sha256"]
                if fingerprint in seen_images:
                    raise ValueError("Duplicate source image or cross-source split leakage")
                seen_images[fingerprint] = (digest, split)
        sources.append({"manifest_sha256": digest, "frames_by_split": counts})
    return sources


def train(manifest, output, epochs=15, seed=42, *, glyph_pair=False, augment=False, rehearsal_manifests=()):
    output = Path(output)
    if output.exists() or epochs < 1:
        raise ValueError("Use a new output directory and positive epochs")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    manifests = [manifest, *rehearsal_manifests]
    source_audit = audit_source_isolation(manifests) if rehearsal_manifests else None
    source_datasets = [{split: token_dataset(source, split, glyph_pair=glyph_pair)
                       for split in ("train", "validation", "test")} for source in manifests]
    datasets = {split: tuple(torch.cat([source[split][part] for source in source_datasets]) for part in range(3))
                for split in ("train", "validation", "test")}
    model = CardGlyphNetwork(input_channels=6 if glyph_pair else 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    best, weights, history, selected_epoch = None, None, [], None
    for epoch in range(epochs):
        model.train()
        data = datasets["train"]
        order = torch.randperm(len(data[0]))
        for indices in order.split(64):
            optimizer.zero_grad()
            tokens = data[0][indices]
            result = model(augment_tokens(tokens) if augment else tokens)
            loss = F.cross_entropy(result["rank"], data[1][indices]) + F.cross_entropy(result["suit"], data[2][indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            optimizer.step()
        by_source = [metrics(model, source["validation"]) for source in source_datasets]
        validation = {key: sum(result[key] for result in by_source) for key in by_source[0]}
        key = (min(result["exact_correct"] / result["cards"] for result in by_source),
               validation["exact_correct"] / validation["cards"])
        if best is None or key > best:
            best = key
            selected_epoch = epoch + 1
            weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "validation": validation})
        if rehearsal_manifests:
            history[-1]["validation_by_source"] = by_source
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(weights)
    report = {"model_kind": "card_glyph_recognizer", "architecture_version": "card-glyph-v1",
              "seed": seed, "epochs": epochs, "torch_version": torch.__version__,
              "token_augmentation": "color_scale_translation_v1" if augment else None,
              "selected_epoch": selected_epoch,
              "selection_rule": "max_min_domain_validation_accuracy_then_pooled_accuracy",
              "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "history": history, "test_exact_box": metrics(model, datasets["test"]),
              "full_table_readiness": False, "confidence_calibrated": False}
    if rehearsal_manifests:
        report["source_isolation_audit"] = source_audit
        report["test_by_source"] = [metrics(model, source["test"]) for source in source_datasets]
    if glyph_pair:
        report.update(model_kind="annotated_card_glyph_pair_recognizer", architecture_version="card-glyph-pair-v1")
    output.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "model_kind": report["model_kind"],
                "architecture_version": report["architecture_version"]}, output / "glyph_weights.pt")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--glyph-pair", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Training-only color, scale and translation variation")
    parser.add_argument("--rehearsal-manifest", action="append", default=[], help="Additional disjoint source with its own train/validation/test split")
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, args.epochs, glyph_pair=args.glyph_pair,
        augment=args.augment, rehearsal_manifests=args.rehearsal_manifest)["test_exact_box"], indent=2))
