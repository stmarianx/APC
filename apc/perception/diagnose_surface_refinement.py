"""Paired crop diagnostic on reused browser references, never a readiness audit."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from apc.perception.card_proposals import ADAPTIVE_SURFACE_VERSION, match_proposals, refine_card_surface
from apc.perception.evaluate_browser_collection import load_collection
from apc.perception.infer_card_locator import NeuralCardLocator
from apc.perception.infer_cards import CardReader
from apc.perception.region_model import RANKS, SUITS


def diagnose(collection, glyph_checkpoint, locator_checkpoint):
    _, frames = load_collection(collection)
    reader = CardReader(glyph_checkpoint, corner_pairs=True, surface_pairs=True)
    locator = NeuralCardLocator(locator_checkpoint)
    rows = []
    for frame in frames:
        raw = locator.propose(frame["image"])["proposals"]
        with Image.open(frame["image"]) as image:
            modes = {"raw": raw, "legacy": [refine_card_surface(image, p) for p in raw],
                     "adaptive": [refine_card_surface(image, p, adaptive=True) for p in raw]}
        results = {}
        for mode, proposals in modes.items():
            matched = match_proposals(proposals, frame["targets"])
            predictions = reader.predict_regions(frame["image"], [p["box_xyxy"] for p in proposals]) if proposals else None
            matches = {m["proposal_index"]: m for m in matched["matches"]}
            items = []
            for index, proposal in enumerate(proposals):
                match = matches.get(index)
                target = frame["targets"][match["target_index"]] if match else None
                candidate = (RANKS[int(predictions["rank"][index].argmax())] + SUITS[int(predictions["suit"][index].argmax())]
                             if predictions["ink_present"][index] else None)
                items.append(dict(proposal, candidate=candidate,
                    target=target["identity"] if target else None, iou=match["iou"] if match else None,
                    exact=bool(target and candidate == target["identity"])))
            results[mode] = items
        rows.append({"image": frame["image"].name, "modes": results})
    return {"reference_reuse": True, "training_performed": False, "full_table_readiness": False,
            "surface_refinement_version": ADAPTIVE_SURFACE_VERSION,
            "selection_uses_labels": False, "diagnostic_matching_uses_labels": True,
            "collection_sha256": hashlib.sha256(Path(collection).read_bytes()).hexdigest(),
            "checkpoints": {"glyph": hashlib.sha256(Path(glyph_checkpoint).read_bytes()).hexdigest(),
                            "locator": hashlib.sha256(Path(locator_checkpoint).read_bytes()).hexdigest()},
            "frames": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("glyph_checkpoint", type=Path)
    parser.add_argument("locator_checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = diagnose(args.collection, args.glyph_checkpoint, args.locator_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for frame in report["frames"]:
        print(frame["image"], {mode: sum(p["exact"] for p in items) for mode, items in frame["modes"].items()})
