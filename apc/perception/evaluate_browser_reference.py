"""Provisional card audit on a frozen browser capture, never a readiness gate."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from apc.perception.card_proposals import match_proposals
from apc.perception.infer_cards import CardReader


REFERENCE = Path(__file__).resolve().parents[1] / "data/reference/browser-training-v1"
IMAGE_SHA256 = "69bf88519279924a7f07c5a04e7d34f4837e87f48ecf7ad5b295242d56b3cdd9"


def provisional_targets(labels):
    width, height = labels["viewport"]["width"], labels["viewport"]["height"]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid DOM viewport")
    suits = {"♠": "s", "♥": "h", "♣": "c", "♦": "d"}
    targets = []
    for role in ("hero", "board"):
        for item in labels[role]:
            rank, suit = item["text"].split()
            rank = "T" if rank == "10" else rank
            if rank not in "23456789TJQKA" or len(rank) != 1 or suit not in suits:
                raise ValueError("Unknown visible card label")
            box = item["box"]
            bounds = [box["x"]/width, box["y"]/height,
                      (box["x"]+box["width"])/width, (box["y"]+box["height"])/height]
            if not (0 <= bounds[0] < bounds[2] <= 1 and 0 <= bounds[1] < bounds[3] <= 1):
                raise ValueError("Label lies outside DOM viewport")
            targets.append({"box_xyxy": bounds, "identity": rank+suits[suit], "role": role})
    return targets


def evaluate():
    image = REFERENCE / "river-frame.png"
    if hashlib.sha256(image.read_bytes()).hexdigest() != IMAGE_SHA256:
        raise ValueError("Frozen image changed")
    labels = json.loads((REFERENCE / "river-visible-labels.json").read_text(encoding="utf-8"))
    targets = provisional_targets(labels)
    checkpoints = Path(__file__).resolve().parents[1] / "checkpoints"
    reader = CardReader(checkpoints / "glyph-surface-pair-v1.pt", corner_pairs=True,
                        locator_checkpoint=checkpoints / "card-locator-mixed-v1.pt", refine_surfaces=True)
    result = reader.read(image)
    matches = match_proposals(result["cards"], targets)
    exact = sum(result["cards"][row["proposal_index"]]["candidate"] == targets[row["target_index"]]["identity"]
                for row in matches["matches"])
    return {"audit_only": True, "labels": "DOM-derived, independent geometry review pending",
            "targets": len(targets), "proposals": len(result["cards"]), "localization": matches,
            "exact_candidate_correct": exact,
            "confirmed_cards": sum(card["card"] is not None for card in result["cards"]),
            "full_table_readiness": False}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(2)
    print(json.dumps(evaluate(), indent=2))
