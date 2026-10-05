"""Pixel-proposal + neural card recognition; no hand labels at inference."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from apc.perception.card_proposals import propose_cards
from apc.perception.region_model import RegionRecognizer, RANKS, SUITS
from apc.perception.train_regions import image_tensor


class CardReader:
    def __init__(self, checkpoint: str | Path, *, confidence_threshold=.9):
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be finite in [0,1]")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("architecture_version") != "spatial-grid-v2":
            raise ValueError("Unsupported region architecture")
        if saved.get("model_kind") != "region_recognizer_requires_proposed_boxes":
            raise ValueError("Unsupported checkpoint kind")
        self.model = RegionRecognizer(saved["crop_size"])
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()
        self.threshold = confidence_threshold
        self.max_image_dimension = saved.get("max_image_dimension", 1280)

    def read(self, image_path: str | Path):
        started = time.perf_counter()
        proposals = propose_cards(image_path)
        cards = []
        if proposals:
            image = image_tensor({"image": {"path": str(image_path)}}, size=self.max_image_dimension)
            regions = torch.tensor([[0, *item["box_xyxy"]] for item in proposals], dtype=torch.float32)
            with torch.no_grad():
                predictions = self.model(image, regions)
                ranks = predictions["rank"].softmax(-1)
                suits = predictions["suit"].softmax(-1)
            for index, proposal in enumerate(proposals):
                rank_score, rank_index = ranks[index].max(0)
                suit_score, suit_index = suits[index].max(0)
                score = min(float(rank_score), float(suit_score))
                accepted = score >= self.threshold
                cards.append({"box_xyxy": proposal["box_xyxy"],
                              "rank_candidate": RANKS[int(rank_index)],
                              "suit_candidate": SUITS[int(suit_index)],
                              "rank_score": float(rank_score), "suit_score": float(suit_score),
                              "candidate": RANKS[int(rank_index)] + SUITS[int(suit_index)],
                              "card": None,
                              "status": "high_score_unvalidated" if accepted else "uncertain",
                              "role": "unresolved"})
            identities = [item["candidate"] for item in cards if item["status"] == "high_score_unvalidated"]
            for item in cards:
                if item["status"] == "high_score_unvalidated" and identities.count(item["candidate"]) > 1:
                    item.update(card=None, status="duplicate_conflict")
        return {"schema_version": "1.0.0", "image_path": str(Path(image_path).resolve()),
                "cards": cards, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "confidence_threshold": self.threshold, "confidence_calibrated": False,
                "full_table_readiness": False,
                "limitations": ["light_card_proposals_only", "hero_board_roles_unresolved",
                                "rank_accuracy_not_ready", "not_for_coaching"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=.9)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(CardReader(args.checkpoint, confidence_threshold=args.confidence_threshold).read(args.image), indent=2))


if __name__ == "__main__":
    main()
