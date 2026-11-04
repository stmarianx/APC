"""Pixel-proposal + neural card recognition; no hand labels at inference."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image

from apc.perception.card_proposals import propose_cards
from apc.perception.region_model import RegionRecognizer, RANKS, SUITS
from apc.perception.train_regions import image_tensor
from apc.perception.card_glyphs import CardGlyphNetwork, glyph_tensor, corner_pair_tensor, adaptive_pair_tensor


class CardReader:
    def __init__(self, checkpoint: str | Path, *, confidence_threshold=.9, corner_pairs=False, locator_checkpoint=None, refine_surfaces=False, surface_pairs=False):
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be finite in [0,1]")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.glyph_model = saved.get("model_kind") == "card_glyph_recognizer"
        self.token_extractor = glyph_tensor
        paired_checkpoint = saved.get("model_kind") == "annotated_card_glyph_pair_recognizer" and saved.get("architecture_version") == "card-glyph-pair-v1"
        if surface_pairs and (not corner_pairs or not paired_checkpoint):
            raise ValueError("Surface pair extraction requires a paired glyph checkpoint")
        if corner_pairs and saved.get("model_kind") == "annotated_card_glyph_pair_recognizer" and saved.get("architecture_version") == "card-glyph-pair-v1":
            self.glyph_model = True
            self.token_extractor = adaptive_pair_tensor if surface_pairs else corner_pair_tensor
            self.model = CardGlyphNetwork(input_channels=6)
        elif self.glyph_model and saved.get("architecture_version") == "card-glyph-v1":
            self.model = CardGlyphNetwork()
        elif saved.get("architecture_version") == "spatial-grid-v2" and saved.get("model_kind") == "region_recognizer_requires_proposed_boxes":
            self.model = RegionRecognizer(saved["crop_size"])
        else:
            raise ValueError("Unsupported checkpoint kind or architecture")
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()
        self.pair_extraction = "surface_two_band_then_corner" if surface_pairs else ("upper_left_corner" if self.token_extractor is corner_pair_tensor else "not_paired")
        self.locator = None
        if locator_checkpoint is not None:
            from apc.perception.infer_card_locator import NeuralCardLocator
            self.locator = NeuralCardLocator(locator_checkpoint, refine_surfaces=refine_surfaces)
        self.threshold = confidence_threshold
        self.max_image_dimension = saved.get("max_image_dimension", 1280)

    def predict_regions(self, image_path, boxes):
        with torch.no_grad():
            if self.glyph_model:
                with Image.open(image_path) as image:
                    tokens = [self.token_extractor(image, box) for box in boxes]
                result = self.model(torch.stack([item[0] for item in tokens]))
                result["ink_present"] = torch.tensor([item[1] for item in tokens], dtype=torch.bool)
                return result
            image = image_tensor({"image": {"path": str(image_path)}}, size=self.max_image_dimension)
            regions = torch.tensor([[0, *box] for box in boxes], dtype=torch.float32)
            return self.model(image, regions)

    def read(self, image_path: str | Path):
        started = time.perf_counter()
        locator = getattr(self, "locator", None)
        proposals = propose_cards(image_path) if locator is None else locator.propose(image_path)["proposals"]
        cards = []
        if proposals:
            with torch.no_grad():
                predictions = self.predict_regions(image_path, [item["box_xyxy"] for item in proposals])
                ranks = predictions["rank"].softmax(-1)
                suits = predictions["suit"].softmax(-1)
            for index, proposal in enumerate(proposals):
                rank_score, rank_index = ranks[index].max(0)
                suit_score, suit_index = suits[index].max(0)
                score = min(float(rank_score), float(suit_score))
                ink_present = bool(predictions["ink_present"][index]) if "ink_present" in predictions else True
                accepted = ink_present and score >= self.threshold
                cards.append({"box_xyxy": proposal["box_xyxy"],
                              "rank_candidate": RANKS[int(rank_index)] if ink_present else None,
                              "suit_candidate": SUITS[int(suit_index)] if ink_present else None,
                              "rank_score": float(rank_score) if ink_present else 0,
                              "suit_score": float(suit_score) if ink_present else 0,
                              "candidate": RANKS[int(rank_index)] + SUITS[int(suit_index)] if ink_present else None,
                              "card": None,
                              "status": "missing_ink" if not ink_present else ("high_score_unvalidated" if accepted else "uncertain"),
                              "role": "unresolved"})
            identities = [item["candidate"] for item in cards if item["status"] == "high_score_unvalidated"]
            for item in cards:
                if item["status"] == "high_score_unvalidated" and identities.count(item["candidate"]) > 1:
                    item.update(card=None, status="duplicate_conflict")
        return {"schema_version": "1.0.0", "image_path": str(Path(image_path).resolve()),
                "cards": cards, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "confidence_threshold": self.threshold, "confidence_calibrated": False,
                "pair_extraction": getattr(self, "pair_extraction", "unspecified"),
                "full_table_readiness": False,
                "limitations": ["light_card_proposals_only" if locator is None else "experimental_neural_proposals",
                                "hero_board_roles_unresolved",
                                "rank_accuracy_not_ready", "not_for_coaching"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=.9)
    parser.add_argument("--corner-symbols", action="store_true", help="Use separate corner rank/suit extraction for paired checkpoints")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(CardReader(args.checkpoint, confidence_threshold=args.confidence_threshold,
                                corner_pairs=args.corner_symbols).read(args.image), indent=2))


if __name__ == "__main__":
    main()
