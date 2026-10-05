"""End-to-end card candidate audit with independent pixel proposals."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from apc.perception.card_proposals import match_proposals
from apc.perception.infer_cards import CardReader
from apc.perception.spatial_targets import load_split
from apc.perception.region_model import RANKS, SUITS
from apc.perception.train_regions import image_tensor


def evaluate_reader(checkpoint, manifest, split="test"):
    reader = CardReader(checkpoint)
    totals = {"frames": 0, "targets": 0, "true_positives": 0,
              "false_positives": 0, "false_negatives": 0,
              "candidate_exact_correct": 0, "high_score_candidates": 0,
              "high_score_correct": 0, "high_score_candidates_all": 0}
    latency = []
    for target in load_split(manifest, split):
        cards = [item for item in target["objects"] if item["class_name"] in ("hero_card", "board_card")
                 and item["box_mask"] and item["attribute_masks"].get("rank")
                 and item["attribute_masks"].get("suit")]
        result = reader.read(target["image"]["path"])
        totals["high_score_candidates_all"] += sum(
            min(item["rank_score"], item["suit_score"]) >= reader.threshold for item in result["cards"])
        match = match_proposals(result["cards"], cards)
        totals["frames"] += 1
        totals["targets"] += len(cards)
        for key in ("true_positives", "false_positives", "false_negatives"):
            totals[key] += match[key]
        for item in match["matches"]:
            predicted = result["cards"][item["proposal_index"]]
            truth = cards[item["target_index"]]["attributes"]
            correct = predicted["candidate"] == truth["rank"] + truth["suit"]
            totals["candidate_exact_correct"] += int(correct)
            high = min(predicted["rank_score"], predicted["suit_score"]) >= reader.threshold
            totals["high_score_candidates"] += int(high)
            totals["high_score_correct"] += int(high and correct)
        latency.append(result["elapsed_ms"])
    totals.update({"candidate_exact_recall": totals["candidate_exact_correct"] / totals["targets"] if totals["targets"] else None,
                   "high_score_matched_precision": totals["high_score_correct"] / totals["high_score_candidates"] if totals["high_score_candidates"] else None,
                   "high_score_end_to_end_precision": totals["high_score_correct"] / totals["high_score_candidates_all"] if totals["high_score_candidates_all"] else None,
                   "p95_frame_ms": float(np.percentile(latency, 95)) if latency else None,
                   "latency_scope": "image_io_proposals_neural_not_checkpoint_loading",
                   "confirmed_cards_emitted": 0, "full_table_readiness": False})
    return totals


def compare_crops(checkpoint, manifest, split="validation"):
    """Paired diagnostic only: annotations supply the exact-box comparison."""
    reader = CardReader(checkpoint)
    result = {"matched_cards": 0, "exact_box_correct": 0, "proposed_box_correct": 0,
              "exact_correct_proposed_wrong": 0, "candidate_disagreements": 0}
    overlaps, shifts = [], []
    for target in load_split(manifest, split):
        cards = [item for item in target["objects"] if item["class_name"] in ("hero_card", "board_card")
                 and item["attribute_masks"].get("rank") and item["attribute_masks"].get("suit")]
        if not cards:
            continue
        detected = reader.read(target["image"]["path"])["cards"]
        matched = match_proposals(detected, cards)["matches"]
        regions = torch.tensor([[0, *card["box_xyxy"]] for card in cards], dtype=torch.float32)
        with torch.no_grad():
            logits = reader.model(image_tensor(target), regions)
        for match in matched:
            index = match["target_index"]
            truth = cards[index]["attributes"]
            identity = truth["rank"] + truth["suit"]
            exact = RANKS[int(logits["rank"][index].argmax())] + SUITS[int(logits["suit"][index].argmax())]
            proposed = detected[match["proposal_index"]]
            result["matched_cards"] += 1
            result["exact_box_correct"] += int(exact == identity)
            result["proposed_box_correct"] += int(proposed["candidate"] == identity)
            result["exact_correct_proposed_wrong"] += int(exact == identity and proposed["candidate"] != identity)
            result["candidate_disagreements"] += int(exact != proposed["candidate"])
            overlaps.append(match["iou"])
            dimensions = [target["image"]["width"], target["image"]["height"]] * 2
            shifts.append(max(abs(a - b) * d for a, b, d in zip(cards[index]["box_xyxy"], proposed["box_xyxy"], dimensions)))
    result.update(mean_iou=float(np.mean(overlaps)) if overlaps else None,
                  mean_max_edge_shift_pixels=float(np.mean(shifts)) if shifts else None,
                  diagnostic_uses_annotations=True, full_table_readiness=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("manifest")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--compare-crops", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    function = compare_crops if args.compare_crops else evaluate_reader
    print(json.dumps(function(args.checkpoint, args.manifest, args.split), indent=2))


if __name__ == "__main__":
    main()
