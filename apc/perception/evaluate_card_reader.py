"""End-to-end card candidate audit with independent pixel proposals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from apc.perception.card_proposals import match_proposals
from apc.perception.infer_cards import CardReader
from apc.perception.spatial_targets import load_split
from apc.perception.region_model import RANKS, SUITS


def evaluate_reader(checkpoint, manifest, split="test", *, corner_pairs=False, locator_checkpoint=None, refine_surfaces=False, surface_pairs=False):
    options = {"corner_pairs": True} if corner_pairs else {}
    if surface_pairs:
        options["surface_pairs"] = True
    if locator_checkpoint is not None:
        options["locator_checkpoint"] = locator_checkpoint
        options["refine_surfaces"] = refine_surfaces
    reader = CardReader(checkpoint, **options)
    totals = {"frames": 0, "targets": 0, "true_positives": 0,
              "false_positives": 0, "false_negatives": 0,
              "candidate_rank_correct": 0, "candidate_suit_correct": 0,
              "candidate_exact_correct": 0, "high_score_candidates": 0,
              "high_score_correct": 0, "high_score_candidates_all": 0,
              "confirmed_cards_emitted": 0, "frames_with_card_targets": 0,
              "frames_all_card_candidates_exact": 0, "frames_with_high_score_error": 0,
              "frames_excluded_partial_card_labels": 0, "excluded_partial_frame_proposals": 0}
    latency = []
    for target in load_split(manifest, split):
        cards = [item for item in target["objects"] if item["class_name"] in ("hero_card", "board_card")]
        result = reader.read(target["image"]["path"])
        totals["frames"] += 1
        latency.append(result["elapsed_ms"])
        totals["confirmed_cards_emitted"] += sum(item.get("card") is not None for item in result["cards"])
        # Dropping just the unknown target can make an incomplete frame appear
        # perfect, or penalize a true proposal as an invented card.
        if any(not (card["box_mask"] and card["attribute_masks"].get("rank")
                    and card["attribute_masks"].get("suit")) for card in cards):
            totals["frames_excluded_partial_card_labels"] += 1
            totals["excluded_partial_frame_proposals"] += len(result["cards"])
            continue
        frame_high = sum(
            min(item["rank_score"], item["suit_score"]) >= reader.threshold for item in result["cards"])
        totals["high_score_candidates_all"] += frame_high
        frame_exact = frame_high_correct = 0
        match = match_proposals(result["cards"], cards)
        totals["targets"] += len(cards)
        for key in ("true_positives", "false_positives", "false_negatives"):
            totals[key] += match[key]
        for item in match["matches"]:
            predicted = result["cards"][item["proposal_index"]]
            truth = cards[item["target_index"]]["attributes"]
            correct = predicted["candidate"] == truth["rank"] + truth["suit"]
            candidate = predicted["candidate"] or ""
            totals["candidate_rank_correct"] += int(candidate[:-1] == truth["rank"])
            totals["candidate_suit_correct"] += int(candidate[-1:] == truth["suit"])
            totals["candidate_exact_correct"] += int(correct)
            high = min(predicted["rank_score"], predicted["suit_score"]) >= reader.threshold
            totals["high_score_candidates"] += int(high)
            totals["high_score_correct"] += int(high and correct)
            frame_exact += int(correct)
            frame_high_correct += int(high and correct)
        totals["frames_with_card_targets"] += int(bool(cards))
        totals["frames_all_card_candidates_exact"] += int(bool(cards) and frame_exact == len(cards)
                                                          and not match["false_positives"])
        totals["frames_with_high_score_error"] += int(frame_high > frame_high_correct)
    totals.update({"candidate_exact_recall": totals["candidate_exact_correct"] / totals["targets"] if totals["targets"] else None,
                   "high_score_matched_precision": totals["high_score_correct"] / totals["high_score_candidates"] if totals["high_score_candidates"] else None,
                   "high_score_end_to_end_precision": totals["high_score_correct"] / totals["high_score_candidates_all"] if totals["high_score_candidates_all"] else None,
                   "p95_frame_ms": float(np.percentile(latency, 95)) if latency else None,
                   "latency_scope": "image_io_proposals_neural_not_checkpoint_loading",
                   "high_score_wrong_matched": totals["high_score_candidates"] - totals["high_score_correct"],
                   "high_score_unmatched": totals["high_score_candidates_all"] - totals["high_score_candidates"],
                   "full_table_readiness": False})
    return totals


def compare_crops(checkpoint, manifest, split="validation", *, corner_pairs=False):
    """Paired diagnostic only: annotations supply the exact-box comparison."""
    reader = CardReader(checkpoint, corner_pairs=True) if corner_pairs else CardReader(checkpoint)
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
        with torch.no_grad():
            logits = reader.predict_regions(target["image"]["path"], [card["box_xyxy"] for card in cards])
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
    parser.add_argument("--corner-symbols", action="store_true", help="Use independent corner rank/suit proposals")
    parser.add_argument("--surface-pairs", action="store_true", help="Experimental full-face two-band extraction")
    parser.add_argument("--locator-checkpoint", type=Path)
    parser.add_argument("--refine-surfaces", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.compare_crops:
        if args.surface_pairs or args.locator_checkpoint or args.refine_surfaces:
            parser.error("Extended extraction/locator options are currently supported only by the end-to-end audit")
        report = compare_crops(args.checkpoint, args.manifest, args.split, corner_pairs=args.corner_symbols)
    else:
        report = evaluate_reader(args.checkpoint, args.manifest, args.split, corner_pairs=args.corner_symbols,
            surface_pairs=args.surface_pairs, locator_checkpoint=args.locator_checkpoint, refine_surfaces=args.refine_surfaces)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
