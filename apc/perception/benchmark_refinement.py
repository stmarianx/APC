"""Matched end-to-end benchmark of equivalent surface-component implementations."""
import argparse
import json
import torch
import apc.perception.card_proposals as proposals
from apc.perception.table_locator import _components
from apc.perception.evaluate_card_reader import evaluate_reader


def compare(reader, locator, manifest):
    optimized = proposals.scanline_components
    reports = {}
    try:
        for name, implementation in (("pixel", _components), ("scanline", optimized)):
            proposals.scanline_components = implementation
            reports[name] = evaluate_reader(reader, manifest, corner_pairs=True,
                                           locator_checkpoint=locator, refine_surfaces=True)
    finally:
        proposals.scanline_components = optimized
    fields = ("candidate_exact_correct", "candidate_rank_correct", "candidate_suit_correct",
              "true_positives", "false_positives", "false_negatives", "frames_all_card_candidates_exact",
              "high_score_candidates", "high_score_correct")
    return {"reports": reports, "same_accuracy_counts": all(
        reports["pixel"][field] == reports["scanline"][field] for field in fields),
        "limitations": "Sequential CPU benchmark; no captured-table or full-coach latency claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reader")
    parser.add_argument("locator")
    parser.add_argument("manifest")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(compare(args.reader, args.locator, args.manifest), indent=2))
