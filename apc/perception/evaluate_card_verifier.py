"""Paired synthetic verifier diagnostics, never captured-table readiness."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from apc.perception.card_verifier import ProposalVerifier
from apc.perception.train_card_verifier import patch_dataset


def binary_counts(actual, predicted):
    actual, predicted = actual.bool(), predicted.bool()
    tp, fp, fn, tn = (int(mask.sum()) for mask in (actual & predicted, ~actual & predicted, actual & ~predicted, ~actual & ~predicted))
    return {"samples": len(actual), "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "true_negatives": tn,
            "positive_recall": tp/(tp+fn) if tp+fn else None,
            "negative_rejection": tn/(tn+fp) if tn+fp else None}


def compare(manifest, checkpoints, *, negative_scenes=8, seed=20261114, positive_aspect_variants=0, generated_positive_faces=0, positive_fonts=(), text_fragments=0):
    torch.set_num_threads(2)
    data = patch_dataset(manifest, "test", seed, negative_scenes=negative_scenes,
        positive_aspect_variants=positive_aspect_variants, generated_positive_faces=generated_positive_faces, positive_fonts=positive_fonts,text_fragments=text_fragments)
    reports = []
    for checkpoint in checkpoints:
        verifier = ProposalVerifier(checkpoint)
        with torch.no_grad():
            scores = torch.cat([verifier.model(batch).sigmoid() for batch in data[0].split(128)])
        predicted = scores >= verifier.threshold
        per_kind = {}
        for kind in sorted(set(data[2])):
            indices = torch.tensor([i for i, value in enumerate(data[2]) if value == kind])
            per_kind[kind] = binary_counts(data[1][indices], predicted[indices])
        reports.append({"checkpoint": str(checkpoint), "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                        "threshold": verifier.threshold, "totals": binary_counts(data[1], predicted), "per_kind": per_kind})
    return {"reports": reports, "split": "test", "seed": seed, "negative_scenes": negative_scenes,
        "positive_aspect_variants": positive_aspect_variants, "generated_positive_faces": generated_positive_faces,
        "text_fragments": text_fragments,
        "positive_font_sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in positive_fonts},
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "synthetic_test_reuse": True, "checkpoint_selection_performed": False,
        "captured_images_used": False, "full_table_readiness": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-aspect-variants", type=int, default=0)
    parser.add_argument("--generated-positive-faces", type=int, default=0)
    parser.add_argument("--positive-font", type=Path, action="append", default=[])
    parser.add_argument("--text-fragments",type=int,default=0)
    args = parser.parse_args()
    report = compare(args.manifest, args.checkpoint, positive_aspect_variants=args.positive_aspect_variants,
                     generated_positive_faces=args.generated_positive_faces, positive_fonts=tuple(args.positive_font),text_fragments=args.text_fragments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps([{key: row[key] for key in ("checkpoint", "totals")} for row in report["reports"]], indent=2))
