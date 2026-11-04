"""Compare frozen detector decoders without training or checkpoint selection."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from apc.perception.infer_card_locator import NeuralCardLocator
from apc.perception.train_card_locator import dataset, evaluate, frozen_spatial
from apc.perception.card_locator import decode_locator
from apc.perception.card_proposals import box_iou, match_proposals


def missed_card_diagnostics(model, data):
    failures = []
    with torch.no_grad():
        for frame_index, (image, truth) in enumerate(zip(data[0], data[2])):
            output = model(image[None])[0]
            grid = decode_locator(output)
            # IoU=1 retains every above-threshold cell for this oracle audit.
            dense = decode_locator(output, box_nms=True, iou_threshold=1)
            matched = {m["target_index"] for m in match_proposals(grid, truth)["matches"]}
            height, width = output.shape[1:]
            values = output.sigmoid()
            for target_index, target in enumerate(truth):
                if target_index in matched:
                    continue
                box = target["box_xyxy"]
                col, row = int((box[0]+box[2])*width/2), int((box[1]+box[3])*height/2)
                dx, dy, w, h = values[1:, row, col].tolist()
                cx, cy = (col+dx)/width, (row+dy)/height
                predicted = [max(0, cx-w/2), max(0, cy-h/2), min(1, cx+w/2), min(1, cy+h/2)]
                best = max((box_iou(p["box_xyxy"], box) for p in dense), default=0)
                score = float(values[0, row, col])
                reason = "grid_suppression_or_assignment" if best >= .5 else ("low_target_cell_score" if score < .5 else "box_geometry")
                failures.append({"frame_index": frame_index, "target_index": target_index,
                    "reason": reason, "target_cell_score": score,
                    "target_cell_box_iou": box_iou(predicted, box), "best_above_threshold_box_iou": best})
    return {"uses_ground_truth_for_diagnosis_only": True, "failures": failures,
        "reason_counts": {reason: sum(f["reason"] == reason for f in failures) for reason in sorted({f["reason"] for f in failures})}}


def compare(checkpoint, manifest, *, spatial_seed=44, select_validation=False):
    torch.set_num_threads(2)
    detector = NeuralCardLocator(checkpoint)
    selected, validation_audit = .5, None
    if select_validation:
        validation = dataset(manifest, "validation")
        varied = frozen_spatial(validation, spatial_seed-1)
        pooled = (torch.cat([validation[0], varied[0]]), torch.cat([validation[1], varied[1]]), validation[2]+varied[2])
        validation_audit = {value: evaluate(detector.model, pooled, box_nms=True, iou_threshold=value) for value in (.1, .2, .3, .5)}
        selected = max(validation_audit, key=lambda value: (validation_audit[value]["f1"], value))
    original = dataset(manifest, "test")
    spatial = frozen_spatial(original, spatial_seed)
    result = {}
    for name, data in (("original_test", original), ("frozen_spatial_test", spatial)):
        result[name] = {"grid_peak": evaluate(detector.model, data),
                        "box_overlap": evaluate(detector.model, data, box_nms=True, iou_threshold=selected)}
    return {"checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "spatial_seed": spatial_seed, "objectness_threshold": .5, "box_iou_threshold": selected,
        "validation_threshold_audit": validation_audit,
        "threshold_selection": "Highest original-plus-frozen validation F1, prefer higher IoU on ties" if select_validation else "Fixed 0.5 IoU diagnostic",
        "scopes": result, "training_performed": False, "weights_changed": False,
        "frozen_spatial_grid_misses": missed_card_diagnostics(detector.model, spatial),
        "default_decoder_changed": False, "full_table_readiness": False,
        "limitation": "Reused synthetic test diagnostic, not an independent captured-table readiness audit"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--select-validation", action="store_true")
    args = parser.parse_args()
    report = compare(args.checkpoint, args.manifest, select_validation=args.select_validation)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
