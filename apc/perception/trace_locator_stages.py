"""Read-only locator tracing: decode, refinement, added-view policy, suppression."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch

from apc.perception.card_locator import decode_locator
from apc.perception.card_proposals import box_iou, refine_card_surface, suppress_duplicate_surfaces
from apc.perception.evaluate_browser_collection import load_collection
from apc.perception.infer_card_locator import NeuralCardLocator


def trace_frame(frame,locator):
    with Image.open(frame["image"]) as image:
        width,height = image.size
    views = [(0,0,width,height)]
    if locator.tiled_views:
        w,h = round(width*2/3),round(height*2/3)
        views += [(x,y,x+w,y+h) for y in (0,height-h) for x in (0,width-w)]
    stages = {"decoded_default":[],"box_nms_decode_control":[],"refined_all_views":[]}
    view_index = 0
    def mapped(rows,view):
        left,top,right,bottom = view
        return [dict(p,box_xyxy=[(left+p["box_xyxy"][0]*(right-left))/width,
            (top+p["box_xyxy"][1]*(bottom-top))/height,
            (left+p["box_xyxy"][2]*(right-left))/width,
            (top+p["box_xyxy"][3]*(bottom-top))/height]) for p in rows]
    def traced_decode(output):
        nonlocal view_index
        result = decode_locator(output)
        stages["decoded_default"].extend(mapped(copy.deepcopy(result),views[view_index]))
        stages["box_nms_decode_control"].extend(mapped(decode_locator(output,box_nms=True),views[view_index]))
        view_index += 1
        return result
    def traced_refine(*args,**kwargs):
        result = refine_card_surface(*args,**kwargs)
        stages["refined_all_views"].append(copy.deepcopy(result))
        return result
    def traced_suppress(rows):
        stages["after_added_view_policy"] = copy.deepcopy(rows)
        return suppress_duplicate_surfaces(rows)
    # Only observe intermediate values; wrappers return incumbent results.
    with patch("apc.perception.infer_card_locator.decode_locator",side_effect=traced_decode), \
         patch("apc.perception.card_proposals.refine_card_surface",side_effect=traced_refine), \
         patch("apc.perception.card_proposals.suppress_duplicate_surfaces",side_effect=traced_suppress):
        result = locator.propose(frame["image"])
    stages["final"] = result["proposals"]
    target_rows = []
    for target in frame["targets"]:
        overlaps = {name:max((box_iou(p["box_xyxy"],target["box_xyxy"]) for p in rows),default=0)
                    for name,rows in stages.items()}
        target_rows.append({"identity":target["identity"],"best_iou_by_stage":overlaps})
    return {"image":frame["image"].name,"targets":target_rows,
            "stage_proposal_counts":{name:len(rows) for name,rows in stages.items()},
            "box_nms_is_diagnostic_control_only":True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection",type=Path)
    parser.add_argument("--locator-checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    data,frames = load_collection(args.collection)
    torch.set_num_threads(2)
    locator = NeuralCardLocator(args.locator_checkpoint,refine_surfaces=True,adaptive_surfaces=True,
        deduplicate_surfaces=True,tiled_views=True)
    report = {"schema_version":"apc_locator_stage_trace_v1","training_performed":False,
        "readiness_credit":False,"reference_reuse_not_fresh_audit":True,
        "collection_sha256":hashlib.sha256(args.collection.read_bytes()).hexdigest(),
        "locator_sha256":hashlib.sha256(args.locator_checkpoint.read_bytes()).hexdigest(),
        "frames":[trace_frame(frame,locator) for frame in frames]}
    with args.output.open("x",encoding="utf-8") as output:
        output.write(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["frames"][2],indent=2))
