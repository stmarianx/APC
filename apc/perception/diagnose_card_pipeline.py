"""Attribute draft-reference failures without changing models or thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch

from apc.perception.card_proposals import box_iou, match_proposals
from apc.perception.evaluate_browser_collection import load_collection
from apc.perception.region_model import RANKS, SUITS
from apc.perception.verified_card_reader import VerifiedCardReader


def diagnose_frame(frame, reader):
    locator = reader.locator
    raw = locator.locator.propose(frame["image"])["proposals"]
    indexed = [dict(p, diagnostic_index=i) for i,p in enumerate(raw)]
    with Image.open(frame["image"]) as image:
        kept, rejected = locator.verifier.filter(image,indexed)
    matched = match_proposals(kept,frame["targets"])
    target_matches = {m["target_index"]:m for m in matched["matches"]}
    candidate_by_index = {}
    if kept:
        predictions = reader.predict_regions(frame["image"],[p["box_xyxy"] for p in kept])
        for index, (rank,suit,ink) in enumerate(zip(predictions["rank"].argmax(1).tolist(),
                predictions["suit"].argmax(1).tolist(),predictions["ink_present"].tolist())):
            candidate_by_index[index] = RANKS[rank]+SUITS[suit] if ink else None
    def overlaps(proposals,target):
        return [{"diagnostic_index":p["diagnostic_index"],"iou":box_iou(p["box_xyxy"],target["box_xyxy"]),
                 "verifier_score":p.get("verifier_score"),"verifier_route":p.get("verifier_route"),
                 "refinement":p.get("refinement")} for p in proposals
                if box_iou(p["box_xyxy"],target["box_xyxy"]) >= .5]
    targets = []
    for index,target in enumerate(frame["targets"]):
        match = target_matches.get(index)
        candidate = candidate_by_index[match["proposal_index"]] if match else None
        raw_matches = overlaps(indexed,target)
        dropped = overlaps(rejected,target)
        failure = ("correct" if candidate == target["identity"] else "recognition") if match else (
            "verifier_rejection" if dropped else "assignment_conflict" if raw_matches else "locator_or_refinement")
        targets.append({"identity":target["identity"],"stage":failure,"candidate":candidate,
                        "raw_matches":raw_matches,"rejected_matches":dropped})
    used = {m["proposal_index"] for m in matched["matches"]}
    false_proposals = [dict(p,candidate=candidate_by_index[i]) for i,p in enumerate(kept) if i not in used]
    return {"image":frame["image"].name,"frame_key":frame.get("frame_key",frame["image"].as_posix()),
            "hand_group":frame.get("hand_group"),"street":frame.get("street"),"raw_proposals":len(raw),"retained":len(kept),
            "rejected":len(rejected),"targets":targets,"false_proposals":false_proposals}


def diagnose_collection(collection,reader):
    data,frames = load_collection(collection)
    root = Path(collection).resolve().parent
    rows = [diagnose_frame(dict(frame,frame_key=frame["image"].relative_to(root).as_posix(),street=row["street"]),reader)
            for frame,row in zip(frames,data["frames"])]
    counts = {}
    for row in rows:
        for target in row["targets"]:
            counts[target["stage"]] = counts.get(target["stage"],0)+1
    return {"schema_version":"apc_card_pipeline_diagnostic_v1",
            "collection_sha256":hashlib.sha256(Path(collection).read_bytes()).hexdigest(),
            "capture_session_id":data["capture_session_id"],"frames":rows,"target_stage_counts":counts,
            "proposal_verifier":reader.verifier_metadata,"iou_threshold":.5,
            "reference_reuse_not_fresh_audit":True,"training_performed":False,"readiness_credit":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection",type=Path)
    for name in ("glyph", "locator", "verifier", "bounded-verifier"):
        parser.add_argument("--"+name+"-checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wide-search",action="store_true")
    parser.add_argument("--fast-components",action="store_true")
    parser.add_argument("--fast-refinement",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    reader_type,extra = VerifiedCardReader,{}
    if args.fast_components or args.fast_refinement:
        if not args.wide_search:
            parser.error("Fast diagnostics require explicit wide-search candidate")
    if args.wide_search:
        from apc.perception.bright_surface_candidate import BrightSurfaceReader
        reader_type = BrightSurfaceReader
        extra = dict(wide_search=True,fast_components=args.fast_components,fast_refinement=args.fast_refinement)
    reader = reader_type(args.glyph_checkpoint,corner_pairs=True,surface_pairs=True,
        locator_checkpoint=args.locator_checkpoint,refine_surfaces=True,adaptive_surfaces=True,
        deduplicate_surfaces=True,tiled_views=True,verifier_checkpoint=args.verifier_checkpoint,
        bounded_verifier_checkpoint=args.bounded_verifier_checkpoint,**extra)
    with torch.no_grad():
        report = diagnose_collection(args.collection,reader)
    report["candidate_options"] = extra
    report["checkpoints"] = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
        (("glyph",args.glyph_checkpoint),("locator",args.locator_checkpoint),
         ("verifier",args.verifier_checkpoint),("bounded_verifier",args.bounded_verifier_checkpoint))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as output:
        output.write(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["target_stage_counts"]))
