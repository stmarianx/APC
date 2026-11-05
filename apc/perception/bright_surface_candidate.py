"""Opt-in research recovery when a proposal's median samples mostly dark felt."""
import numpy as np
from PIL import Image

from apc.perception.card_proposals import scanline_components, suppress_duplicate_surfaces
from apc.perception.verified_card_reader import VerifiedCardReader


def recover_surface(image,proposal,pixels=None,*,wide_search=False,fast_components=False):
    if proposal.get("refinement") == "adaptive_bounded_surface":
        return dict(proposal)
    pixels = np.asarray(image.convert("RGB"),dtype=np.float32) if pixels is None else pixels
    width,height = image.size
    x1,y1,x2,y2 = [round(v*d) for v,d in zip(proposal["box_xyxy"],image.size*2)]
    crop = pixels[y1:y2,x1:x2].reshape(-1,3)
    if not crop.size:
        return dict(proposal)
    bright = crop[crop.min(1) >= 150]
    if len(bright) < max(8,.15*len(crop)):
        return dict(proposal)
    surface = np.median(bright,axis=0)
    original_area = (x2-x1)*(y2-y1)
    expansions = (.35,.7,1.4,2.8) if wide_search else (.35,.7,1.4)
    for expansion in expansions:
        dx,dy = round((x2-x1)*expansion),round((y2-y1)*expansion)
        left,top,right,bottom = max(0,x1-dx),max(0,y1-dy),min(width,x2+dx),min(height,y2+dy)
        mask = np.abs(pixels[top:bottom,left:right]-surface).max(2) < 25
        candidates = []
        extractor = scanline_components
        if fast_components:
            from apc.perception.fast_components import fast_components as extractor
        for a,b,c,d,area in extractor(mask):
            if a == 0 or b == 0 or c == right-left or d == bottom-top:
                continue
            bounds = [left+a,top+b,left+c,top+d]
            candidate_area = (c-a)*(d-b)
            overlap = max(0,min(x2,bounds[2])-max(x1,bounds[0]))*max(0,min(y2,bounds[3])-max(y1,bounds[1]))
            containment = overlap/min(original_area,candidate_area)
            if (.35 <= (c-a)/(d-b) <= 1.2 and area/candidate_area >= .55
                    and .4*original_area <= candidate_area <= 8*original_area and containment >= .7):
                candidates.append((containment,area,bounds))
        if candidates:
            bounds = max(candidates,key=lambda row:(row[0],row[1]))[2]
            return dict(proposal,box_xyxy=[v/s for v,s in zip(bounds,image.size*2)],
                        refinement="adaptive_bounded_surface",surface_recovery="bright_seed_wide_v2" if wide_search else "bright_seed_v1")
    return dict(proposal)


class BrightRecoveryLocator:
    def __init__(self,locator,*,wide_search=False,fast_components=False):
        self.locator = locator
        self.wide_search = wide_search
        self.fast_components = fast_components

    def propose(self,path):
        result = self.locator.propose(path)
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"),dtype=np.float32)
            proposals = [recover_surface(image,p,pixels,wide_search=self.wide_search,fast_components=self.fast_components) for p in result["proposals"]]
        return dict(result,proposals=suppress_duplicate_surfaces(proposals),surface_recovery="bright_seed_wide_v2" if self.wide_search else "bright_seed_v1")


class BrightSurfaceReader(VerifiedCardReader):
    def __init__(self,*args,wide_search=False,fast_components=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.locator.locator = BrightRecoveryLocator(self.locator.locator,wide_search=wide_search,fast_components=fast_components)


if __name__ == "__main__":
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import torch
    from apc.perception.evaluate_browser_collection import evaluate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wide-search",action="store_true",help="Opt-in v2 extra expansion for narrow proposals")
    parser.add_argument("--verifier-checkpoint",type=Path,help="Explicit experimental ambiguous-surface verifier; bounded expert unchanged")
    parser.add_argument("--checkpoint-dir",type=Path,default=Path(__file__).resolve().parents[1]/"checkpoints")
    args = parser.parse_args()
    p = args.checkpoint_dir
    paths = {"glyph":p/"glyph-font-diversity-v1.pt","locator":p/"card-locator-hardneg-v1.pt",
             "verifier":p/"card-face-verifier-appearance-v4.pt","bounded":p/"card-face-verifier-v1.pt"}
    if args.verifier_checkpoint is not None:
        paths["verifier"] = args.verifier_checkpoint
    torch.set_num_threads(2)
    reader = BrightSurfaceReader(paths["glyph"],corner_pairs=True,surface_pairs=True,
        locator_checkpoint=paths["locator"],refine_surfaces=True,adaptive_surfaces=True,
        deduplicate_surfaces=True,tiled_views=True,verifier_checkpoint=paths["verifier"],
        bounded_verifier_checkpoint=paths["bounded"],wide_search=args.wide_search)
    report = evaluate(args.collection,reader)
    report.update(experiment="bright_seed_wide_v2" if args.wide_search else "bright_seed_v1",promotion_authorized=False,
        candidate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()})
    with args.output.open("x",encoding="utf-8") as output:
        output.write(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["totals"]))
