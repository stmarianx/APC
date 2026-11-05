"""Explicit versioned adaptive locator path; incumbent frozen sources unchanged."""
import time
import numpy as np
import torch
from PIL import Image

from apc.perception.card_locator import decode_locator
from apc.perception.card_proposals import suppress_duplicate_surfaces
from apc.perception.fast_components import fast_components


def refine_fast(image,proposal,pixels):
    width,height = image.size
    x1,y1,x2,y2 = [round(v*d) for v,d in zip(proposal["box_xyxy"],image.size*2)]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Invalid proposal bounds")
    surface = np.median(pixels[y1:y2,x1:x2].reshape(-1,3),axis=0)
    original_area = (x2-x1)*(y2-y1)
    for expansion in (.35,.7,1.4):
        dx,dy = round((x2-x1)*expansion),round((y2-y1)*expansion)
        left,top,right,bottom = max(0,x1-dx),max(0,y1-dy),min(width,x2+dx),min(height,y2+dy)
        mask = np.abs(pixels[top:bottom,left:right]-surface).max(2) < 25
        candidates = []
        for a,b,c,d,area in fast_components(mask):
            if a == 0 or b == 0 or c == right-left or d == bottom-top:
                continue
            bounds = [left+a,top+b,left+c,top+d]
            candidate_area = (c-a)*(d-b)
            intersection = max(0,min(x2,bounds[2])-max(x1,bounds[0]))*max(0,min(y2,bounds[3])-max(y1,bounds[1]))
            containment = intersection/min(original_area,candidate_area)
            if (.35 <= (c-a)/(d-b) <= 1.2 and area/candidate_area >= .55
                    and .4*original_area <= candidate_area <= 8*original_area and containment >= .7):
                candidates.append((containment,area,[v/s for v,s in zip(bounds,image.size*2)]))
        if candidates:
            return dict(proposal,box_xyxy=max(candidates,key=lambda row:(row[0],row[1]))[2],
                        refinement="adaptive_bounded_surface",expansion=expansion)
    return dict(proposal,refinement="adaptive_no_bounded_surface")


class FastRefinementLocator:
    """Reuse loaded neural weights, preserve decode/refinement/suppression order.

    This version supports adaptive refinement only. It has no global monkeypatch
    and no image/result cache, so independent readers cannot affect each other.
    """
    def __init__(self,base):
        if not base.refine_surfaces or not base.adaptive_surfaces:
            raise ValueError("Fast refinement requires the adaptive refined locator")
        self.base = base

    def propose(self,path):
        started = time.perf_counter()
        base = self.base
        with Image.open(path) as image:
            original = image.convert("RGB")
        views = [(0,0,original.width,original.height)]
        if base.tiled_views:
            w,h = round(original.width*2/3),round(original.height*2/3)
            views += [(x,y,x+w,y+h) for y in (0,original.height-h) for x in (0,original.width-w)]
        inputs = torch.stack([torch.from_numpy(np.asarray(original.crop(view).resize((320,184)),dtype=np.float32).copy()/255).permute(2,0,1) for view in views])
        with torch.no_grad():
            outputs = base.model(inputs)
        proposals = []
        for view_index,(view,output) in enumerate(zip(views,outputs)):
            left,top,right,bottom = view
            for proposal in decode_locator(output):
                box = proposal["box_xyxy"]
                proposal["box_xyxy"] = [(left+box[0]*(right-left))/original.width,
                    (top+box[1]*(bottom-top))/original.height,(left+box[2]*(right-left))/original.width,
                    (top+box[3]*(bottom-top))/original.height]
                if base.tiled_views:
                    proposal["source_view"] = view_index
                proposals.append(proposal)
        pixels = np.asarray(original,dtype=np.float32)
        proposals = [refine_fast(original,p,pixels) for p in proposals]
        if base.tiled_views:
            proposals = [p for p in proposals if p["source_view"] == 0 or p.get("refinement") == "adaptive_bounded_surface"]
        if base.deduplicate_surfaces:
            proposals = suppress_duplicate_surfaces(proposals)
        return {"proposals":proposals,"elapsed_ms":(time.perf_counter()-started)*1000,
            "view_count":len(views),"deduplicate_surfaces":base.deduplicate_surfaces,
            "added_view_policy":"bounded_surfaces_only" if base.tiled_views else None,
            "confidence_calibrated":False,"full_table_readiness":False,
            "limitation":"Experimental boxes only, no card identities or confirmed hero/board roles"}
