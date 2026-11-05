"""Opt-in light-surface proposals supplement missed neural peaks; not identities."""
import numpy as np
from PIL import Image

from apc.perception.card_proposals import suppress_duplicate_surfaces
from apc.perception.fast_components import fast_components


def surface_proposals(image,analysis_width=640):
    if type(analysis_width) is not int or analysis_width < 160:
        raise ValueError("Expected analysis width >=160")
    scale = min(1,analysis_width/image.width)
    image = image.convert("RGB").resize((round(image.width*scale),max(1,round(image.height*scale))))
    pixels = np.asarray(image,dtype=np.int16)
    height,width = pixels.shape[:2]
    mask = (pixels.min(2) >= 165) & ((pixels.max(2)-pixels.min(2)) <= 65)
    proposals = []
    # Same broad surface shape/size limits as the existing light-card baseline;
    # no card count, table position, identity or captured annotation is used.
    for left,top,right,bottom,area in fast_components(mask):
        w,h = right-left,bottom-top
        fill = area/(w*h)
        if (.45 <= w/h <= .95 and w >= width*.018 and h >= height*.045
                and area >= width*height*.0008 and fill >= .65
                and w < width*.2 and h < height*.3):
            proposals.append({"box_xyxy":[left/width,top/height,right/width,bottom/height],
                "score":0.,"surface_fill":fill,"kind":"light_surface_rescue",
                "refinement":"pixel_component_surface","recognition_required":True})
    return sorted(proposals,key=lambda p:(p["box_xyxy"][1],p["box_xyxy"][0]))


class SurfaceRescueLocator:
    def __init__(self,base):
        self.base = base

    def propose(self,path):
        result = self.base.propose(path)
        with Image.open(path) as image:
            additions = surface_proposals(image)
        # Neural/refined boxes retain precedence through their original scores.
        # All additional boxes still go through learned verification and glyphs.
        return dict(result,proposals=suppress_duplicate_surfaces(result["proposals"]+additions),
                    surface_rescue="light_components_v1")
