"""Image-bound, manually measured DOM-to-raster calibration for draft labels.

This validates geometric consistency, not the truth of human landmark matches.
It never grants annotation approval or model-readiness credit.
"""
import math

import numpy as np


def validate_alignment_landmarks(measured, snapshot):
    """Bind manual correspondences to recorded visible DOM, including controls.

    Optional control extrema are derived from immutable observed rectangles; no
    new DOM positions or screenshot pixel locations are invented here.
    """
    declared=snapshot.get("alignment_landmarks")
    if not isinstance(declared,list) or not declared:
        raise ValueError("Missing captured alignment landmarks")
    available={}
    for anchor in declared:
        if not isinstance(anchor,dict) or not isinstance(anchor.get("description"),str):
            raise ValueError("Invalid captured alignment landmark")
        name=anchor["description"]
        if name in available:
            raise ValueError("Duplicate captured alignment landmark")
        available[name]=anchor.get("dom_xy")
    base_names=set(available)
    public=snapshot.get("public_regions")
    if public is not None:
        if not isinstance(public,dict) or public.get("source") != "owned_practice_visible_dom_v1":
            raise ValueError("Unsupported captured public landmark source")
        buttons=public.get("action_buttons")
        if not isinstance(buttons,list):
            raise ValueError("Missing captured control rectangles")
        for row in buttons:
            if not isinstance(row,dict) or row.get("action") not in {"fold","check","call","bet","raise","all_in"}:
                raise ValueError("Invalid captured control")
            values=row.get("box_xywh")
            if (not isinstance(values,list) or len(values)!=4
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
                    or values[2]<=0 or values[3]<=0):
                raise ValueError("Invalid captured control rectangle")
            x,y,w,h=values
            for edge,point in (("top",[x+w/2,y]),("right",[x+w,y+h/2]),
                               ("bottom",[x+w/2,y+h]),("left",[x,y+h/2])):
                name=f"action {row['action']} outer border {edge} midpoint"
                if name in available:
                    raise ValueError("Duplicate captured control landmark")
                available[name]=point
    anchors=measured.get("anchors")
    if not isinstance(anchors,list):
        raise ValueError("Missing measured landmarks")
    seen=set()
    for anchor in anchors:
        if not isinstance(anchor,dict):
            raise ValueError("Invalid measured landmark")
        name=anchor.get("description")
        if not isinstance(name,str) or name in seen or name not in available or anchor.get("dom_xy")!=available[name]:
            raise ValueError("Measured landmarks do not match captured DOM evidence")
        seen.add(name)
    if not base_names.issubset(seen):
        raise ValueError("Measured landmarks must retain all captured table anchors")


class PixelAlignment:
    def __init__(self, evidence, *, image_sha256, image_size, viewport):
        if not isinstance(evidence, dict) or evidence.get("schema_version") != "apc_pixel_alignment_v1":
            raise ValueError("Missing pixel alignment evidence")
        if (evidence.get("image_sha256") != image_sha256
                or evidence.get("image_size") != list(image_size)
                or evidence.get("viewport") != list(viewport)):
            raise ValueError("Pixel alignment image/viewport binding mismatch")
        if evidence.get("measurement_method") != "manual_visible_landmarks":
            raise ValueError("Pixel landmarks must be measured independently of model predictions")
        anchors = evidence.get("anchors")
        if not isinstance(anchors, list) or len(anchors) < 4:
            raise ValueError("At least four visible landmark correspondences required")
        dom, pixels, names = [], [], set()
        for anchor in anchors:
            if not isinstance(anchor, dict):
                raise ValueError("Invalid landmark")
            name = anchor.get("description")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("Distinct landmark descriptions required")
            names.add(name)
            for key, bounds, destination in (("dom_xy", viewport, dom), ("pixel_xy", image_size, pixels)):
                value = anchor.get(key)
                if (not isinstance(value, list) or len(value) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                        or any(not 0 <= v <= b for v, b in zip(value, bounds))):
                    raise ValueError("Invalid landmark coordinates")
                destination.append(value)
        dom, pixels = np.asarray(dom, dtype=float), np.asarray(pixels, dtype=float)
        # Reject collinear or spatially clustered landmarks. Four copies of the
        # same corner cannot establish a full-frame mapping.
        span = np.ptp(dom, axis=0)
        if (np.any(span < np.asarray(viewport) * 0.25)
                or np.linalg.matrix_rank(np.column_stack((dom, np.ones(len(dom))))) != 3):
            raise ValueError("Landmarks must span both viewport axes")
        self.coefficients = []
        for axis in range(2):
            design = np.column_stack((dom[:, axis], np.ones(len(dom))))
            coef = np.linalg.lstsq(design, pixels[:, axis], rcond=None)[0]
            if coef[0] <= 0 or np.max(np.abs(design @ coef - pixels[:, axis])) > 2.0:
                raise ValueError("Landmarks disagree with axis-aligned pixel mapping")
            self.coefficients.append(coef)
        sx, sy = (coef[0] for coef in self.coefficients)
        if abs(sx - sy) / max(sx, sy) > 0.02:
            raise ValueError("Anisotropic screenshot mapping is unsupported")
        self.dom_min, self.dom_max = dom.min(axis=0), dom.max(axis=0)
        self.image_size = list(image_size)

    def box(self, x, y, width, height):
        values = (x, y, width, height)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or width <= 0 or height <= 0:
            raise ValueError("Invalid DOM box")
        if x < self.dom_min[0] or y < self.dom_min[1] or x + width > self.dom_max[0] or y + height > self.dom_max[1]:
            raise ValueError("Box outside measured landmark coverage")
        sx, tx = self.coefficients[0]
        sy, ty = self.coefficients[1]
        iw, ih = self.image_size
        box = [float((x * sx + tx) / iw), float((y * sy + ty) / ih),
               float(((x + width) * sx + tx) / iw), float(((y + height) * sy + ty) / ih)]
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError("Mapped box outside screenshot")
        return box
