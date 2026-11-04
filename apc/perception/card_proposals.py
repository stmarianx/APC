"""Annotation-free light-card proposals; explicitly limited appearance baseline.

This supplies candidate boxes from frame pixels, not stored training geometry.
It can miss dark/occluded cards and must be evaluated separately from recognition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from apc.perception.table_locator import _components


def propose_cards(image_path: str | Path, *, analysis_width: int = 640):
    if analysis_width < 160:
        raise ValueError("analysis_width must be at least 160")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        scale = min(1, analysis_width / image.width)
        image = image.resize((round(image.width * scale), max(1, round(image.height * scale))))
    pixels = np.asarray(image, dtype=np.int16)
    height, width = pixels.shape[:2]
    # Bright near-neutral surfaces survive rank/suit glyph holes. No labels,
    # session IDs, board count, or known table layout enter this computation.
    mask = (pixels.min(2) >= 165) & ((pixels.max(2) - pixels.min(2)) <= 65)
    proposals = []
    for left, top, right, bottom, area in _components(mask):
        w, h = right - left, bottom - top
        aspect, fill = w / h, area / (w * h)
        if (.45 <= aspect <= .95 and w >= width * .018 and h >= height * .045
                and area >= width * height * .0008 and fill >= .65
                and w < width * .2 and h < height * .3):
            proposals.append({"box_xyxy": [left / width, top / height, right / width, bottom / height],
                              "surface_fill": fill, "kind": "card_candidate",
                              "recognition_required": True})
    return sorted(proposals, key=lambda item: (item["box_xyxy"][1], item["box_xyxy"][0]))


def box_iou(left, right):
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection) if intersection else 0


def scanline_components(mask):
    """Four-connected components using row runs, not a Python visit per pixel."""
    parents, runs, previous = [], [], []
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for y, row in enumerate(mask):
        changes = np.flatnonzero(np.diff(np.r_[False, row, False]))
        current = []
        for left, right in zip(changes[::2], changes[1::2]):
            index = len(parents)
            parents.append(index)
            for a,b,other in previous:
                if b <= left:
                    continue
                if a >= right:
                    break
                one, two = root(index), root(other)
                parents[max(one,two)] = min(one,two)
            current.append((int(left),int(right),index))
            runs.append((int(left),int(right),y,index))
        previous = current
    groups = {}
    for left,right,y,index in runs:
        key = root(index)
        if key not in groups:
            groups[key] = [left,y,right,y+1,right-left]
        else:
            item = groups[key]
            item[0],item[1],item[2],item[3] = min(item[0],left),min(item[1],y),max(item[2],right),max(item[3],y+1)
            item[4] += right-left
    return [tuple(groups[key]) for key in sorted(groups)]


def refine_card_surface(image, proposal, *, pixels=None):
    """Experimental local edge refinement, independent of card identity/labels."""
    width, height = image.size
    box = proposal["box_xyxy"]
    x1, y1, x2, y2 = [round(v*d) for v, d in zip(box, (width,height,width,height))]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Invalid proposal bounds")
    if pixels is None:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    surface = np.median(pixels[y1:y2,x1:x2].reshape(-1,3), axis=0)
    dx, dy = round((x2-x1)*.35), round((y2-y1)*.35)
    left, top, right, bottom = max(0,x1-dx), max(0,y1-dy), min(width,x2+dx), min(height,y2+dy)
    local = pixels[top:bottom,left:right]
    mask = np.abs(local-surface).max(2) < 25
    components = scanline_components(mask)
    candidates = []
    for a,b,c,d,area in components:
        candidate = [(left+a)/width,(top+b)/height,(left+c)/width,(top+d)/height]
        if area >= .4*(x2-x1)*(y2-y1) and box_iou(box,candidate) >= .4:
            candidates.append((area,candidate))
    if not candidates:
        return dict(proposal, refinement="no_surface_match")
    return dict(proposal, box_xyxy=max(candidates, key=lambda row: row[0])[1],
                refinement="local_surface_component")


def match_proposals(proposals, targets, threshold=.5):
    """One-to-one IoU audit; no duplicated credit for one detected card."""
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    pairs = sorted(((box_iou(p["box_xyxy"], t["box_xyxy"]), pi, ti)
                    for pi, p in enumerate(proposals) for ti, t in enumerate(targets)), reverse=True)
    used_proposals, used_targets, matches = set(), set(), []
    for overlap, pi, ti in pairs:
        if overlap >= threshold and pi not in used_proposals and ti not in used_targets:
            used_proposals.add(pi)
            used_targets.add(ti)
            matches.append({"proposal_index": pi, "target_index": ti, "iou": overlap})
    return {"matches": matches, "true_positives": len(matches),
            "false_positives": len(proposals) - len(matches),
            "false_negatives": len(targets) - len(matches)}
