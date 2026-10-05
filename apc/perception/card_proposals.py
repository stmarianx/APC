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
