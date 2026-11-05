"""Inspect uncalibrated neural card proposals on a saved frame."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from apc.perception.card_locator import CardLocator, decode_locator


class NeuralCardLocator:
    def __init__(self, checkpoint, *, refine_surfaces=False, adaptive_surfaces=False, deduplicate_surfaces=False, tiled_views=False):
        self.refine_surfaces = refine_surfaces
        if adaptive_surfaces and not refine_surfaces:
            raise ValueError("Adaptive surfaces require surface refinement")
        self.adaptive_surfaces = adaptive_surfaces
        if (deduplicate_surfaces or tiled_views) and not adaptive_surfaces:
            raise ValueError("Surface deduplication/tiled views require adaptive refinement")
        self.deduplicate_surfaces = deduplicate_surfaces
        self.tiled_views = tiled_views
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved.get("model_kind") != "full_frame_neural_card_locator" or
                saved.get("architecture_version") != "card-locator-v1" or saved.get("input_size") != [320, 184]):
            raise ValueError("Unsupported card locator checkpoint")
        self.model = CardLocator()
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()

    def propose(self, path):
        started = time.perf_counter()
        with Image.open(path) as image:
            original = image.convert("RGB")
        views = [(0, 0, original.width, original.height)]
        if self.tiled_views:
            # Uniform overlapping windows, independent of table roles/counts.
            w, h = round(original.width*2/3), round(original.height*2/3)
            views += [(x, y, x+w, y+h) for y in (0, original.height-h) for x in (0, original.width-w)]
        inputs = torch.stack([torch.from_numpy(np.asarray(original.crop(view).resize((320, 184)), dtype=np.float32).copy()/255).permute(2, 0, 1) for view in views])
        with torch.no_grad():
            outputs = self.model(inputs)
        proposals = []
        for view_index, (view, output) in enumerate(zip(views, outputs)):
            left, top, right, bottom = view
            for proposal in decode_locator(output):
                box = proposal["box_xyxy"]
                proposal["box_xyxy"] = [(left+box[0]*(right-left))/original.width,
                    (top+box[1]*(bottom-top))/original.height,
                    (left+box[2]*(right-left))/original.width,
                    (top+box[3]*(bottom-top))/original.height]
                if self.tiled_views:
                    proposal["source_view"] = view_index
                proposals.append(proposal)
        if self.refine_surfaces:
            from apc.perception.card_proposals import refine_card_surface
            original_pixels = np.asarray(original, dtype=np.float32)
            proposals = [refine_card_surface(original, proposal, pixels=original_pixels, adaptive=self.adaptive_surfaces) for proposal in proposals]
        if self.tiled_views:
            # Zoomed views are out-of-training-distribution; accept additions
            # only with bounded pixel-surface evidence. Retain original-view
            # proposals regardless, rather than silently changing its recall.
            proposals = [p for p in proposals if p["source_view"] == 0 or p.get("refinement") == "adaptive_bounded_surface"]
        if self.deduplicate_surfaces:
            from apc.perception.card_proposals import suppress_duplicate_surfaces
            proposals = suppress_duplicate_surfaces(proposals)
        return {"proposals": proposals, "elapsed_ms": (time.perf_counter()-started)*1000,
                "view_count": len(views), "deduplicate_surfaces": self.deduplicate_surfaces,
                "added_view_policy": "bounded_surfaces_only" if self.tiled_views else None,
                "confidence_calibrated": False, "full_table_readiness": False,
                "limitation": "Experimental boxes only, no card identities or confirmed hero/board roles"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(NeuralCardLocator(args.checkpoint).propose(args.image), indent=2))
