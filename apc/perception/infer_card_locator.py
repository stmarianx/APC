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
    def __init__(self, checkpoint, *, refine_surfaces=False):
        self.refine_surfaces = refine_surfaces
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
            pixels = np.asarray(image.convert("RGB").resize((320, 184)), dtype=np.float32).copy() / 255
        inputs = torch.from_numpy(pixels).permute(2, 0, 1)[None]
        with torch.no_grad():
            proposals = decode_locator(self.model(inputs)[0])
        if self.refine_surfaces:
            from apc.perception.card_proposals import refine_card_surface
            original_pixels = np.asarray(original, dtype=np.float32)
            proposals = [refine_card_surface(original, proposal, pixels=original_pixels) for proposal in proposals]
        return {"proposals": proposals, "elapsed_ms": (time.perf_counter()-started)*1000,
                "confidence_calibrated": False, "full_table_readiness": False,
                "limitation": "Experimental boxes only, no card identities or confirmed hero/board roles"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(NeuralCardLocator(args.checkpoint).propose(args.image), indent=2))
