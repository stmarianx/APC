"""Matched sequential baseline/verifier timing; no coaching-readiness claim."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from apc.perception.evaluate_browser_collection import load_collection
from apc.perception.infer_cards import CardReader
from apc.perception.verified_card_reader import VerifiedCardReader


def benchmark(collection, glyph, locator, verifier, repeats=3, *, bounded_verifier=None):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Expected positive timing repeats")
    torch.set_num_threads(2)
    _, frames = load_collection(collection)
    options = dict(corner_pairs=True, surface_pairs=True, locator_checkpoint=locator,
                   refine_surfaces=True, adaptive_surfaces=True, deduplicate_surfaces=True, tiled_views=True)
    verifier_options = {"bounded_verifier_checkpoint": bounded_verifier} if bounded_verifier is not None else {}
    readers = {"baseline": CardReader(glyph, **options),
               "verified": VerifiedCardReader(glyph, verifier_checkpoint=verifier, **options, **verifier_options)}
    for reader in readers.values():
        reader.read(frames[0]["image"])
    observations = []
    for repeat in range(repeats):
        for frame in frames:
            order = ("baseline", "verified") if repeat % 2 == 0 else ("verified", "baseline")
            for mode in order:
                started = time.perf_counter()
                result = readers[mode].read(frame["image"])
                observations.append({"repeat": repeat, "image": frame["image"].name, "mode": mode,
                    "elapsed_ms": (time.perf_counter()-started)*1000, "proposals": len(result["cards"])})
    summary = {}
    for mode in readers:
        times = [r["elapsed_ms"] for r in observations if r["mode"] == mode]
        summary[mode] = {"samples": len(times), "p50_ms": float(np.percentile(times, 50)),
                         "p95_ms": float(np.percentile(times, 95)), "max_ms": max(times)}
    return {"reference_reuse": True, "repeats": repeats, "torch_threads": 2, "summary": summary,
        "warmup": "One untimed first-frame read per reader; alternating variant order between repeats",
        "scope": "Sequential warm saved-frame reading, includes image I/O/proposals/verification/glyphs; excludes model loading, screen capture, state tracking, strategy and rendering",
        "limitations": "Small reused reference set, shared laptop runtime; not a production or full-coach latency gate",
        "collection_sha256": hashlib.sha256(Path(collection).read_bytes()).hexdigest(),
        "checkpoint_sha256": {str(Path(p).name): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ((glyph, locator, verifier) + ((bounded_verifier,) if bounded_verifier is not None else ()))},
        "full_table_readiness": False, "observations": observations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("glyph", type=Path)
    parser.add_argument("locator", type=Path)
    parser.add_argument("verifier", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bounded-verifier", type=Path)
    args = parser.parse_args()
    report = benchmark(args.collection, args.glyph, args.locator, args.verifier, bounded_verifier=args.bounded_verifier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
