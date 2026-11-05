"""Matched warm saved-frame candidate timing, not full coaching latency."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from apc.perception.bright_surface_candidate import BrightSurfaceReader
from apc.perception.evaluate_browser_collection import load_collection
from apc.perception.verified_card_reader import VerifiedCardReader


def benchmark_readers(collection,readers,repeats=3):
    if type(repeats) is not int or repeats < 1 or len(readers) < 2:
        raise ValueError("Expected positive repeats and at least two readers")
    _,frames = load_collection(collection)
    root = Path(collection).resolve().parent
    frame_keys = [frame["image"].resolve().relative_to(root).as_posix() for frame in frames]
    if len(set(frame_keys)) != len(frame_keys):
        raise ValueError("Duplicate frame paths cannot be benchmarked as distinct captures")
    names = list(readers)
    for reader in readers.values():
        reader.read(frames[0]["image"])
    observations, fingerprints = [],{}
    for repeat in range(repeats):
        for frame_index,frame in enumerate(frames):
            shift = (repeat+frame_index)%len(names)
            order = names[shift:]+names[:shift]
            for name in order:
                start = time.perf_counter()
                result = readers[name].read(frame["image"])
                elapsed = (time.perf_counter()-start)*1000
                digest = hashlib.sha256(json.dumps(result["cards"],sort_keys=True,allow_nan=False).encode()).hexdigest()
                key = (name,frame_keys[frame_index])
                fingerprints.setdefault(key,set()).add(digest)
                observations.append({"repeat":repeat,"image":frame["image"].name,"frame_key":frame_keys[frame_index],"mode":name,
                    "elapsed_ms":elapsed,"proposal_count":len(result["cards"]),"cards_sha256":digest})
    summary = {}
    for name in names:
        values = [row["elapsed_ms"] for row in observations if row["mode"] == name]
        summary[name] = {"samples":len(values),"p50_ms":float(np.percentile(values,50)),
                         "p95_ms":float(np.percentile(values,95)),"max_ms":max(values)}
    return {"schema_version":"apc_matched_reader_timing_v1","summary":summary,"observations":observations,
        "repeats":repeats,"prediction_repeatability_passed":all(len(v)==1 for v in fingerprints.values()),
        "cross_variant_predictions_equal":all(len({row["cards_sha256"] for row in observations if row["frame_key"] == key}) == 1 for key in frame_keys),
        "collection_sha256":hashlib.sha256(Path(collection).read_bytes()).hexdigest(),
        "warmup":"One untimed first-frame read per reader; rotating variant order across frames/repeats",
        "scope":"Warm saved-frame I/O, proposals, verification and glyphs; excludes loading/capture/state/strategy/UI",
        "limitations":"Small reused frame set on shared laptop; not an end-to-end latency or readiness gate",
        "reference_reuse":True,"full_table_readiness":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection",type=Path)
    parser.add_argument("--new-verifier",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--compare-fast",action="store_true",help="Compare identical v5 stacks with only recovery component extraction changed")
    comparison.add_argument("--compare-full-fast",action="store_true",help="Compare original, recovery-only and full refinement acceleration")
    args = parser.parse_args()
    p = Path(__file__).resolve().parents[1]/"checkpoints"
    paths = {"glyph":p/"glyph-font-diversity-v1.pt","locator":p/"card-locator-hardneg-v1.pt",
             "incumbent_verifier":p/"card-face-verifier-appearance-v4.pt","bounded":p/"card-face-verifier-v1.pt",
             "new_verifier":args.new_verifier}
    options = dict(corner_pairs=True,surface_pairs=True,locator_checkpoint=paths["locator"],
        refine_surfaces=True,adaptive_surfaces=True,deduplicate_surfaces=True,tiled_views=True,
        bounded_verifier_checkpoint=paths["bounded"])
    torch.set_num_threads(2)
    readers = {
        "incumbent_routed":VerifiedCardReader(paths["glyph"],verifier_checkpoint=paths["incumbent_verifier"],**options),
        "wide_v2_v4":BrightSurfaceReader(paths["glyph"],wide_search=True,verifier_checkpoint=paths["incumbent_verifier"],**options),
        "wide_v2_v5":BrightSurfaceReader(paths["glyph"],wide_search=True,verifier_checkpoint=paths["new_verifier"],**options)}
    if args.compare_fast:
        readers = {"wide_v2_v5":readers["wide_v2_v5"],
            "wide_v2_v5_fast_recovery":BrightSurfaceReader(paths["glyph"],wide_search=True,fast_components=True,
                verifier_checkpoint=paths["new_verifier"],**options)}
    if args.compare_full_fast:
        readers = {"wide_v2_v5":readers["wide_v2_v5"],
            "fast_recovery":BrightSurfaceReader(paths["glyph"],wide_search=True,fast_components=True,
                verifier_checkpoint=paths["new_verifier"],**options),
            "fast_recovery_and_refinement":BrightSurfaceReader(paths["glyph"],wide_search=True,
                fast_components=True,fast_refinement=True,verifier_checkpoint=paths["new_verifier"],**options)}
    report = benchmark_readers(args.collection,readers)
    report.update(torch_threads=2,torch_version=str(torch.__version__),
        checkpoint_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()})
    report["fast_recovery_comparison"] = args.compare_fast
    report["fast_refinement_comparison"] = args.compare_full_fast
    report["candidate_source_sha256"] = {name:hashlib.sha256((p.parent/"perception"/name).read_bytes()).hexdigest()
        for name in ("bright_surface_candidate.py","fast_components.py","fast_refinement.py")}
    with args.output.open("x",encoding="utf-8") as output:
        output.write(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["summary"],indent=2))
