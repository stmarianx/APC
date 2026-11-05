"""Evaluate a planned, new capture session with fingerprint-locked models."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from apc.perception.evaluate_browser_collection import evaluate, load_collection
from apc.perception.infer_cards import CardReader


ROOT = Path(__file__).resolve().parents[2]


def run_audit(plan_path, collection_path, glyph_checkpoint, locator_checkpoint,
              previous_collection=ROOT / "apc/data/development/browser-domain-bridge-v1/collection.json",
              additional_previous_collections=(), *, verifier_checkpoint=None, bounded_verifier_checkpoint=None):
    plan_raw = Path(plan_path).read_bytes()
    plan = json.loads(plan_raw)
    if plan.get("training_prohibited") is not True or plan.get("weights_frozen_before_collection") is not True:
        raise ValueError("Expected a frozen-model audit plan with training prohibited")
    for checkpoint, key in ((glyph_checkpoint, "recognizer_sha256"), (locator_checkpoint, "detector_sha256")):
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != plan[key]:
            raise ValueError("Checkpoint does not match the frozen audit plan")
    options = plan.get("inference_options", {})
    allowed = {"adaptive_surfaces", "deduplicate_surfaces", "tiled_views"}
    if not isinstance(options, dict) or set(options)-allowed or any(type(v) is not bool for v in options.values()):
        raise ValueError("Unsupported frozen inference options")
    sources = plan.get("inference_source_sha256", {})
    verifier_options = {}
    if bounded_verifier_checkpoint is not None and verifier_checkpoint is None:
        raise ValueError("Bounded verifier requires a primary verifier")
    for checkpoint, key, option in ((verifier_checkpoint, "verifier_sha256", "verifier_checkpoint"),
                                    (bounded_verifier_checkpoint, "bounded_verifier_sha256", "bounded_verifier_checkpoint")):
        if (checkpoint is None) != (key not in plan):
            raise ValueError("Verifier configuration differs from frozen plan")
        if checkpoint is not None:
            if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != plan[key]:
                raise ValueError("Verifier checkpoint differs from frozen plan")
            verifier_options[option] = checkpoint
    if bounded_verifier_checkpoint is not None and not options.get("adaptive_surfaces"):
        raise ValueError("Routed audit requires adaptive surfaces")
    required_sources = {"apc/perception/" + name for name in ("infer_cards.py", "infer_card_locator.py", "card_proposals.py", "card_locator.py", "card_glyphs.py")}
    if verifier_options:
        required_sources.update({"apc/perception/card_verifier.py", "apc/perception/verified_card_reader.py"})
    if not isinstance(sources, dict) or ((options or verifier_options) and not required_sources.issubset(sources)):
        raise ValueError("Explicit inference options require frozen source fingerprints")
    for relative, expected in sources.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Inference source does not match frozen audit plan")
    data, frames = load_collection(collection_path)
    if data.get("use_policy") != "sealed_audit_no_training":
        raise ValueError("Audit collection must prohibit training")
    schedule = [[row["hand_group"], row["street"]] for row in data["frames"]]
    if schedule != plan["frame_schedule"] or len(frames) > plan["maximum_frames"]:
        raise ValueError("Audit frame schedule incomplete or changed")
    if "frame_viewports" in plan and [frame["viewport"] for frame in frames] != plan["frame_viewports"]:
        raise ValueError("Audit viewport schedule changed")
    for previous_path in (previous_collection, *additional_previous_collections):
        previous, _ = load_collection(previous_path)
        if previous["capture_session_id"] == data["capture_session_id"]:
            raise ValueError("Audit must use a new capture session")
        previous_hashes = {row["sha256"] for row in previous["frames"]}
        if any(row["sha256"] in previous_hashes for row in data["frames"]):
            raise ValueError("Audit reuses development images")
    torch.set_num_threads(2)
    reader_type = CardReader
    if verifier_options:
        from apc.perception.verified_card_reader import VerifiedCardReader
        reader_type = VerifiedCardReader
    reader = reader_type(glyph_checkpoint, corner_pairs=True, surface_pairs=True,
                        locator_checkpoint=locator_checkpoint, refine_surfaces=True, **options, **verifier_options)
    report = evaluate(collection_path, reader)
    if verifier_options:
        report["frozen_verifier"] = reader.verifier_metadata
        report["audit_checks_verifier_hashes_match"] = True
    report.update(audit_id=plan["audit"], plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
        frozen_recognizer_sha256=plan["recognizer_sha256"], frozen_detector_sha256=plan["detector_sha256"],
        audit_checks={"frozen_model_hashes_match": True, "complete_planned_schedule": True,
                      "new_session": True, "images_disjoint_from_development": True},
        collection_use_policy=data["use_policy"],
        scope=plan.get("scope", "New hands/session in the same application and layout; correlated frames and draft geometry, not cross-platform readiness"))
    if options:
        report["frozen_inference_options"] = options
        report["frozen_inference_source_sha256"] = plan["inference_source_sha256"]
        report["audit_checks"]["inference_sources_match"] = True
    if "frame_viewports" in plan:
        report["audit_checks"]["planned_viewports_match"] = True
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    audit_version = parser.add_mutually_exclusive_group()
    audit_version.add_argument("--viewport-v3", action="store_true", help="Run the separately planned multi-viewport audit")
    audit_version.add_argument("--multiview-v4", action="store_true", help="Run the source/config-frozen multiview audit")
    args = parser.parse_args()
    name = "browser-multiview-audit-v4" if args.multiview_v4 else ("browser-viewport-audit-v3" if args.viewport_v3 else "browser-capture-audit-v2")
    report = run_audit(ROOT / f"apc/perception/baselines/{name}.plan.json",
        ROOT / f"apc/data/reference/{name}/collection.json",
        ROOT / ("apc/checkpoints/glyph-font-diversity-v1.pt" if args.viewport_v3 or args.multiview_v4 else "apc/checkpoints/browser-symbol-rehearsal-v1.pt"),
        ROOT / "apc/checkpoints/card-locator-hardneg-v1.pt",
        additional_previous_collections=((ROOT / "apc/data/reference/browser-capture-audit-v2/collection.json", ROOT / "apc/data/reference/browser-viewport-audit-v3/collection.json") if args.multiview_v4 else ((ROOT / "apc/data/reference/browser-capture-audit-v2/collection.json",) if args.viewport_v3 else ())))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
