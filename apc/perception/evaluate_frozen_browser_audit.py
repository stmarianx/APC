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
              previous_collection=ROOT / "apc/data/development/browser-domain-bridge-v1/collection.json"):
    plan_raw = Path(plan_path).read_bytes()
    plan = json.loads(plan_raw)
    if plan.get("training_prohibited") is not True or plan.get("weights_frozen_before_collection") is not True:
        raise ValueError("Expected a frozen-model audit plan with training prohibited")
    for checkpoint, key in ((glyph_checkpoint, "recognizer_sha256"), (locator_checkpoint, "detector_sha256")):
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != plan[key]:
            raise ValueError("Checkpoint does not match the frozen audit plan")
    data, frames = load_collection(collection_path)
    if data.get("use_policy") != "sealed_audit_no_training":
        raise ValueError("Audit collection must prohibit training")
    schedule = [[row["hand_group"], row["street"]] for row in data["frames"]]
    if schedule != plan["frame_schedule"] or len(frames) > plan["maximum_frames"]:
        raise ValueError("Audit frame schedule incomplete or changed")
    previous, _ = load_collection(previous_collection)
    if previous["capture_session_id"] == data["capture_session_id"]:
        raise ValueError("Audit must use a new capture session")
    previous_hashes = {row["sha256"] for row in previous["frames"]}
    if any(row["sha256"] in previous_hashes for row in data["frames"]):
        raise ValueError("Audit reuses development images")
    torch.set_num_threads(2)
    reader = CardReader(glyph_checkpoint, corner_pairs=True, surface_pairs=True,
                        locator_checkpoint=locator_checkpoint, refine_surfaces=True)
    report = evaluate(collection_path, reader)
    report.update(audit_id=plan["audit"], plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
        frozen_recognizer_sha256=plan["recognizer_sha256"], frozen_detector_sha256=plan["detector_sha256"],
        audit_checks={"frozen_model_hashes_match": True, "complete_planned_schedule": True,
                      "new_session": True, "images_disjoint_from_development": True},
        collection_use_policy=data["use_policy"],
        scope="New hands/session in the same application and layout; correlated frames and draft geometry, not cross-platform readiness")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(ROOT / "apc/perception/baselines/browser-capture-audit-v2.plan.json",
        ROOT / "apc/data/reference/browser-capture-audit-v2/collection.json",
        ROOT / "apc/checkpoints/browser-symbol-rehearsal-v1.pt",
        ROOT / "apc/checkpoints/card-locator-hardneg-v1.pt")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
