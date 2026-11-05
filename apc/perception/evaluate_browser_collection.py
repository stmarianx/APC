"""Diagnose captured-browser card failures; draft labels are not training inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from apc.perception.card_glyphs import glyph_tensor
from apc.perception.card_proposals import match_proposals
from apc.perception.infer_cards import CardReader
from apc.perception.region_model import RANKS, SUITS


def load_collection(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema_version") != "apc_browser_card_development_v1"
            or data.get("draft_only") is not True or data.get("readiness_credit") is not False
            or data.get("virtual_chips") is not True):
        raise ValueError("Expected draft virtual-chip browser development collection")
    width, height = data["viewport"]
    card_width, card_height = data["card_size"]
    if not all(math.isfinite(v) and v > 0 for v in (width, height, card_width, card_height)):
        raise ValueError("Invalid DOM dimensions")
    def box(x, y, w, h):
        values = [x / width, y / height, (x + w) / width, (y + h) / height]
        if not all(math.isfinite(v) for v in values) or not (0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1):
            raise ValueError("Box outside DOM viewport")
        return values
    frames = []
    for row in data["frames"]:
        width, height = row.get("viewport", data["viewport"])
        card_width, card_height = row.get("card_size", data["card_size"])
        if not all(math.isfinite(v) and v > 0 for v in (width, height, card_width, card_height)):
            raise ValueError("Invalid per-frame DOM dimensions")
        image = (path.parent / row["image"]).resolve()
        if not image.is_relative_to(path.parent) or image.suffix.lower() != ".png":
            raise ValueError("Invalid collection image path")
        if hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Collection image fingerprint mismatch")
        viewport = [width, height]
        mapping = row.get("screenshot_mapping", data.get("screenshot_mapping", "normalized_viewport"))
        if mapping == "uniform_width_scale_top_left_crop":
            # This explicit capture-provider mapping preserves scale when the
            # screenshot is cropped vertically, rather than stretching labels.
            with Image.open(image) as captured:
                height = captured.height * width / captured.width
        elif mapping != "normalized_viewport":
            raise ValueError("Unknown screenshot coordinate mapping")
        targets = []
        for role, identity, x, y in row["cards"]:
            if role not in {"hero", "board"} or len(identity) != 2 or identity[0] not in RANKS or identity[1] not in SUITS:
                raise ValueError("Invalid draft card label")
            regions = {}
            parent = box(x, y, card_width, card_height)
            for kind in ("rank", "suit"):
                dx, dy, w, h = row.get(kind + "_region_offset_xywh", data[kind + "_region_offset_xywh"])
                region = box(x + dx, y + dy, w, h)
                if not (parent[0] <= region[0] < region[2] <= parent[2] and parent[1] <= region[1] < region[3] <= parent[3]):
                    raise ValueError("Glyph region outside its card")
                regions[kind] = region
            targets.append({"role": role, "identity": identity, "box_xyxy": parent, "regions": regions})
        if len({t["identity"] for t in targets}) != len(targets):
            raise ValueError("Duplicate visible card identity")
        frames.append({"image": image, "hand_group": row["hand_group"], "targets": targets,
                       "viewport": viewport, "coordinate_extent": [width, height],
                       "screenshot_mapping": mapping, "card_size": [card_width, card_height]})
    if not frames:
        raise ValueError("Empty browser collection")
    return data, frames


def evaluate(collection, reader):
    data, frames = load_collection(collection)
    rows = []
    for frame in frames:
        targets = frame["targets"]
        result = reader.read(frame["image"])
        matched = match_proposals(result["cards"], targets)
        exact = sum(result["cards"][m["proposal_index"]]["candidate"] == targets[m["target_index"]]["identity"] for m in matched["matches"])
        # Oracle crops isolate recognizer/domain errors from detector/crop errors.
        with Image.open(frame["image"]) as image:
            tokens, present = [], []
            for target in targets:
                coords = [round(v * d) for v, d in zip(target["box_xyxy"], image.size * 2)]
                background = np.median(np.asarray(image.convert("RGB").crop(coords)).reshape(-1, 3), axis=0)
                parts = [glyph_tensor(image, target["regions"][kind], border_margin=False, background_rgb=background) for kind in ("rank", "suit")]
                tokens.append(torch.cat([part[0] for part in parts]))
                present.append(all(part[1] for part in parts))
        with torch.no_grad():
            predicted = reader.model(torch.stack(tokens))
        identities = [RANKS[r] + SUITS[s] for r, s in zip(predicted["rank"].argmax(1).tolist(), predicted["suit"].argmax(1).tolist())]
        rows.append({"image": frame["image"].name, "hand_group": frame["hand_group"],
            "viewport": frame["viewport"], "card_size": frame["card_size"],
            "screenshot_mapping": frame["screenshot_mapping"], "coordinate_extent": frame["coordinate_extent"],
            "targets": len(targets), "proposals": len(result["cards"]), "localized": len(matched["matches"]),
            "false_positives": len(result["cards"]) - len(matched["matches"]),
            "missed_cards": len(targets) - len(matched["matches"]),
            "end_to_end_exact": exact, "oracle_glyph_exact": sum(ok and value == target["identity"] for ok, value, target in zip(present, identities, targets)),
            "oracle_candidates": identities, "confirmed_cards": sum(c["card"] is not None for c in result["cards"])})
    totals = {key: sum(row[key] for row in rows) for key in ("targets", "proposals", "localized", "false_positives", "missed_cards", "end_to_end_exact", "oracle_glyph_exact", "confirmed_cards")}
    return {"development_diagnostic_only": True, "draft_geometry": True,
        "capture_sessions": 1, "capture_session_id": data["capture_session_id"],
        "hand_groups": len({row["hand_group"] for row in rows}), "frames": rows,
        "unique_visible_identities": len({target["identity"] for frame in frames for target in frame["targets"]}),
        "collection_sha256": hashlib.sha256(Path(collection).read_bytes()).hexdigest(),
        "totals": totals, "training_performed": False, "full_table_readiness": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--surface-pairs", action="store_true", help="Experimental full-face two-band extraction")
    checkpoints = Path(__file__).resolve().parents[1] / "checkpoints"
    parser.add_argument("--glyph-checkpoint", type=Path, default=checkpoints / "glyph-surface-pair-v1.pt")
    parser.add_argument("--locator-checkpoint", type=Path, default=checkpoints / "card-locator-mixed-v1.pt")
    parser.add_argument("--pixel-proposals", action="store_true", help="Diagnostic light-surface proposal control; no neural locator")
    parser.add_argument("--adaptive-surfaces", action="store_true", help="Experimental bounded surface expansion")
    parser.add_argument("--deduplicate-surfaces", action="store_true")
    parser.add_argument("--tiled-views", action="store_true")
    args = parser.parse_args()
    if args.pixel_proposals and args.adaptive_surfaces:
        parser.error("Adaptive surfaces require neural proposals")
    torch.set_num_threads(2)
    reader = CardReader(args.glyph_checkpoint, corner_pairs=True,
        locator_checkpoint=None if args.pixel_proposals else args.locator_checkpoint,
        refine_surfaces=True, surface_pairs=args.surface_pairs, adaptive_surfaces=args.adaptive_surfaces,
        deduplicate_surfaces=args.deduplicate_surfaces, tiled_views=args.tiled_views)
    report = evaluate(args.collection, reader)
    report["checkpoint_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in ([args.glyph_checkpoint] if args.pixel_proposals else [args.glyph_checkpoint, args.locator_checkpoint])}
    report["torch_version"] = str(torch.__version__)
    report["pair_extraction"] = "surface_two_band_then_corner" if args.surface_pairs else "upper_left_corner"
    report["proposal_mode"] = "light_surface_pixels" if args.pixel_proposals else "neural_locator_refined"
    report["surface_refinement"] = "none" if args.pixel_proposals else ("adaptive_bounded" if args.adaptive_surfaces else "legacy_fixed_roi")
    report["deduplicate_surfaces"] = args.deduplicate_surfaces
    report["tiled_views"] = args.tiled_views
    report["added_view_policy"] = "bounded_surfaces_only" if args.tiled_views else None
    if args.adaptive_surfaces:
        from apc.perception.card_proposals import ADAPTIVE_SURFACE_VERSION
        report["surface_refinement_version"] = ADAPTIVE_SURFACE_VERSION
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
