"""Opt-in research reader with a learned binary proposal filter."""
import hashlib
from pathlib import Path

from PIL import Image

from apc.perception.card_verifier import ProposalVerifier
from apc.perception.infer_cards import CardReader


class SurfaceRoutedVerifier:
    """Route by bounded pixel evidence only; neither expert sees identity labels."""
    def __init__(self, ambiguous, bounded):
        self.ambiguous, self.bounded = ambiguous, bounded

    def filter(self, image, proposals):
        retained, rejected = [], []
        for route, verifier in (("bounded_surface", self.bounded), ("ambiguous_surface", self.ambiguous)):
            selected = [dict(p, _route_index=i) for i, p in enumerate(proposals)
                        if (p.get("refinement") == "adaptive_bounded_surface") == (route == "bounded_surface")]
            keep, drop = verifier.filter(image, selected)
            retained.extend(dict(p, verifier_route=route) for p in keep)
            rejected.extend(dict(p, verifier_route=route) for p in drop)
        for rows in (retained, rejected):
            rows.sort(key=lambda item: item["_route_index"])
            for item in rows:
                item.pop("_route_index")
        return retained, rejected


class VerifyingLocator:
    def __init__(self, locator, verifier):
        self.locator, self.verifier = locator, verifier

    def propose(self, path):
        result = self.locator.propose(path)
        with Image.open(path) as image:
            retained, rejected = self.verifier.filter(image, result["proposals"])
        return dict(result, proposals=retained, verifier_rejected=len(rejected))


class VerifiedCardReader(CardReader):
    def __init__(self, checkpoint, *, verifier_checkpoint, bounded_verifier_checkpoint=None, **options):
        if options.get("locator_checkpoint") is None:
            raise ValueError("Proposal verifier requires explicit neural locator")
        if bounded_verifier_checkpoint is not None and not options.get("adaptive_surfaces"):
            raise ValueError("Surface routing requires adaptive surface evidence")
        super().__init__(checkpoint, **options)
        verifier = ProposalVerifier(verifier_checkpoint)
        self.verifier_metadata = {"checkpoint_sha256": hashlib.sha256(Path(verifier_checkpoint).read_bytes()).hexdigest(),
                                  "threshold": verifier.threshold, "confidence_calibrated": False,
                                  "research_only": True}
        if bounded_verifier_checkpoint is not None:
            bounded = ProposalVerifier(bounded_verifier_checkpoint)
            self.verifier_metadata.update(route="bounded_surface_v1_else_ambiguous_v4",
                bounded_checkpoint_sha256=hashlib.sha256(Path(bounded_verifier_checkpoint).read_bytes()).hexdigest(),
                bounded_threshold=bounded.threshold)
            verifier = SurfaceRoutedVerifier(verifier, bounded)
        self.locator = VerifyingLocator(self.locator, verifier)

    def read(self, image_path):
        result = super().read(image_path)
        result["proposal_verifier"] = self.verifier_metadata
        result["limitations"].append("synthetic_only_proposal_verifier_not_validated_for_coaching")
        return result
