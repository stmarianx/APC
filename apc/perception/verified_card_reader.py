"""Opt-in research reader with a learned binary proposal filter."""
import hashlib
from pathlib import Path

from PIL import Image

from apc.perception.card_verifier import ProposalVerifier
from apc.perception.infer_cards import CardReader


class VerifyingLocator:
    def __init__(self, locator, verifier):
        self.locator, self.verifier = locator, verifier

    def propose(self, path):
        result = self.locator.propose(path)
        with Image.open(path) as image:
            retained, rejected = self.verifier.filter(image, result["proposals"])
        return dict(result, proposals=retained, verifier_rejected=len(rejected))


class VerifiedCardReader(CardReader):
    def __init__(self, checkpoint, *, verifier_checkpoint, **options):
        if options.get("locator_checkpoint") is None:
            raise ValueError("Proposal verifier requires explicit neural locator")
        super().__init__(checkpoint, **options)
        verifier = ProposalVerifier(verifier_checkpoint)
        self.locator = VerifyingLocator(self.locator, verifier)
        self.verifier_metadata = {"checkpoint_sha256": hashlib.sha256(Path(verifier_checkpoint).read_bytes()).hexdigest(),
                                  "threshold": verifier.threshold, "confidence_calibrated": False,
                                  "research_only": True}

    def read(self, image_path):
        result = super().read(image_path)
        result["proposal_verifier"] = self.verifier_metadata
        result["limitations"].append("synthetic_only_proposal_verifier_not_validated_for_coaching")
        return result
