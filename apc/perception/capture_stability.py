"""Validate collection-time geometry evidence, independently of model output.

Collectors must observe visible DOM geometry immediately before and after the
screenshot. This detects movement, not all possible transient rendering races;
pixel-level annotation review remains necessary.
"""
import math


def validate_capture_stability(evidence, *, image_size, image_sha256, viewport, cards):
    """Fail closed on absent/malformed evidence or movement. Return no readiness claim."""
    if not isinstance(evidence, dict) or evidence.get("schema_version") != "apc_capture_stability_v1":
        raise ValueError("Invalid capture stability evidence")
    if evidence.get("image_sha256") != image_sha256 or evidence.get("image_size") != list(image_size):
        raise ValueError("Capture stability image binding mismatch")

    def numbers(value, length):
        return (isinstance(value, list) and len(value) == length
                and all(type(v) in (int, float) and math.isfinite(v) for v in value))

    snapshots = []
    for name in ("before", "after"):
        snapshot = evidence.get(name)
        if not isinstance(snapshot, dict):
            raise ValueError("Missing capture stability snapshot")
        if (not numbers(snapshot.get("viewport"), 2) or snapshot["viewport"] != list(viewport)
                or not numbers(snapshot.get("scroll"), 2)):
            raise ValueError("Invalid capture stability viewport/scroll")
        observed = snapshot.get("cards")
        if not isinstance(observed, list) or len(observed) != len(cards):
            raise ValueError("Capture stability card count mismatch")
        for actual, label in zip(observed, cards):
            if (not isinstance(actual, list) or len(actual) != 6
                    or actual[:2] != label[:2] or not numbers(actual[2:], 4)
                    or actual[4] <= 0 or actual[5] <= 0
                    or actual[2:4] != label[2:4]):
                raise ValueError("Capture stability label geometry mismatch")
        snapshots.append(snapshot)
    # Exact equality is deliberate: unstable frames are retained as rejected
    # evidence, never silently relabeled or selected using model predictions.
    if snapshots[0] != snapshots[1]:
        raise ValueError("Capture moved during screenshot")
