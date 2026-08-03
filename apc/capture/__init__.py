"""Explicit, read-only visible-table capture utilities for APC datasets."""

__all__ = ["CapturePlan", "CaptureRegion", "capture_frames"]


def __getattr__(name: str):
    if name in __all__:
        from . import screen_capture

        return getattr(screen_capture, name)
    raise AttributeError(name)
