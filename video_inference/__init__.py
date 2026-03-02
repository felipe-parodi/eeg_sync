"""Video inference helpers (schema, device routing, tracking)."""

from .schema import ValidationResult, validate_session_output
from .device import resolve_device, resolve_inference_mode
from .tracking import (
    AssignedDetection,
    TwoPersonTrackerState,
    assign_two_person_tracks,
    bbox_iou,
)
__all__ = [
    "ValidationResult",
    "validate_session_output",
    "resolve_device",
    "resolve_inference_mode",
    "bbox_iou",
    "AssignedDetection",
    "TwoPersonTrackerState",
    "assign_two_person_tracks",
]
