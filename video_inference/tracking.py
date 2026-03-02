"""Two-person temporal ID assignment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox.astype(float)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a.astype(float)
    bx1, by1, bx2, by2 = b.astype(float)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


@dataclass
class AssignedDetection:
    """Detection enriched with stable track metadata."""

    track_id: int
    track_label: str
    bbox: np.ndarray
    confidence: float
    detection_index: int


@dataclass
class TwoPersonTrackerState:
    """Track last known boxes for parent/child."""

    parent_bbox: Optional[np.ndarray] = None
    child_bbox: Optional[np.ndarray] = None


def _assign_by_area(detections: List[Dict]) -> List[AssignedDetection]:
    sorted_indices = sorted(
        range(len(detections)),
        key=lambda idx: _bbox_area(np.asarray(detections[idx]["bbox"], dtype=float)),
        reverse=True,
    )
    parent_idx, child_idx = sorted_indices[:2]
    parent_det = detections[parent_idx]
    child_det = detections[child_idx]
    return [
        AssignedDetection(
            track_id=0,
            track_label="parent",
            bbox=np.asarray(parent_det["bbox"], dtype=float),
            confidence=float(parent_det.get("confidence", 1.0)),
            detection_index=parent_idx,
        ),
        AssignedDetection(
            track_id=1,
            track_label="child",
            bbox=np.asarray(child_det["bbox"], dtype=float),
            confidence=float(child_det.get("confidence", 1.0)),
            detection_index=child_idx,
        ),
    ]


def assign_two_person_tracks(
    detections: List[Dict],
    state: Optional[TwoPersonTrackerState] = None,
    min_iou_for_temporal: float = 0.05,
) -> Tuple[List[AssignedDetection], TwoPersonTrackerState]:
    """
    Assign stable parent/child identities to exactly two detections.

    Detection entries should include:
    - bbox: array-like [x1, y1, x2, y2]
    - confidence: optional float
    """
    if len(detections) != 2:
        raise ValueError(f"Expected exactly 2 detections, got {len(detections)}")

    if state is None:
        state = TwoPersonTrackerState()

    for det in detections:
        bbox = np.asarray(det.get("bbox"), dtype=float)
        if bbox.shape != (4,):
            raise ValueError("Each detection bbox must be a 4-element xyxy array")

    # First frame (or reset): use area prior (parent larger).
    if state.parent_bbox is None or state.child_bbox is None:
        assigned = _assign_by_area(detections)
        new_state = TwoPersonTrackerState(
            parent_bbox=assigned[0].bbox.copy(),
            child_bbox=assigned[1].bbox.copy(),
        )
        return assigned, new_state

    d0 = np.asarray(detections[0]["bbox"], dtype=float)
    d1 = np.asarray(detections[1]["bbox"], dtype=float)
    iou_same = bbox_iou(d0, state.parent_bbox) + bbox_iou(d1, state.child_bbox)
    iou_swap = bbox_iou(d1, state.parent_bbox) + bbox_iou(d0, state.child_bbox)

    if max(iou_same, iou_swap) < min_iou_for_temporal:
        # Ambiguous frame; fall back to area-based prior.
        assigned = _assign_by_area(detections)
    else:
        if iou_same >= iou_swap:
            parent_idx, child_idx = 0, 1
        else:
            parent_idx, child_idx = 1, 0
        parent_det = detections[parent_idx]
        child_det = detections[child_idx]
        assigned = [
            AssignedDetection(
                track_id=0,
                track_label="parent",
                bbox=np.asarray(parent_det["bbox"], dtype=float),
                confidence=float(parent_det.get("confidence", 1.0)),
                detection_index=parent_idx,
            ),
            AssignedDetection(
                track_id=1,
                track_label="child",
                bbox=np.asarray(child_det["bbox"], dtype=float),
                confidence=float(child_det.get("confidence", 1.0)),
                detection_index=child_idx,
            ),
        ]

    new_state = TwoPersonTrackerState(
        parent_bbox=assigned[0].bbox.copy(),
        child_bbox=assigned[1].bbox.copy(),
    )
    return assigned, new_state
