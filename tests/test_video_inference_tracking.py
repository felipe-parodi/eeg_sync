import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_inference.tracking import (  # noqa: E402
    TwoPersonTrackerState,
    assign_two_person_tracks,
    bbox_iou,
)


def test_bbox_iou_returns_expected_value():
    a = np.array([0, 0, 10, 10], dtype=float)
    b = np.array([5, 5, 15, 15], dtype=float)
    assert np.isclose(bbox_iou(a, b), 25 / 175)


def test_first_frame_assigns_parent_as_larger_bbox():
    detections = [
        {"bbox": [100, 100, 140, 180], "confidence": 0.9},  # smaller
        {"bbox": [20, 50, 120, 230], "confidence": 0.95},  # larger
    ]
    assigned, _ = assign_two_person_tracks(detections)

    parent = next(item for item in assigned if item.track_label == "parent")
    child = next(item for item in assigned if item.track_label == "child")

    assert parent.track_id == 0
    assert child.track_id == 1
    assert parent.detection_index == 1
    assert child.detection_index == 0


def test_temporal_assignment_prevents_swap_when_detection_order_flips():
    # Frame 0: parent left, child right
    frame0 = [
        {"bbox": [10, 10, 110, 220], "confidence": 0.95},  # parent-like
        {"bbox": [180, 40, 240, 170], "confidence": 0.92},  # child-like
    ]
    assigned0, state = assign_two_person_tracks(frame0, state=None)
    assert (
        next(item for item in assigned0 if item.track_label == "parent").detection_index
        == 0
    )

    # Frame 1: detector output order flips but spatial continuity is preserved.
    frame1 = [
        {"bbox": [182, 42, 242, 172], "confidence": 0.91},  # child first now
        {"bbox": [12, 11, 112, 221], "confidence": 0.96},  # parent second
    ]
    assigned1, _ = assign_two_person_tracks(frame1, state=state)
    parent = next(item for item in assigned1 if item.track_label == "parent")
    child = next(item for item in assigned1 if item.track_label == "child")

    assert parent.detection_index == 1
    assert child.detection_index == 0


def test_assignment_requires_exactly_two_detections():
    state = TwoPersonTrackerState()
    detections = [{"bbox": [0, 0, 10, 10], "confidence": 1.0}]
    try:
        assign_two_person_tracks(detections, state=state)
    except ValueError as error:
        assert "Expected exactly 2 detections" in str(error)
    else:
        raise AssertionError("Expected ValueError for invalid detection count")
