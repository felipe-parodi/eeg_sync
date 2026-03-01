import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_inference.ultralytics_runner import (  # noqa: E402
    _assign_from_role_indices,
    _resolve_role_indices_from_tracker_ids,
)


def _fake_detections():
    return [
        {"bbox": np.asarray([10, 20, 60, 100], dtype=float), "confidence": 0.9},
        {"bbox": np.asarray([100, 30, 220, 260], dtype=float), "confidence": 0.95},
    ]


def test_resolve_role_indices_initializes_with_area_prior():
    detections = _fake_detections()
    role_map = {}

    parent_idx, child_idx = _resolve_role_indices_from_tracker_ids(
        detections=detections,
        tracker_ids=[11, 22],
        role_by_tracker_id=role_map,
    )

    assert parent_idx == 1
    assert child_idx == 0
    assert role_map[22] == 0
    assert role_map[11] == 1


def test_resolve_role_indices_uses_existing_tracker_role_map():
    detections = _fake_detections()
    role_map = {11: 1, 22: 0}

    parent_idx, child_idx = _resolve_role_indices_from_tracker_ids(
        detections=detections,
        tracker_ids=[11, 22],
        role_by_tracker_id=role_map,
    )

    assert parent_idx == 1
    assert child_idx == 0


def test_resolve_role_indices_handles_partial_known_ids():
    detections = _fake_detections()
    role_map = {99: 0}

    parent_idx, child_idx = _resolve_role_indices_from_tracker_ids(
        detections=detections,
        tracker_ids=[88, 99],
        role_by_tracker_id=role_map,
    )

    assert parent_idx == 1
    assert child_idx == 0
    assert role_map[99] == 0
    assert role_map[88] == 1


def test_assign_from_role_indices_preserves_detection_indices():
    detections = _fake_detections()
    assigned = _assign_from_role_indices(
        detections=detections,
        parent_idx=1,
        child_idx=0,
    )

    parent = assigned[0]
    child = assigned[1]
    assert parent.track_label == "parent"
    assert child.track_label == "child"
    assert parent.detection_index == 1
    assert child.detection_index == 0
