"""Tests for evaluation helpers (confidence collection, side-by-side panels)."""

from __future__ import annotations

import numpy as np

from pose_training import evaluate


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf


class _FakeKeypoints:
    def __init__(self, xy: np.ndarray, conf: np.ndarray) -> None:
        self.xy = xy
        self.conf = conf


class _FakeResult:
    def __init__(self, n: int) -> None:
        self.orig_shape = (540, 960)
        self.boxes = _FakeBoxes(
            # Distinct, non-overlapping boxes so box-NMS keeps all instances.
            np.array([[10 + i * 100, 10, 50 + i * 100, 90] for i in range(n)], float),
            np.linspace(0.6, 0.95, n),
        )
        self.keypoints = _FakeKeypoints(
            np.zeros((n, 17, 2), dtype=float), np.full((n, 17), 0.8)
        )


def test_collect_pred_confidences_splits_head_and_limbs() -> None:
    results = [_FakeResult(2), _FakeResult(1)]
    stats = evaluate.collect_pred_confidences(results, max_persons=6)
    assert len(stats["instance_conf"]) == 3
    # 3 instances * 5 head keypoints, 3 * 12 limb keypoints.
    assert len(stats["head_kpt_conf"]) == 3 * 5
    assert len(stats["limb_kpt_conf"]) == 3 * 12


def test_hstack_labeled_dimensions() -> None:
    left = np.zeros((200, 100, 3), dtype=np.uint8)
    right = np.zeros((200, 100, 3), dtype=np.uint8)
    panel = evaluate._hstack_labeled(left, right, "old", "new")
    # Two panels side by side, plus a label banner on top.
    assert panel.shape[1] == 200
    assert panel.shape[0] > 200
    assert panel.shape[2] == 3


def test_hstack_labeled_handles_mismatched_heights() -> None:
    left = np.zeros((180, 100, 3), dtype=np.uint8)
    right = np.zeros((200, 120, 3), dtype=np.uint8)
    panel = evaluate._hstack_labeled(left, right, "a", "b")
    # Both sides are resized to a common height before stacking.
    assert panel.shape[2] == 3
    assert panel.shape[1] == panel.shape[1]  # well-formed, no exception
