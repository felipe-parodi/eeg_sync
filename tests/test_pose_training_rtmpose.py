"""Tests for the RTMPose (rtmlib) teacher's pure helpers."""

from __future__ import annotations

import numpy as np

from pose_training import rtmpose_label


def test_bbox_from_keypoints_uses_confident_points() -> None:
    kpts = np.array([[10.0, 20.0], [30.0, 60.0], [1000.0, 1000.0]])
    scores = np.array([0.9, 0.9, 0.0])  # third is unconfident -> ignored
    bbox = rtmpose_label.bbox_from_keypoints(
        kpts, scores, conf_floor=0.3, expansion=1.0
    )
    assert bbox == [10.0, 20.0, 30.0, 60.0]


def test_build_person_record_shapes() -> None:
    kpts = np.tile([15.0, 25.0], (17, 1))
    scores = np.full(17, 0.8)
    rec = rtmpose_label.build_person_record(kpts, scores, conf_floor=0.3, expansion=1.0)
    assert len(rec["keypoints"]) == 17
    assert rec["keypoints"][0] == [15.0, 25.0, 0.8]
    assert abs(rec["box_conf"] - 0.8) < 1e-6


def test_top_persons_by_score_caps_and_ranks() -> None:
    scores = np.stack([np.full(17, 0.4), np.full(17, 0.9), np.full(17, 0.7)])
    order = rtmpose_label.top_persons_by_score(scores, max_persons=2)
    assert order == [1, 2]  # highest mean score first


def test_top_persons_handles_empty() -> None:
    assert rtmpose_label.top_persons_by_score(np.zeros((0, 17)), max_persons=4) == []
