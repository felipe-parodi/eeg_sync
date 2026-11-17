"""Tests for the ViTPose top-down teacher's pure helpers."""

from __future__ import annotations

import numpy as np

from pose_training import vitpose_label


def test_boxes_xyxy_to_xywh() -> None:
    xyxy = np.array([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 5.0, 8.0]])
    xywh = vitpose_label.boxes_xyxy_to_xywh(xyxy)
    assert xywh.tolist() == [[10.0, 20.0, 20.0, 40.0], [0.0, 0.0, 5.0, 8.0]]


def test_top_person_boxes_filters_and_ranks() -> None:
    # Distinct (non-overlapping) boxes so the ranking/cap is what is tested.
    boxes = np.array(
        [[0, 0, 20, 20], [100, 100, 140, 160], [300, 0, 330, 50]], dtype=float
    )
    confs = np.array([0.4, 0.9, 0.7])
    classes = np.array([0, 0, 0])
    kept_boxes, kept_confs = vitpose_label.top_person_boxes(
        boxes, confs, classes, person_class_id=0, max_persons=2
    )
    assert kept_confs.tolist() == [0.9, 0.7]
    assert kept_boxes[0].tolist() == [100, 100, 140, 160]


def test_top_person_boxes_drops_non_person_classes() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 20, 20]], dtype=float)
    confs = np.array([0.9, 0.8])
    classes = np.array([1, 0])  # first is not a person
    kept_boxes, kept_confs = vitpose_label.top_person_boxes(
        boxes, confs, classes, person_class_id=0, max_persons=4
    )
    assert kept_confs.tolist() == [0.8]
    assert kept_boxes[0].tolist() == [0, 0, 20, 20]


def test_build_person_record_shapes() -> None:
    bbox = np.array([10.0, 20.0, 30.0, 60.0])
    keypoints = np.tile([15.0, 25.0], (17, 1))
    scores = np.full(17, 0.8)
    record = vitpose_label.build_person_record(bbox, 0.9, keypoints, scores)
    assert record["bbox_xyxy"] == [10.0, 20.0, 30.0, 60.0]
    assert record["box_conf"] == 0.9
    assert len(record["keypoints"]) == 17
    assert record["keypoints"][0] == [15.0, 25.0, 0.8]
