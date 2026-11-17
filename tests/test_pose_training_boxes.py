"""Tests for class-agnostic detection-box cleanup (IoU + containment NMS)."""

from __future__ import annotations

import numpy as np

from pose_training.boxes import box_iou_ioa, clean_detection_boxes


def test_iou_and_ioa() -> None:
    iou, ioa = box_iou_ioa([0, 0, 10, 10], [0, 0, 20, 20])
    assert abs(iou - 0.25) < 1e-6  # 100 / 400
    assert abs(ioa - 1.0) < 1e-6  # the small box is fully inside the big one


def test_clean_suppresses_high_iou_duplicate() -> None:
    person_boxes = np.array([[0, 0, 100, 200], [2, 2, 102, 202]], dtype=float)
    confs = np.array([0.9, 0.8])
    kept = clean_detection_boxes(person_boxes, confs, iou_thr=0.6, ioa_thr=0.7)
    assert kept == [0]


def test_clean_suppresses_contained_partial() -> None:
    # A small low-confidence box inside a big high-confidence one with only
    # moderate IoU — IoU-NMS misses it, containment catches it.
    person_boxes = np.array([[0, 0, 100, 300], [20, 40, 70, 130]], dtype=float)
    confs = np.array([0.9, 0.4])
    iou, ioa = box_iou_ioa(person_boxes[1], person_boxes[0])
    assert iou < 0.6 and ioa > 0.9
    kept = clean_detection_boxes(person_boxes, confs, iou_thr=0.6, ioa_thr=0.7)
    assert kept == [0]


def test_clean_keeps_distinct_and_overlapping_people() -> None:
    # Two real people who overlap a bit (parent + child) must both survive.
    person_boxes = np.array([[0, 0, 100, 300], [60, 50, 140, 320]], dtype=float)
    confs = np.array([0.9, 0.85])
    kept = clean_detection_boxes(person_boxes, confs, iou_thr=0.6, ioa_thr=0.7)
    assert sorted(kept) == [0, 1]


def test_clean_returns_confidence_descending() -> None:
    person_boxes = np.array([[0, 0, 10, 10], [200, 200, 260, 320], [500, 0, 560, 120]])
    confs = np.array([0.3, 0.95, 0.6])
    kept = clean_detection_boxes(person_boxes, confs, iou_thr=0.6, ioa_thr=0.7)
    assert kept == [1, 2, 0]
