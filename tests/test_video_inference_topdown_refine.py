"""Tests for top-down keypoint refinement (video_inference.topdown_refine)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from video_inference.topdown_refine import fuse_person_keypoints, refine_pose_table


def _skeleton(x: float, y: float, h: float, conf: float = 0.9) -> np.ndarray:
    k = np.zeros((17, 3))
    k[:, 0] = x + np.linspace(0, h / 2, 17)
    k[:, 1] = y + np.linspace(0, h, 17)
    k[:, 2] = conf
    return k


def test_fuse_takes_refined_positions_when_same_person() -> None:
    student = _skeleton(100, 100, 200)
    refined = student.copy()
    refined[:, :2] += 2.0  # small correction
    fused, used = fuse_person_keypoints(student, refined, box_height=200)
    assert used
    np.testing.assert_allclose(fused[:, :2], refined[:, :2])
    # Confidence stays the student's (downstream thresholds are calibrated to it).
    np.testing.assert_allclose(fused[:, 2], student[:, 2])


def test_fuse_keeps_student_when_refiner_locked_onto_other_person() -> None:
    student = _skeleton(100, 100, 200)
    refined = _skeleton(400, 100, 200)  # a different person entirely
    fused, used = fuse_person_keypoints(student, refined, box_height=200)
    assert not used
    np.testing.assert_allclose(fused, student)


def test_fuse_keeps_student_keypoints_the_refiner_is_unsure_about() -> None:
    student = _skeleton(100, 100, 200)
    refined = student.copy()
    refined[:, :2] += 2.0
    refined[3, 2] = 0.1  # refiner unsure of one keypoint
    fused, used = fuse_person_keypoints(student, refined, box_height=200)
    assert used
    np.testing.assert_allclose(fused[3], student[3])
    np.testing.assert_allclose(fused[4, :2], refined[4, :2])


def test_refine_pose_table_replaces_positions_per_person() -> None:
    tracks = pd.DataFrame(
        {
            "frame_idx": [0, 0],
            "timestamp_s": [1.0, 1.0],
            "track_id": [0, 1],
            "track_label": ["parent", "child"],
            "bbox_x1": [100.0, 400.0],
            "bbox_y1": [100.0, 100.0],
            "bbox_x2": [200.0, 500.0],
            "bbox_y2": [300.0, 300.0],
            "track_confidence": [0.9, 0.9],
        }
    )
    rows = []
    for tid, x in ((0, 100.0), (1, 400.0)):
        k = _skeleton(x, 100, 200)
        for i in range(17):
            rows.append(
                [
                    0,
                    1.0,
                    tid,
                    "parent" if tid == 0 else "child",
                    f"kp_{i:03d}",
                    k[i, 0],
                    k[i, 1],
                    0.0,
                    k[i, 2],
                ]
            )
    pose = pd.DataFrame(
        rows,
        columns=[
            "frame_idx",
            "timestamp_s",
            "track_id",
            "track_label",
            "keypoint_name",
            "x_m",
            "y_m",
            "z_m",
            "keypoint_confidence",
        ],
    )

    def fake_refiner(batch: list) -> list[list[np.ndarray]]:
        out = []
        for _frame_idx, boxes in batch:
            people = []
            for b in boxes:
                k = _skeleton(b[0], 100, 200)
                k[:, 0] += 1.0
                people.append(k)
            out.append(people)
        return out

    refined, summary = refine_pose_table(tracks, pose, fake_refiner)
    assert summary["persons_refined"] == 2
    parent = refined[(refined.track_id == 0)].sort_values("keypoint_name")
    np.testing.assert_allclose(
        parent["x_m"].to_numpy(), _skeleton(100, 100, 200)[:, 0] + 1.0
    )
    assert list(refined.columns) == list(pose.columns)
