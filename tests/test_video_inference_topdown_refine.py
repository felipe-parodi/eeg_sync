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


def test_fast_heatmap_decoding_matches_huggingface() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.vitpose.image_processing_vitpose import (
        VitPoseImageProcessor,
        box_to_center_and_scale,
    )

    from video_inference.topdown_refine import decode_heatmaps

    del transformers
    rng = np.random.default_rng(0)
    n, k, h, w = 3, 17, 64, 48
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.zeros((n, k, h, w), np.float32)
    for i in range(n):
        for j in range(k):
            cy, cx = rng.uniform(3, h - 4), rng.uniform(3, w - 4)
            heat[i, j] = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0**2))
            heat[i, j] *= rng.uniform(0.3, 1.0)
    boxes_xywh = np.array(
        [[10, 20, 80, 200], [300, 50, 120, 150], [5, 5, 40, 60]], float
    )
    centers, scales = zip(
        *[
            box_to_center_and_scale(b, image_width=192, image_height=256)
            for b in boxes_xywh
        ]
    )
    ref_xy, ref_score = VitPoseImageProcessor().keypoints_from_heatmaps(
        heat.copy(), np.asarray(centers), np.asarray(scales), kernel=11
    )
    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
    fast = decode_heatmaps(torch.from_numpy(heat), boxes_xyxy)
    np.testing.assert_allclose(fast[..., :2], ref_xy, atol=1e-2)
    np.testing.assert_allclose(fast[..., 2], ref_score[..., 0], atol=1e-5)


def test_fast_crop_matches_huggingface_preprocessing() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.vitpose.image_processing_vitpose import (
        VitPoseImageProcessor,
    )

    from video_inference.topdown_refine import crop_people, normalize_crops

    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    image = np.ascontiguousarray(
        np.repeat(np.repeat(image[::8, ::8], 8, axis=0), 8, axis=1)[:540, :960]
    )  # smooth-ish image so interpolation conventions matter little
    boxes_xyxy = np.array([[100.0, 50.0, 220.0, 400.0], [600.0, 300.0, 700.0, 520.0]])
    xywh = boxes_xyxy.copy()
    xywh[:, 2:] -= xywh[:, :2]
    ref = VitPoseImageProcessor()(image, boxes=[xywh.tolist()], return_tensors="pt")
    fast = normalize_crops(
        torch.from_numpy(crop_people(image, boxes_xyxy)), "cpu", torch.float32
    )
    assert fast.shape == ref["pixel_values"].shape
    diff = (fast - ref["pixel_values"]).abs()
    assert float(diff.mean()) < 0.02
