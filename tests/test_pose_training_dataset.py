"""Tests for pseudo-label filtering and YOLO pose dataset export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from pose_training import coco, dataset


def _person(box_conf: float, kpt_conf: float) -> dict:
    return {
        "bbox_xyxy": [10.0, 20.0, 30.0, 60.0],
        "box_conf": box_conf,
        "keypoints": [[15.0 + i, 25.0 + i, kpt_conf] for i in range(17)],
    }


def _frame(persons: list, split: str = "train") -> dict:
    return {
        "frame_path": "f.jpg",
        "participant": "P001",
        "angle": "SHORT",
        "split": split,
        "source_video": "P001_FULL_SHORT_compressed.mp4",
        "source_timestamp_s": 1.0,
        "width": 100,
        "height": 200,
        "persons": persons,
    }


def test_filter_drops_low_instance_confidence() -> None:
    cfg = dataset.FilterConfig(instance_conf_thresh=0.5)
    frame = _frame([_person(0.4, 0.9)])
    filtered = dataset.filter_frame(frame, cfg)
    assert filtered["persons"] == []


def test_filter_demotes_low_confidence_keypoints_to_unlabeled() -> None:
    cfg = dataset.FilterConfig(instance_conf_thresh=0.5, kpt_conf_thresh=0.5)
    frame = _frame([_person(0.9, 0.2)])  # all keypoints below kpt threshold
    filtered = dataset.filter_frame(frame, cfg)
    # Too few visible keypoints -> whole instance dropped.
    assert filtered["persons"] == []


def test_filter_keeps_confident_instance_and_marks_visibility() -> None:
    cfg = dataset.FilterConfig(
        instance_conf_thresh=0.5, kpt_conf_thresh=0.3, min_visible_kpts=4
    )
    frame = _frame([_person(0.9, 0.9)])
    filtered = dataset.filter_frame(frame, cfg)
    assert len(filtered["persons"]) == 1
    visibilities = [kp[2] for kp in filtered["persons"][0]["keypoints"]]
    assert all(v == 2 for v in visibilities)


def test_filter_drops_degenerate_boxes() -> None:
    cfg = dataset.FilterConfig(min_box_px=50)
    frame = _frame([_person(0.9, 0.9)])  # box is 20x40, below 50px
    filtered = dataset.filter_frame(frame, cfg)
    assert filtered["persons"] == []


def test_filter_demotes_keypoints_outside_box() -> None:
    cfg = dataset.FilterConfig(
        instance_conf_thresh=0.5, kpt_conf_thresh=0.3, min_visible_kpts=1
    )
    person = {
        "bbox_xyxy": [10.0, 20.0, 30.0, 60.0],
        "box_conf": 0.9,
        # First keypoint far outside the box (teacher error), rest inside.
        "keypoints": [[500.0, 500.0, 0.9]] + [[20.0, 40.0, 0.9] for _ in range(16)],
    }
    filtered = dataset.filter_frame(_frame([person]), cfg)
    kpts = filtered["persons"][0]["keypoints"]
    assert kpts[0][2] == 0  # outside box -> demoted to unlabeled
    assert kpts[1][2] == 2  # inside box -> kept


def test_filter_drops_extreme_aspect_box() -> None:
    cfg = dataset.FilterConfig(instance_conf_thresh=0.5, max_aspect_ratio=4.0)
    # 30 px wide, 200 px tall -> aspect ~6.7, above the limit but past min_box_px.
    person = {
        "bbox_xyxy": [10.0, 0.0, 40.0, 200.0],
        "box_conf": 0.9,
        "keypoints": [[25.0, 100.0, 0.9] for _ in range(17)],
    }
    filtered = dataset.filter_frame(_frame([person]), cfg)
    assert filtered["persons"] == []


def test_yolo_label_lines_are_normalized() -> None:
    cfg = dataset.FilterConfig()
    frame = _frame([_person(0.9, 0.9)])
    filtered = dataset.filter_frame(frame, cfg)
    lines = dataset.to_yolo_label_lines(filtered)
    parts = lines[0].split()
    assert parts[0] == "0"  # class id
    # cx = (10+30)/2 / 100 = 0.2 ; cy = (20+60)/2 / 200 = 0.2
    assert abs(float(parts[1]) - 0.2) < 1e-6
    assert abs(float(parts[2]) - 0.2) < 1e-6
    # w = 20/100 = 0.2 ; h = 40/200 = 0.2
    assert abs(float(parts[3]) - 0.2) < 1e-6
    assert abs(float(parts[4]) - 0.2) < 1e-6
    # 5 box fields + 17*3 keypoint fields.
    assert len(parts) == 5 + 17 * 3


def test_export_dataset_writes_yolo_layout(tmp_path: Path) -> None:
    import cv2

    # Two frames in different splits; create real tiny images on disk.
    payload = {"frames": []}
    for split in ("train", "val"):
        image_path = tmp_path / f"{split}.jpg"
        cv2.imwrite(str(image_path), np.zeros((200, 100, 3), dtype=np.uint8))
        frame = _frame([_person(0.9, 0.9)], split=split)
        frame["frame_path"] = str(image_path)
        payload["frames"].append(frame)

    out_root = tmp_path / "yolo"
    summary = dataset.export_dataset(payload, out_root, dataset.FilterConfig())

    data_yaml = out_root / "data.yaml"
    assert data_yaml.exists()
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert spec["kpt_shape"] == [17, 3]
    assert spec["flip_idx"] == coco.FLIP_IDX
    assert (out_root / "images" / "train").exists()
    assert (out_root / "labels" / "train").glob("*.txt")
    assert summary["train"]["frames"] == 1
    assert summary["val"]["frames"] == 1
