"""Tests for teacher pseudo-labeling person extraction and the run loop."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pose_training import pseudo_label


def test_extract_persons_keeps_top_k_by_confidence() -> None:
    boxes_xyxy = np.array([[0, 0, 10, 20], [5, 5, 15, 25], [1, 1, 2, 2]], dtype=float)
    boxes_conf = np.array([0.4, 0.9, 0.7], dtype=float)
    kpts_xy = np.zeros((3, 17, 2), dtype=float)
    kpts_conf = np.ones((3, 17), dtype=float)
    persons = pseudo_label.extract_persons(
        boxes_xyxy, boxes_conf, kpts_xy, kpts_conf, max_persons=2
    )
    assert len(persons) == 2
    # Sorted by descending confidence: 0.9 then 0.7.
    assert persons[0]["box_conf"] == 0.9
    assert persons[1]["box_conf"] == 0.7
    assert len(persons[0]["keypoints"]) == 17
    assert len(persons[0]["keypoints"][0]) == 3


def test_flip_persons_back_inverts_a_mirror() -> None:
    width = 100
    # A person with the nose (idx 0) and left_shoulder (idx 5) placed; on the
    # mirrored image their x is width-1-x and left/right swap on the way back.
    person = {
        "bbox_xyxy": [10.0, 20.0, 40.0, 80.0],
        "box_conf": 0.9,
        "keypoints": [[30.0, 25.0, 0.9]] + [[0.0, 0.0, 0.0]] * 16,
    }
    back = pseudo_label.flip_persons_back([person], width)[0]
    # Box corners swap and reflect: x1' = width-1-x2, x2' = width-1-x1.
    assert back["bbox_xyxy"][0] == width - 1 - 40.0
    assert back["bbox_xyxy"][2] == width - 1 - 10.0
    # The nose is central; its x reflects to width-1-30.
    assert back["keypoints"][0][0] == width - 1 - 30.0


def test_flip_persons_back_is_self_inverse() -> None:
    width = 200
    person = {
        "bbox_xyxy": [10.0, 20.0, 40.0, 80.0],
        "box_conf": 0.5,
        "keypoints": [[float(i), float(2 * i), 0.5] for i in range(17)],
    }
    once = pseudo_label.flip_persons_back([person], width)
    twice = pseudo_label.flip_persons_back(once, width)[0]
    for original, restored in zip(person["keypoints"], twice["keypoints"]):
        assert abs(original[0] - restored[0]) < 1e-6
        assert abs(original[1] - restored[1]) < 1e-6


def test_merge_persons_averages_matched_and_keeps_unmatched() -> None:
    a = [
        {
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
            "box_conf": 0.8,
            "keypoints": [[2.0, 2.0, 1.0]] + [[0.0, 0.0, 0.0]] * 16,
        }
    ]
    b = [
        {
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
            "box_conf": 0.6,
            "keypoints": [[4.0, 4.0, 1.0]] + [[0.0, 0.0, 0.0]] * 16,
        },
        {
            "bbox_xyxy": [50.0, 50.0, 60.0, 60.0],
            "box_conf": 0.5,
            "keypoints": [[55.0, 55.0, 1.0]] + [[0.0, 0.0, 0.0]] * 16,
        },
    ]
    merged = pseudo_label.merge_persons(a, b, max_persons=2, match_iou=0.5)
    assert len(merged) == 2
    # Matched pair averages the nose x to (2+4)/2 = 3.
    assert merged[0]["keypoints"][0][0] == 3.0
    # The far, unmatched instance survives.
    assert any(p["bbox_xyxy"][0] == 50.0 for p in merged)


def test_extract_persons_handles_no_detections() -> None:
    empty = np.zeros((0, 4), dtype=float)
    persons = pseudo_label.extract_persons(
        empty,
        np.zeros((0,), dtype=float),
        np.zeros((0, 17, 2), dtype=float),
        np.zeros((0, 17), dtype=float),
        max_persons=2,
    )
    assert persons == []


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf


class _FakeKeypoints:
    def __init__(self, xy: np.ndarray, conf: np.ndarray) -> None:
        self.xy = xy
        self.conf = conf


class _FakeResult:
    def __init__(self) -> None:
        self.orig_shape = (480, 960)
        self.boxes = _FakeBoxes(
            np.array([[10, 10, 50, 90]], dtype=float), np.array([0.8], dtype=float)
        )
        self.keypoints = _FakeKeypoints(
            np.zeros((1, 17, 2), dtype=float), np.ones((1, 17), dtype=float)
        )


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, paths, **kwargs):  # noqa: ANN001
        self.calls += 1
        return [_FakeResult() for _ in paths]


def _write_manifest(tmp_path: Path) -> Path:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True)
    manifest = tmp_path / "frame_manifest.csv"
    rows = []
    for i in range(3):
        frame = frame_dir / f"f{i}.jpg"
        frame.write_bytes(b"x")
        rows.append(
            {
                "frame_path": str(frame),
                "participant": "P001",
                "angle": "SHORT",
                "split": "train",
                "source_video": "P001_FULL_SHORT_compressed.mp4",
                "source_timestamp_s": float(i),
                "dhash": 0,
            }
        )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_run_pseudo_labeling_writes_records(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    out_path = tmp_path / "pseudo_labels_raw.json"
    config = pseudo_label.PseudoLabelConfig(
        manifest_path=manifest,
        output_path=out_path,
        batch_size=2,
    )
    payload = pseudo_label.run_pseudo_labeling(config, model=_FakeModel())
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(on_disk["frames"]) == 3
    first = on_disk["frames"][0]
    assert first["width"] == 960
    assert first["height"] == 480
    assert first["participant"] == "P001"
    assert len(first["persons"]) == 1
    assert payload["frames"][0]["persons"][0]["box_conf"] == 0.8
