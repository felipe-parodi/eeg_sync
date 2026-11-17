"""Tests for the pose/tracking benchmark harness (pose_training.bench)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pose_training.bench import (
    AnchorGold,
    anchor_scores,
    box_iou,
    load_anchor_gold,
    role_flips_per_min,
    split_anchors,
    two_person_rate,
)
from video_inference.detection_cache import DetCache


def _assign(rows: list[tuple[int, float, str, tuple[float, float, float, float]]]):
    return pd.DataFrame(
        [
            {
                "frame_idx": f,
                "timestamp_s": t,
                "role": r,
                "x1": b[0],
                "y1": b[1],
                "x2": b[2],
                "y2": b[3],
            }
            for f, t, r, b in rows
        ]
    )


def test_box_iou_basic() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_det_cache_roundtrip(tmp_path: Path) -> None:
    cache = DetCache(
        frame_idx=np.array([0, 1, 2]),
        timestamp_s=np.array([10.0, 10.2, 10.4]),
        offsets=np.array([0, 2, 2, 3]),
        boxes=np.arange(12, dtype=np.float32).reshape(3, 4),
        conf=np.array([0.9, 0.2, 0.5], dtype=np.float32),
        keypoints=np.zeros((3, 17, 3), dtype=np.float32),
        embeddings=np.ones((3, 8), dtype=np.float32),
        image_size=(960, 540),
    )
    path = tmp_path / "c.npz"
    cache.save(path)
    loaded = DetCache.load(path)
    assert loaded.n_frames == 3
    assert loaded.frame_dets(0) == slice(0, 2)
    assert loaded.frame_dets(1) == slice(2, 2)
    np.testing.assert_allclose(loaded.boxes, cache.boxes)
    assert loaded.image_size == (960, 540)


def test_two_person_rate_counts_frames_with_both_roles() -> None:
    a = _assign(
        [
            (0, 0.0, "parent", (0, 0, 1, 1)),
            (0, 0.0, "child", (2, 2, 3, 3)),
            (1, 0.2, "parent", (0, 0, 1, 1)),
        ]
    )
    assert two_person_rate(a, n_frames=4) == pytest.approx(0.25)


def test_anchor_scores_role_and_recall() -> None:
    gold = [
        AnchorGold(10.0, (0, 0, 10, 10), (50, 50, 60, 60)),
        AnchorGold(20.0, (0, 0, 10, 10), (50, 50, 60, 60)),
        AnchorGold(30.0, (0, 0, 10, 10), (50, 50, 60, 60)),
    ]
    a = _assign(
        [
            # correct roles
            (0, 10.0, "parent", (0, 0, 10, 10)),
            (0, 10.0, "child", (50, 50, 60, 60)),
            # swapped roles
            (1, 20.0, "parent", (50, 50, 60, 60)),
            (1, 20.0, "child", (0, 0, 10, 10)),
            # child missing at t=30
            (2, 30.0, "parent", (0, 0, 10, 10)),
        ]
    )
    scores = anchor_scores(a, gold)
    assert scores["n_anchors"] == 3
    # role accuracy counts each gold person (6) that is matched with right role
    assert scores["role_acc"] == pytest.approx(3 / 6)
    # recall: any output box overlapping the gold box, 5 of 6
    assert scores["recall"] == pytest.approx(5 / 6)


def test_role_flips_detects_swap() -> None:
    rows = []
    for f in range(10):
        p, c = (0, 0, 10, 10), (100, 0, 110, 10)
        if f >= 5:
            p, c = c, p
        rows += [(f, f / 6, "parent", p), (f, f / 6, "child", c)]
    flips = role_flips_per_min(_assign(rows))
    # one swap over 10 frames at 6 fps (1.5 s of footage)
    assert flips == pytest.approx(1 / (9 / 6 / 60))


def test_split_anchors_alternates_by_time() -> None:
    gold = [AnchorGold(float(t), (0, 0, 1, 1), (2, 2, 3, 3)) for t in (5, 1, 3, 2, 4)]
    fit, test = split_anchors(gold)
    assert [g.timestamp_s for g in fit] == [1.0, 3.0, 5.0]
    assert [g.timestamp_s for g in test] == [2.0, 4.0]


def test_load_anchor_gold_maps_track_ids_to_boxes(tmp_path: Path) -> None:
    cam = tmp_path / "anchors" / "camera_a"
    cam.mkdir(parents=True)
    manifest = [
        {
            "anchor_idx": 1,
            "frame_idx": 6,
            "timestamp_s": 778.0,
            "tracks": [
                {"track_id": 0, "bbox_xyxy": [0, 0, 10, 10]},
                {"track_id": 1, "bbox_xyxy": [50, 50, 60, 60]},
            ],
        },
        {
            "anchor_idx": 2,
            "frame_idx": 9,
            "timestamp_s": 779.0,
            "tracks": [
                {"track_id": 0, "bbox_xyxy": [0, 0, 10, 10]},
                {"track_id": 1, "bbox_xyxy": [50, 50, 60, 60]},
            ],
        },
    ]
    labels = [
        {"frame_idx": 6, "parent_track_id": 1, "child_track_id": 0},
        {"frame_idx": 9, "skipped": True},
    ]
    (cam / "manifest.json").write_text(json.dumps(manifest))
    (cam / "anchors.json").write_text(json.dumps(labels))
    gold = load_anchor_gold(tmp_path, "camera_a")
    assert len(gold) == 1
    assert gold[0].timestamp_s == 778.0
    assert gold[0].parent_box == (50, 50, 60, 60)
    assert gold[0].child_box == (0, 0, 10, 10)


def test_child_taller_rate() -> None:
    from pose_training.bench import child_taller_rate

    a = _assign(
        [
            (0, 0.0, "parent", (0, 0, 10, 100)),
            (0, 0.0, "child", (0, 0, 10, 50)),
            (1, 0.2, "parent", (0, 0, 10, 50)),
            (1, 0.2, "child", (0, 0, 10, 100)),
            (2, 0.4, "parent", (0, 0, 10, 50)),
        ]
    )
    assert child_taller_rate(a) == pytest.approx(0.5)


def test_kept_confident_rate() -> None:
    from pose_training.bench import kept_confident_rate

    ref = pd.DataFrame(
        {
            "frame_idx": [0, 0, 1],
            "bbox_x1": [0, 50, 0],
            "bbox_y1": [0, 50, 0],
            "bbox_x2": [10, 60, 10],
            "bbox_y2": [10, 60, 10],
            "track_confidence": [0.9, 0.9, 0.2],
        }
    )
    a = _assign([(0, 0.0, "parent", (0, 0, 10, 10))])
    assert kept_confident_rate(a, ref) == pytest.approx(0.5)


def test_discover_cameras_finds_archived_sessions(tmp_path, monkeypatch) -> None:
    """Sessions moved to output/_archive stay available to the benchmark."""
    import json

    from pose_training import bench

    out = tmp_path / "output"
    jobs = tmp_path / "jobs"
    job = jobs / "20260101_000000_1010_abcd"
    (job / "anchors" / "camera_a").mkdir(parents=True)
    (job / "anchors" / "camera_a" / "anchors.json").write_text("[]", encoding="utf-8")
    (job / "submission_metadata.json").write_text(
        json.dumps({"session_id": "1010"}), encoding="utf-8"
    )
    frames = out / "_archive" / "1010" / "camera_a" / "frames"
    frames.mkdir(parents=True)
    (frames / "frame_index.csv").write_text("frame_idx,timestamp_s\n", encoding="utf-8")
    monkeypatch.setattr(bench, "OUTPUT_ROOT", out)

    cams = bench.discover_cameras(jobs)

    assert [c.key for c in cams] == ["1010_camera_a"]
    assert cams[0].camera_dir == out / "_archive" / "1010" / "camera_a"
