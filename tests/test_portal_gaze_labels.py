"""Tests for the gaze ground-truth labeling helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gaze_analysis.config import SessionBlock
from portal.gaze_labels import (
    GAZE_TARGETS,
    load_gaze_labels,
    sample_gaze_label_frames,
    save_gaze_labels,
)


def _camera(tmp_path: Path, n: int = 600, child_from: int = 0) -> Path:
    cam = tmp_path / "camera_a"
    frames = cam / "frames"
    frames.mkdir(parents=True)
    index, tracks = [], []
    for i in range(n):
        t = 100.0 + i / 6
        name = f"frame_{i + 1:06d}.jpg"
        index.append((i, t, name))
        g = int(round(t * 6))
        tracks.append((g, t, 0, "parent", 10, 10, 100, 300, False))
        if i >= child_from:
            tracks.append((g, t, 1, "child", 200, 100, 260, 300, i % 50 == 0))
    pd.DataFrame(index, columns=["frame_idx", "timestamp_s", "image_name"]).to_csv(
        frames / "frame_index.csv", index=False
    )
    pd.DataFrame(
        tracks,
        columns=[
            "frame_idx",
            "timestamp_s",
            "track_id",
            "track_label",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "interpolated",
        ],
    ).to_csv(cam / "tracks_2d_smooth.csv", index=False)
    return cam


def test_sample_frames_spread_over_blocks_with_both_people(tmp_path: Path) -> None:
    cam = _camera(tmp_path, child_from=300)  # child only in second half
    blocks = [SessionBlock("a", 100.0, 150.0), SessionBlock("b", 150.0, 199.0)]

    items = sample_gaze_label_frames(cam, blocks, frame_rate=6.0, n_frames=20)

    assert len(items) == 20
    times = [it["timestamp_s"] for it in items]
    assert times == sorted(times)
    # Only frames where both people are observed (child appears at t=150).
    assert min(times) >= 150.0
    assert {it["block"] for it in items} == {"b"}
    first = items[0]
    assert first["image_name"].startswith("frame_")
    assert set(first["boxes"]) == {"parent", "child"}


def test_sample_frames_is_deterministic(tmp_path: Path) -> None:
    cam = _camera(tmp_path)
    blocks = [SessionBlock("a", 100.0, 199.0)]
    a = sample_gaze_label_frames(cam, blocks, frame_rate=6.0, n_frames=10)
    b = sample_gaze_label_frames(cam, blocks, frame_rate=6.0, n_frames=10)
    assert a == b


def test_save_and_load_gaze_labels_round_trip(tmp_path: Path) -> None:
    cam = _camera(tmp_path)
    labels = [
        {
            "frame_idx": 600,
            "timestamp_s": 100.0,
            "person": "child",
            "target": "partner_face",
            "point": [0.4, 0.3],
        }
    ]
    save_gaze_labels(cam, labels, labeler="RA")
    assert load_gaze_labels(cam) == labels
    raw = json.loads((cam / "gaze_labels.json").read_text(encoding="utf-8"))
    assert raw["labeler"] == "RA"


@pytest.mark.parametrize(
    "bad",
    [
        {"frame_idx": 1, "timestamp_s": 0.0, "person": "dog", "target": "object"},
        {"frame_idx": 1, "timestamp_s": 0.0, "person": "child", "target": "moon"},
        {
            "frame_idx": 1,
            "timestamp_s": 0.0,
            "person": "child",
            "target": "object",
            "point": [2.0, 0.1],
        },
    ],
)
def test_save_gaze_labels_rejects_bad_entries(tmp_path: Path, bad: dict) -> None:
    with pytest.raises(ValueError):
        save_gaze_labels(tmp_path, [bad])


def test_targets_have_plain_language_labels() -> None:
    assert "partner_face" in GAZE_TARGETS
    assert all(isinstance(v, str) and v for v in GAZE_TARGETS.values())
