"""Tests for track annotator and correction logic (non-GUI parts)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analysis.track_annotator import (
    _bboxes_for_frame,
    _format_time,
    _parse_time,
    _point_in_bbox,
    bbox_iou_dict,
    load_tracks,
    propagate_corrections,
)
from video_analysis.track_correction import (
    apply_corrections,
    load_corrections,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tracks_csv(path: Path, n_frames: int = 5, n_tracks: int = 2):
    """Create a minimal tracks_2d.csv for testing."""
    rows = []
    for f in range(n_frames):
        for t in range(n_tracks):
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / 30.0,
                    "track_id": t,
                    "track_label": f"person_{t:02d}",
                    "bbox_x1": 100.0 + t * 200.0,
                    "bbox_y1": 100.0,
                    "bbox_x2": 250.0 + t * 200.0,
                    "bbox_y2": 400.0,
                    "track_confidence": 0.9,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------


def test_parse_time_mm_ss():
    assert _parse_time("13:26") == pytest.approx(806.0)
    assert _parse_time("0:00") == pytest.approx(0.0)


def test_parse_time_hh_mm_ss():
    assert _parse_time("1:00:00") == pytest.approx(3600.0)


def test_format_time():
    assert _format_time(806.0) == "13:26.00"


def test_bbox_iou_dict_identical():
    box = {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0}
    assert bbox_iou_dict(box, box) == pytest.approx(1.0)


def test_bbox_iou_dict_no_overlap():
    a = {"x1": 0.0, "y1": 0.0, "x2": 50.0, "y2": 50.0}
    b = {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 200.0}
    assert bbox_iou_dict(a, b) == pytest.approx(0.0)


def test_bbox_iou_dict_partial():
    a = {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0}
    b = {"x1": 50.0, "y1": 50.0, "x2": 150.0, "y2": 150.0}
    # Intersection = 50*50 = 2500, union = 10000+10000-2500 = 17500
    assert bbox_iou_dict(a, b) == pytest.approx(2500.0 / 17500.0)


def test_point_in_bbox():
    box = {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 200.0}
    assert _point_in_bbox(150, 150, box, 1.0) is True
    assert _point_in_bbox(50, 50, box, 1.0) is False
    # With scale
    assert _point_in_bbox(75, 75, box, 0.5) is True


# ---------------------------------------------------------------------------
# Tests: data loading
# ---------------------------------------------------------------------------


def test_load_tracks(tmp_path):
    csv_path = tmp_path / "tracks_2d.csv"
    _make_tracks_csv(csv_path, n_frames=10, n_tracks=2)
    df = load_tracks(csv_path, start_frame=3, end_frame=7)
    assert sorted(df["frame_idx"].unique()) == [3, 4, 5, 6]


def test_load_tracks_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tracks(tmp_path / "nonexistent.csv")


def test_bboxes_for_frame(tmp_path):
    csv_path = tmp_path / "tracks_2d.csv"
    _make_tracks_csv(csv_path, n_frames=3, n_tracks=2)
    df = load_tracks(csv_path)
    boxes = _bboxes_for_frame(df, 1)
    assert len(boxes) == 2
    assert boxes[0]["track_id"] == 0
    assert boxes[1]["track_id"] == 1


# ---------------------------------------------------------------------------
# Tests: corrections
# ---------------------------------------------------------------------------


def test_apply_corrections():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 0, 1, 1],
            "track_id": [0, 1, 0, 1],
            "value": [10, 20, 30, 40],
        }
    )
    corrections = {0: {0: 1, 1: 0}}  # swap at frame 0
    result = apply_corrections(df, corrections)
    # Frame 0: IDs swapped.
    f0 = result[result["frame_idx"] == 0]
    assert list(f0["track_id"]) == [1, 0]
    # Frame 1: unchanged.
    f1 = result[result["frame_idx"] == 1]
    assert list(f1["track_id"]) == [0, 1]


def test_load_corrections(tmp_path):
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps({"100": {"0": 2, "2": 0}}))
    corrections = load_corrections(path)
    assert corrections == {100: {0: 2, 2: 0}}


def test_load_corrections_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corrections(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# Tests: propagation
# ---------------------------------------------------------------------------


def test_propagate_corrections(tmp_path):
    """Propagation should carry corrections forward via IoU."""
    csv_path = tmp_path / "tracks_2d.csv"
    # All frames have the same bboxes → IoU = 1.0
    _make_tracks_csv(csv_path, n_frames=5, n_tracks=2)
    df = load_tracks(csv_path)
    frame_indices = sorted(df["frame_idx"].unique())

    corrections = {0: {0: 1, 1: 0}}  # swap at frame 0
    propagate_corrections(df, corrections, 0, frame_indices, iou_threshold=0.3)

    # Should have propagated to frames 1-4.
    assert 1 in corrections
    assert 2 in corrections
    assert 3 in corrections
    assert 4 in corrections


def test_propagate_stops_at_next_annotation(tmp_path):
    """Propagation should stop at already-annotated frames."""
    csv_path = tmp_path / "tracks_2d.csv"
    _make_tracks_csv(csv_path, n_frames=10, n_tracks=2)
    df = load_tracks(csv_path)
    frame_indices = sorted(df["frame_idx"].unique())

    # User annotated frame 0 and frame 5.
    corrections = {
        0: {0: 1, 1: 0},
        5: {0: 0, 1: 1},
    }
    propagate_corrections(df, corrections, 0, frame_indices, iou_threshold=0.3)

    # Should propagate frames 1-4 only (stop at 5).
    assert 1 in corrections
    assert 4 in corrections
    # Frame 5 should keep its original annotation, not be overwritten.
    assert corrections[5] == {0: 0, 1: 1}
