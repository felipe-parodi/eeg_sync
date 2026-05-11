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
    _read_source_fps_from_manifest,
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


# ---------------------------------------------------------------------------
# Tests: source FPS resolution from manifest
# ---------------------------------------------------------------------------


def _write_manifest(camera_dir: Path, fps: float) -> None:
    """Write a minimal pipeline-style manifest into *camera_dir*."""
    camera_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "0.1.0",
        "session_id": "test_session",
        "source_videos": [
            {"camera_id": camera_dir.name, "relative_path": "x.mp4", "fps": fps}
        ],
        "assumptions": {"max_persons": 2, "enforce_exact_person_count": False},
        "outputs": {"tracks_2d": "tracks_2d.csv", "pose_3d": "pose_3d.csv"},
    }
    (camera_dir / "manifest.json").write_text(json.dumps(manifest))


def test_read_source_fps_uses_manifest_value(tmp_path):
    """Annotator should pick up the inference fps from manifest.json."""
    camera_dir = tmp_path / "camera_a"
    _write_manifest(camera_dir, fps=5.0)
    assert _read_source_fps_from_manifest(camera_dir) == 5.0


def test_read_source_fps_missing_manifest_returns_default(tmp_path):
    """No manifest -> fall back to the supplied default."""
    assert _read_source_fps_from_manifest(tmp_path, default=15.0) == 15.0


def test_read_source_fps_malformed_manifest_returns_default(tmp_path):
    """Unparseable manifest -> fall back to default (don't crash the UI)."""
    (tmp_path / "manifest.json").write_text("not json")
    assert _read_source_fps_from_manifest(tmp_path, default=15.0) == 15.0


# ---------------------------------------------------------------------------
# Tests: corrections reload round-trip
# ---------------------------------------------------------------------------


def test_corrections_reload_preserves_int_keys_at_both_levels(tmp_path):
    """Saved-then-reloaded corrections must keep int keys at outer AND inner.

    Regression: an earlier reload path int-converted only outer keys,
    leaving inner keys as JSON strings. That made previously-saved
    corrections invisible to UI display and to propagate_corrections.
    """
    corrections = {5: {0: 1, 1: 0}}
    # Mimic the annotator's save format (line 539 of track_annotator.py).
    out = {str(k): v for k, v in corrections.items()}
    path = tmp_path / "track_corrections.json"
    path.write_text(json.dumps(out))

    reloaded = load_corrections(path)

    assert reloaded == corrections
    assert all(isinstance(k, int) for k in reloaded.keys())
    assert all(isinstance(k, int) for k in reloaded[5].keys())
    # And the lookups the annotator/propagation rely on must succeed:
    assert reloaded[5].get(0) == 1
    assert reloaded[5].get(1) == 0
