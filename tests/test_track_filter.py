"""Tests for block-based track filtering."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analysis.track_filter import (
    BlockDef,
    TrackFilterConfig,
    identify_top_tracks,
    parse_blocks_string,
)


def _make_tracks_df(n_frames: int = 100, tracks=None):
    """Create synthetic tracks DataFrame.

    Args:
        n_frames: Number of frames.
        tracks: List of (track_id, start_frame, end_frame, bbox_size) tuples.
    """
    if tracks is None:
        tracks = [
            (0, 0, n_frames, 200),  # parent: large bbox, always present
            (1, 0, n_frames, 100),  # child: small bbox, always present
            (2, 30, 50, 150),  # researcher: medium, brief appearance
        ]

    rows = []
    for tid, sf, ef, size in tracks:
        for f in range(sf, ef):
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / 30.0,
                    "track_id": tid,
                    "track_label": f"person_{tid:02d}",
                    "bbox_x1": 100.0,
                    "bbox_y1": 100.0,
                    "bbox_x2": 100.0 + size,
                    "bbox_y2": 100.0 + size,
                    "track_confidence": 0.9,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: identify_top_tracks
# ---------------------------------------------------------------------------


def test_identify_top_tracks_basic():
    df = _make_tracks_df(100)
    result = identify_top_tracks(df, 0, 99, n_keep=2)
    assert len(result) == 2
    # Track 0 (area=200*200=40000) should be parent (first).
    assert result[0][0] == 0
    assert result[0][1] == "parent"
    # Track 1 (area=100*100=10000) should be child.
    assert result[1][0] == 1
    assert result[1][1] == "child"


def test_identify_top_tracks_ignores_brief():
    """Brief researcher track should not be in top 2."""
    df = _make_tracks_df(100)
    result = identify_top_tracks(df, 0, 99, n_keep=2)
    track_ids = [r[0] for r in result]
    assert 2 not in track_ids


def test_identify_top_tracks_empty():
    df = _make_tracks_df(100)
    result = identify_top_tracks(df, 200, 300, n_keep=2)
    assert result == []


def test_identify_top_tracks_area_ordering():
    """Larger bbox area should be parent (role 0)."""
    tracks = [
        (5, 0, 50, 300),  # big bbox
        (7, 0, 50, 80),  # small bbox
    ]
    df = _make_tracks_df(50, tracks)
    result = identify_top_tracks(df, 0, 49, n_keep=2)
    assert result[0][0] == 5  # bigger = parent
    assert result[0][1] == "parent"
    assert result[1][0] == 7  # smaller = child
    assert result[1][1] == "child"


# ---------------------------------------------------------------------------
# Tests: parse_blocks_string
# ---------------------------------------------------------------------------


def test_parse_blocks_string():
    blocks = parse_blocks_string(
        "grocery,13:26,23:40;synchrony,27:56,28:45;storybook,29:22,37:06"
    )
    assert len(blocks) == 3
    assert blocks[0].name == "grocery"
    assert blocks[0].start_time == "13:26"
    assert blocks[2].name == "storybook"
    assert blocks[2].end_time == "37:06"


def test_parse_blocks_string_invalid():
    with pytest.raises(ValueError):
        parse_blocks_string("bad_format")


# ---------------------------------------------------------------------------
# Tests: filter_tracks (integration)
# ---------------------------------------------------------------------------


def test_filter_tracks_integration(tmp_path):
    """End-to-end test: filter tracks by block, remap IDs."""
    from video_analysis.track_filter import filter_tracks

    # Create session structure.
    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)

    # Track 0 = big bbox (parent), track 1 = small (child), track 2 = brief.
    df = _make_tracks_df(100)
    df.to_csv(cam_dir / "tracks_2d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[BlockDef("test_block", "0:00", "0:03")],  # frames 0-90
        source_fps=30.0,
        n_keep=2,
    )
    summary = filter_tracks(cfg)

    assert summary["total_rows"] > 0
    # Read filtered output.
    filtered = pd.read_csv(cam_dir / "tracks_2d_filtered.csv")
    assert set(filtered["track_id"].unique()) == {0, 1}
    assert set(filtered["track_label"].unique()) == {"parent", "child"}


def test_filter_tracks_uses_timestamp_windows_for_segmented_outputs(tmp_path):
    """Filtering should use original timestamps, not sparse frame indices."""
    from video_analysis.track_filter import filter_tracks

    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)
    rows = []
    for frame_idx, timestamp_s in [(0, 100.0), (1, 100.5), (2, 200.0)]:
        rows.extend(
            [
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "track_id": 7,
                    "track_label": "person_07",
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": 100.0,
                    "bbox_y2": 200.0,
                    "track_confidence": 0.9,
                },
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "track_id": 9,
                    "track_label": "person_09",
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": 80.0,
                    "bbox_y2": 120.0,
                    "track_confidence": 0.9,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(cam_dir / "tracks_2d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[BlockDef("free_play", "1:40", "1:41")],
        source_fps=30.0,
        n_keep=2,
    )
    summary = filter_tracks(cfg)

    filtered = pd.read_csv(cam_dir / "tracks_2d_filtered.csv")

    assert summary["total_rows"] == 4
    assert set(filtered["frame_idx"].unique()) == {0, 1}
    assert set(filtered["timestamp_s"].unique()) == {100.0, 100.5}
    assert set(filtered["track_id"].unique()) == {0, 1}
