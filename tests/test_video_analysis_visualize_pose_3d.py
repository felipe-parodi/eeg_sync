"""Tests for 3D pose skeleton visualization."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analysis.visualize_pose_3d import (
    Visualize3DConfig,
    _bgr_to_rgb_normalized,
    _keypoint_name_to_index,
    compute_axis_limits,
    load_pose_data,
    render_pose_3d,
)

# ---------------------------------------------------------------------------
# Helper to create synthetic pose CSV data
# ---------------------------------------------------------------------------

COCO17_NAMES = [f"kp_{i:03d}" for i in range(17)]


def _make_synthetic_pose_csv(path: Path, n_frames: int = 5, n_tracks: int = 2):
    """Create a minimal pose_3d.csv for testing."""
    rows = []
    for frame_idx in range(n_frames):
        ts = frame_idx / 30.0
        for track_id in range(n_tracks):
            for kp_name in COCO17_NAMES:
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_s": ts,
                        "track_id": track_id,
                        "track_label": f"person_{track_id:02d}",
                        "keypoint_name": kp_name,
                        "x_m": 100.0 + track_id * 50.0 + np.random.rand() * 10,
                        "y_m": 200.0 + track_id * 30.0 + np.random.rand() * 10,
                        "z_m": -0.5 + np.random.rand() * 1.0,
                        "keypoint_confidence": 0.5 + np.random.rand() * 0.5,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


def test_bgr_to_rgb_normalized():
    assert _bgr_to_rgb_normalized((255, 0, 0)) == pytest.approx((0.0, 0.0, 1.0))
    assert _bgr_to_rgb_normalized((0, 128, 255)) == pytest.approx(
        (1.0, 128 / 255.0, 0.0)
    )


def test_keypoint_name_to_index_kp_format():
    assert _keypoint_name_to_index("kp_000") == 0
    assert _keypoint_name_to_index("kp_016") == 16


def test_keypoint_name_to_index_unknown_raises():
    with pytest.raises(ValueError):
        _keypoint_name_to_index("unknown_keypoint")


# ---------------------------------------------------------------------------
# Unit tests: data loading
# ---------------------------------------------------------------------------


def test_load_pose_data_filters_frames(tmp_path):
    csv_path = tmp_path / "pose_3d.csv"
    _make_synthetic_pose_csv(csv_path, n_frames=5, n_tracks=1)

    df = load_pose_data(str(csv_path), start_frame=2, max_frames=2)
    unique_frames = sorted(df["frame_idx"].unique())
    assert unique_frames == [2, 3]


def test_load_pose_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pose_data(str(tmp_path / "nonexistent.csv"))


# ---------------------------------------------------------------------------
# Unit tests: axis limits
# ---------------------------------------------------------------------------


def test_compute_axis_limits(tmp_path):
    csv_path = tmp_path / "pose_3d.csv"
    _make_synthetic_pose_csv(csv_path, n_frames=3, n_tracks=1)
    df = load_pose_data(str(csv_path))
    limits = compute_axis_limits(df)

    assert "x" in limits and "y" in limits and "z" in limits
    # Limits should contain all data points with padding
    assert limits["x"][0] <= df["x_m"].min()
    assert limits["x"][1] >= df["x_m"].max()


# ---------------------------------------------------------------------------
# Integration tests: rendering
# ---------------------------------------------------------------------------


def test_render_pose_3d_writes_snapshots(tmp_path):
    csv_path = tmp_path / "pose_3d.csv"
    _make_synthetic_pose_csv(csv_path, n_frames=3, n_tracks=2)
    output_dir = tmp_path / "viz"

    cfg = Visualize3DConfig(
        pose_csv=str(csv_path),
        output_dir=str(output_dir),
    )
    summary = render_pose_3d(cfg)

    assert summary["rendered_frames"] == 3
    pngs = list(output_dir.glob("*.png"))
    assert len(pngs) == 3


def test_render_pose_3d_snapshot_interval(tmp_path):
    csv_path = tmp_path / "pose_3d.csv"
    _make_synthetic_pose_csv(csv_path, n_frames=10, n_tracks=1)
    output_dir = tmp_path / "viz"

    cfg = Visualize3DConfig(
        pose_csv=str(csv_path),
        output_dir=str(output_dir),
        snapshot_interval=5,
    )
    summary = render_pose_3d(cfg)

    assert summary["rendered_frames"] == 2
    pngs = list(output_dir.glob("*.png"))
    assert len(pngs) == 2


def test_render_pose_3d_start_frame(tmp_path):
    csv_path = tmp_path / "pose_3d.csv"
    _make_synthetic_pose_csv(csv_path, n_frames=10, n_tracks=1)
    output_dir = tmp_path / "viz"

    cfg = Visualize3DConfig(
        pose_csv=str(csv_path),
        output_dir=str(output_dir),
        start_frame=7,
        max_frames=2,
    )
    summary = render_pose_3d(cfg)

    assert summary["rendered_frames"] == 2


def test_render_pose_3d_video_output(tmp_path):
    csv_path = tmp_path / "pose_3d.csv"
    _make_synthetic_pose_csv(csv_path, n_frames=3, n_tracks=1)
    output_dir = tmp_path / "viz"
    video_path = tmp_path / "test_3d.mp4"

    cfg = Visualize3DConfig(
        pose_csv=str(csv_path),
        output_dir=str(output_dir),
        output_video=str(video_path),
        output_fps=10.0,
    )
    summary = render_pose_3d(cfg)

    assert video_path.exists()
    assert video_path.stat().st_size > 0
    assert summary["output_video"] == str(video_path)
