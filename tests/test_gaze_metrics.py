"""Tests for per-block gaze metrics computation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaze_analysis.config import SessionBlock
from video_analysis.gaze_metrics import compute_per_block_gaze_metrics


def _make_gaze_df(
    n_frames: int = 50,
    fps: float = 5.0,
    start_s: float = 0.0,
) -> pd.DataFrame:
    """Create synthetic gaze data with 2 tracks."""
    rows = []
    for f in range(n_frames):
        ts = start_s + f / fps
        frame_idx = int(start_s * 30) + f * 6  # simulate 5Hz from 30fps
        for tid in [0, 1]:
            rows.append({
                "frame_idx": frame_idx,
                "timestamp_s": ts,
                "track_id": tid,
                "gaze_peak_x": 0.5 + np.random.randn() * 0.1,
                "gaze_peak_y": 0.5 + np.random.randn() * 0.1,
                "gaze_peak_value": 0.8,
                "inout_score": 0.9,
                "head_source": "keypoints",
            })
    return pd.DataFrame(rows)


def _make_pose_df(
    n_frames: int = 50,
    fps: float = 5.0,
    start_s: float = 0.0,
) -> pd.DataFrame:
    """Create synthetic pose data with head + torso keypoints."""
    rows = []
    kp_names = [
        "kp_000", "kp_001", "kp_002", "kp_003", "kp_004",  # head
        "kp_005", "kp_006", "kp_011", "kp_012",  # torso
    ]
    for f in range(n_frames):
        ts = start_s + f / fps
        frame_idx = int(start_s * 30) + f * 6
        for tid, cx in [(0, 300.0), (1, 500.0)]:
            for kp in kp_names:
                rows.append({
                    "frame_idx": frame_idx,
                    "timestamp_s": ts,
                    "track_id": tid,
                    "track_label": "parent" if tid == 0 else "child",
                    "keypoint_name": kp,
                    "x_m": cx + np.random.randn() * 2,
                    "y_m": 200.0 + np.random.randn() * 2,
                    "z_m": 0.0,
                    "keypoint_confidence": 0.9,
                })
    return pd.DataFrame(rows)


def _make_tracks_df(
    n_frames: int = 50,
    fps: float = 5.0,
    start_s: float = 0.0,
) -> pd.DataFrame:
    """Create synthetic tracks data."""
    rows = []
    for f in range(n_frames):
        ts = start_s + f / fps
        frame_idx = int(start_s * 30) + f * 6
        for tid, cx in [(0, 300.0), (1, 500.0)]:
            rows.append({
                "frame_idx": frame_idx,
                "timestamp_s": ts,
                "track_id": tid,
                "bbox_x1": cx - 50,
                "bbox_y1": 100.0,
                "bbox_x2": cx + 50,
                "bbox_y2": 400.0,
                "detection_confidence": 0.9,
                "image": f"frame_{frame_idx:06d}.jpg",
            })
    return pd.DataFrame(rows)


def _make_heatmaps(n_frames: int = 50) -> tuple:
    """Create synthetic heatmap data."""
    heatmaps = np.random.rand(n_frames * 2, 64, 64).astype(np.float32)
    keys = []
    for f in range(n_frames):
        frame_idx = f * 6
        keys.append(f"f{frame_idx:06d}_t0")
        keys.append(f"f{frame_idx:06d}_t1")
    return heatmaps, keys


def test_compute_gaze_metrics_basic():
    """Gaze metrics computed for a single block."""
    blocks = [SessionBlock("test_block", 0.0, 10.0, "#999")]
    gaze_df = _make_gaze_df(50, fps=5.0)
    pose_df = _make_pose_df(50, fps=5.0)
    tracks_df = _make_tracks_df(50, fps=5.0)
    heatmaps, hm_keys = _make_heatmaps(50)

    results = compute_per_block_gaze_metrics(
        gaze_df, pose_df, tracks_df, heatmaps, hm_keys, blocks
    )

    assert not results["categories_df"].empty
    assert "block" in results["categories_df"].columns
    assert "gaze_category" in results["categories_df"].columns


def test_compute_gaze_metrics_convergence():
    """Convergence is computed from heatmaps."""
    blocks = [SessionBlock("test_block", 0.0, 10.0, "#999")]
    gaze_df = _make_gaze_df(50, fps=5.0)
    pose_df = _make_pose_df(50, fps=5.0)
    tracks_df = _make_tracks_df(50, fps=5.0)
    heatmaps, hm_keys = _make_heatmaps(50)

    results = compute_per_block_gaze_metrics(
        gaze_df, pose_df, tracks_df, heatmaps, hm_keys, blocks
    )

    assert not results["convergence_df"].empty
    assert "gaze_convergence_score" in results["convergence_df"].columns
    scores = results["convergence_df"]["gaze_convergence_score"]
    assert (scores >= 0).all() and (scores <= 1).all()


def test_compute_gaze_metrics_multiple_blocks():
    """Metrics separated per block."""
    blocks = [
        SessionBlock("block_a", 0.0, 5.0, "#999"),
        SessionBlock("block_b", 6.0, 10.0, "#999"),
    ]
    gaze_df = _make_gaze_df(50, fps=5.0)
    pose_df = _make_pose_df(50, fps=5.0)
    tracks_df = _make_tracks_df(50, fps=5.0)
    heatmaps, hm_keys = _make_heatmaps(50)

    results = compute_per_block_gaze_metrics(
        gaze_df, pose_df, tracks_df, heatmaps, hm_keys, blocks
    )

    block_names = results["categories_df"]["block"].unique()
    assert "block_a" in block_names
    assert "block_b" in block_names


def test_compute_gaze_metrics_stats():
    """Block stats include expected keys."""
    blocks = [SessionBlock("test", 0.0, 10.0, "#999")]
    gaze_df = _make_gaze_df(50, fps=5.0)
    pose_df = _make_pose_df(50, fps=5.0)
    tracks_df = _make_tracks_df(50, fps=5.0)
    heatmaps, hm_keys = _make_heatmaps(50)

    results = compute_per_block_gaze_metrics(
        gaze_df, pose_df, tracks_df, heatmaps, hm_keys, blocks
    )

    stats = results["block_stats"][0]
    assert stats["block"] == "test"
    assert "n_frames" in stats
    assert "mutual_gaze_pct" in stats
    assert "independent_pct" in stats


def test_compute_gaze_metrics_empty():
    """Empty gaze data returns empty results."""
    blocks = [SessionBlock("test", 100.0, 200.0, "#999")]
    gaze_df = _make_gaze_df(10, fps=5.0)  # only covers 0-2s
    pose_df = _make_pose_df(10, fps=5.0)
    tracks_df = _make_tracks_df(10, fps=5.0)

    results = compute_per_block_gaze_metrics(
        gaze_df, pose_df, tracks_df, np.array([]), [], blocks
    )

    assert results["categories_df"].empty
    assert results["convergence_df"].empty


def test_compute_gaze_metrics_no_heatmaps():
    """Metrics work without heatmaps (categories only)."""
    blocks = [SessionBlock("test", 0.0, 10.0, "#999")]
    gaze_df = _make_gaze_df(50, fps=5.0)
    pose_df = _make_pose_df(50, fps=5.0)
    tracks_df = _make_tracks_df(50, fps=5.0)

    results = compute_per_block_gaze_metrics(
        gaze_df, pose_df, tracks_df, np.array([]), [], blocks
    )

    assert not results["categories_df"].empty
    assert results["convergence_df"].empty
