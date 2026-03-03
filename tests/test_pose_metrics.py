"""Tests for per-block pose metrics computation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaze_analysis.config import SessionBlock
from video_analysis.pose_metrics import compute_per_block_metrics


def _make_pose_df(
    n_frames: int = 100,
    fps: float = 30.0,
    parent_x: float = 300.0,
    child_x: float = 500.0,
) -> pd.DataFrame:
    """Create synthetic pose data with 2 tracks and COCO17 torso keypoints."""
    rows = []
    kp_names = ["kp_005", "kp_006", "kp_011", "kp_012"]  # torso
    for f in range(n_frames):
        ts = f / fps
        for tid, cx in [(0, parent_x), (1, child_x)]:
            for kp in kp_names:
                rows.append({
                    "frame_idx": f,
                    "timestamp_s": ts,
                    "track_id": tid,
                    "track_label": "parent" if tid == 0 else "child",
                    "keypoint_name": kp,
                    "x_m": cx + np.random.randn() * 2,
                    "y_m": 300.0 + np.random.randn() * 2,
                    "z_m": 0.0,
                    "keypoint_confidence": 0.9,
                })
    return pd.DataFrame(rows)


def test_compute_per_block_basic():
    """Metrics computed for a single block."""
    blocks = [SessionBlock("test_block", 0.0, 3.0, "#999")]
    pose_df = _make_pose_df(100, fps=30.0, parent_x=300.0, child_x=500.0)
    results = compute_per_block_metrics(pose_df, blocks)

    assert not results["proximity_df"].empty
    assert "block" in results["proximity_df"].columns
    assert results["proximity_df"]["block"].iloc[0] == "test_block"

    # Distance should be ~200px (500 - 300)
    mean_dist = results["proximity_df"]["torso_distance_px"].mean()
    assert 180.0 < mean_dist < 220.0


def test_compute_per_block_xcorr():
    """Cross-correlation is computed when enough frames."""
    blocks = [SessionBlock("test_block", 0.0, 10.0, "#999")]
    pose_df = _make_pose_df(300, fps=30.0)
    results = compute_per_block_metrics(pose_df, blocks)

    assert not results["xcorr_df"].empty
    assert "peak_xcorr" in results["xcorr_df"].columns
    assert "peak_lag_s" in results["xcorr_df"].columns


def test_compute_per_block_multiple_blocks():
    """Metrics separated per block."""
    blocks = [
        SessionBlock("block_a", 0.0, 1.5, "#999"),
        SessionBlock("block_b", 2.0, 3.3, "#999"),
    ]
    pose_df = _make_pose_df(100, fps=30.0)
    results = compute_per_block_metrics(pose_df, blocks)

    block_names = results["proximity_df"]["block"].unique()
    assert "block_a" in block_names
    assert "block_b" in block_names
    assert len(results["block_stats"]) == 2


def test_compute_per_block_stats():
    """Block stats include expected keys."""
    blocks = [SessionBlock("test", 0.0, 3.0, "#999")]
    pose_df = _make_pose_df(100, fps=30.0)
    results = compute_per_block_metrics(pose_df, blocks)

    stats = results["block_stats"][0]
    assert stats["block"] == "test"
    assert "proximity_mean" in stats
    assert "proximity_std" in stats
    assert "proximity_n_frames" in stats


def test_compute_per_block_empty():
    """Empty pose data returns empty results."""
    blocks = [SessionBlock("test", 100.0, 200.0, "#999")]
    pose_df = _make_pose_df(10, fps=30.0)  # only covers 0-0.3s
    results = compute_per_block_metrics(pose_df, blocks)

    assert results["proximity_df"].empty
    assert results["xcorr_df"].empty
