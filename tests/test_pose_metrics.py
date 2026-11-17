"""Tests for per-block pose metrics computation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
                rows.append(
                    {
                        "frame_idx": f,
                        "timestamp_s": ts,
                        "track_id": tid,
                        "track_label": "parent" if tid == 0 else "child",
                        "keypoint_name": kp,
                        "x_m": cx + np.random.randn() * 2,
                        "y_m": 300.0 + np.random.randn() * 2,
                        "z_m": 0.0,
                        "keypoint_confidence": 0.9,
                    }
                )
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


def test_block_stats_report_tracking_coverage() -> None:
    """Each block reports how much of it had parent, child, and both."""
    pose = _make_pose_df(n_frames=100, fps=10.0)  # 0.0 .. 9.9 s
    # Child missing for the second half of the block.
    pose = pose[~((pose.track_id == 1) & (pose.timestamp_s >= 5.0))]
    blocks = [SessionBlock("task", 0.0, 9.9), SessionBlock("empty", 50.0, 60.0)]

    stats = {
        s["block"]: s
        for s in compute_per_block_metrics(pose, blocks, frame_rate=10.0)["block_stats"]
    }

    task = stats["task"]
    assert task["block_frames"] == 100
    assert task["frames_with_parent"] == 100
    assert task["frames_with_child"] == 50
    assert task["frames_with_both"] == 50
    assert task["coverage_both"] == 0.5
    assert task["low_coverage"] is False
    # A block with no tracking at all is still reported, flagged.
    assert stats["empty"]["frames_with_both"] == 0
    assert stats["empty"]["low_coverage"] is True


def test_block_stats_include_synchrony_chance_baseline():
    """Each block reports the shifted-child baseline and the difference."""
    blocks = [SessionBlock("b", 0.0, 60.0, "#999")]
    pose_df = _make_pose_df(6 * 60, fps=6.0)
    stats = compute_per_block_metrics(pose_df, blocks, frame_rate=6.0)["block_stats"][0]

    assert "xcorr_chance_mean" in stats
    assert stats["xcorr_minus_chance"] == pytest.approx(
        stats["xcorr_mean"] - stats["xcorr_chance_mean"]
    )


def test_block_stats_include_body_synchrony():
    """Blocks report whole-body synchrony, its chance level, and the difference."""
    blocks = [SessionBlock("b", 0.0, 120.0, "#999")]
    pose_df = _make_pose_df(6 * 120, fps=6.0)
    stats = compute_per_block_metrics(pose_df, blocks, frame_rate=6.0)["block_stats"][0]

    for key in ("sync_mean", "sync_chance_mean", "sync_minus_chance", "sync_n_windows"):
        assert key in stats
    assert stats["sync_minus_chance"] == pytest.approx(
        stats["sync_mean"] - stats["sync_chance_mean"]
    )


# ---- Valid time and quality verdict (metrics-validity plan, phase 4) ----


def _block_pose(times_both, times_parent_only=(), fps=5.0):
    rows = []
    for tid, times in (
        (0, list(times_both) + list(times_parent_only)),
        (1, list(times_both)),
    ):
        for t in times:
            for kp in ["kp_005", "kp_006", "kp_011", "kp_012"]:
                rows.append(
                    {
                        "frame_idx": int(round(t * fps)),
                        "timestamp_s": t,
                        "track_id": tid,
                        "track_label": "p",
                        "keypoint_name": kp,
                        "x_m": 100.0 * tid,
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "keypoint_confidence": 0.9,
                    }
                )
    return pd.DataFrame(rows)


def test_block_coverage_reports_valid_seconds_and_segments():
    from video_analysis.pose_metrics import block_coverage

    both = list(np.round(np.arange(0, 30, 0.2), 6)) + list(
        np.round(np.arange(50, 60, 0.2), 6)
    )
    pose = _block_pose(both)
    cov = block_coverage(pose, SessionBlock("b", 0.0, 100.0, "#999"), 5.0)

    assert cov["valid_seconds_both"] == pytest.approx(len(both) / 5.0)
    assert cov["n_segments"] == 2
    assert cov["longest_segment_s"] == pytest.approx(30.0)


def test_quality_verdict_levels_and_reasons():
    from video_analysis.pose_metrics import quality_verdict

    good = {"valid_seconds_both": 400.0, "coverage_both": 0.9, "sync_n_windows": 20}
    assert quality_verdict(good) == ("usable", "")

    low = {"valid_seconds_both": 400.0, "coverage_both": 0.3, "sync_n_windows": 20}
    verdict, reasons = quality_verdict(low)
    assert verdict == "check" and "30%" in reasons

    few = {"valid_seconds_both": 400.0, "coverage_both": 0.9, "sync_n_windows": 1}
    verdict, reasons = quality_verdict(few)
    assert verdict == "check" and "synchrony" in reasons

    tiny = {"valid_seconds_both": 20.0, "coverage_both": 0.1, "sync_n_windows": 0}
    verdict, reasons = quality_verdict(tiny)
    assert verdict == "insufficient" and "20 s" in reasons


def test_block_stats_include_quality_verdict():
    blocks = [SessionBlock("b", 0.0, 120.0, "#999")]
    pose_df = _make_pose_df(6 * 120, fps=6.0)
    stats = compute_per_block_metrics(pose_df, blocks, frame_rate=6.0)["block_stats"][0]

    for key in (
        "valid_seconds_both",
        "n_segments",
        "longest_segment_s",
        "quality_verdict",
        "quality_reasons",
    ):
        assert key in stats
    assert stats["quality_verdict"] in {"usable", "check", "insufficient"}
