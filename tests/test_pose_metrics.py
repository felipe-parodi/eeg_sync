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


def test_verdict_thresholds_match_lab_recommendation():
    """Usable needs >= 2 min with both people tracked and >= 10 synchrony
    windows (from the 2026-11-16 reliability check on 1011-1020)."""
    from video_analysis.pose_metrics import quality_verdict

    assert (
        quality_verdict(
            {"valid_seconds_both": 100.0, "coverage_both": 0.9, "sync_n_windows": 20}
        )[0]
        == "insufficient"
    )
    assert (
        quality_verdict(
            {"valid_seconds_both": 300.0, "coverage_both": 0.9, "sync_n_windows": 5}
        )[0]
        == "check"
    )
    assert (
        quality_verdict(
            {"valid_seconds_both": 300.0, "coverage_both": 0.9, "sync_n_windows": 10}
        )[0]
        == "usable"
    )


def _upright_pose(
    n_frames: int,
    parent_x: float,
    child_x: float,
    parent_torso: float,
    child_torso: float,
) -> pd.DataFrame:
    """Two people with shoulders ``torso`` px above their hips (no noise)."""
    rows = []
    for f in range(n_frames):
        for tid, label, cx, torso in (
            (0, "parent", parent_x, parent_torso),
            (1, "child", child_x, child_torso),
        ):
            for kp, dx, y in (
                ("kp_005", -10.0, 300.0 - torso),
                ("kp_006", 10.0, 300.0 - torso),
                ("kp_011", -8.0, 300.0),
                ("kp_012", 8.0, 300.0),
            ):
                rows.append(
                    {
                        "frame_idx": f,
                        "timestamp_s": f / 5.0,
                        "track_id": tid,
                        "track_label": label,
                        "keypoint_name": kp,
                        "x_m": cx + dx,
                        "y_m": y,
                        "z_m": 0.0,
                        "keypoint_confidence": 0.9,
                    }
                )
    return pd.DataFrame(rows)


def test_proximity_frames_carry_torso_lengths():
    from gaze_analysis.synchrony import compute_torso_proximity

    pose = _upright_pose(10, 300.0, 500.0, parent_torso=80.0, child_torso=40.0)
    prox = compute_torso_proximity(pose, 0, 1)

    assert np.allclose(prox["parent_torso_px"], 80.0)
    assert np.allclose(prox["child_torso_px"], 40.0)


def test_block_stats_include_body_scaled_proximity():
    """Distance 300 px with torsos 80 and 40 px: scale 60 px, scaled distance 5."""
    pose = _upright_pose(50, 300.0, 600.0, parent_torso=80.0, child_torso=40.0)
    blocks = [SessionBlock("grocery", 0.0, 30.0, "#999")]

    stats = compute_per_block_metrics(pose, blocks, frame_rate=5.0)["block_stats"][0]

    assert stats["body_scale_px"] == pytest.approx(60.0)
    # Torso centroids sit half a torso above the hips, so they differ in height.
    expected = np.hypot(300.0, 40.0 - 20.0) / 60.0
    assert stats["proximity_scaled_median"] == pytest.approx(expected)
    assert stats["proximity_scaled_mean"] == pytest.approx(expected)


def test_scaled_proximity_is_independent_of_zoom():
    """The same scene filmed twice as large gives the same scaled distance."""
    near = _upright_pose(60, 300.0, 600.0, parent_torso=80.0, child_torso=40.0)
    far = near.assign(x_m=near["x_m"] / 2, y_m=near["y_m"] / 2)
    blocks = [SessionBlock("grocery", 0.0, 10.0, "#999")]

    s_near = compute_per_block_metrics(near, blocks, frame_rate=5.0)["block_stats"][0]
    s_far = compute_per_block_metrics(far, blocks, frame_rate=5.0)["block_stats"][0]

    assert s_far["proximity_median"] == pytest.approx(s_near["proximity_median"] / 2)
    assert s_far["proximity_scaled_median"] == pytest.approx(
        s_near["proximity_scaled_median"]
    )


def test_plot_synchrony_draws_new_measure(tmp_path):
    from video_analysis.pose_metrics import plot_synchrony

    sync_df = pd.DataFrame(
        {
            "segment_id": 0,
            "window_start_s": np.arange(14) * 10.0,
            "window_end_s": np.arange(14) * 10.0 + 20.0,
            "sync_r": np.linspace(0.1, 0.5, 14),
            "block": ["grocery"] * 12 + ["storybook"] * 2,
        }
    )
    stats = [
        {
            "block": "grocery",
            "sync_mean": 0.3,
            "sync_chance_mean": 0.0,
            "sync_n_windows": 12,
        },
        {
            "block": "storybook",
            "sync_mean": 0.45,
            "sync_chance_mean": 0.0,
            "sync_n_windows": 2,
        },
    ]
    blocks = [
        SessionBlock("grocery", 0.0, 120.0, "#999"),
        SessionBlock("storybook", 120.0, 200.0, "#999"),
    ]

    plot_synchrony(sync_df, stats, blocks, tmp_path / "pose_synchrony")

    assert (tmp_path / "pose_synchrony.png").stat().st_size > 0
    assert (tmp_path / "pose_synchrony.svg").exists()


def test_quality_verdict_fails_closed_on_missing_support():
    from video_analysis.pose_metrics import quality_verdict

    for bad in (float("nan"), float("inf"), None):
        verdict, reasons = quality_verdict(
            {"valid_seconds_both": bad, "coverage_both": bad, "sync_n_windows": 10}
        )
        assert verdict == "insufficient", bad
        assert reasons


def test_ruler_ignores_low_confidence_torso_joints():
    """A mis-detected hip (confidence 0.1) on half the frames must not move the ruler."""
    pose = _upright_pose(120, 300.0, 600.0, parent_torso=80.0, child_torso=40.0)
    bad = (
        (pose["track_id"] == 0)
        & (pose["keypoint_name"] == "kp_011")
        & (pose["frame_idx"] < 60)
    )
    pose.loc[bad, "y_m"] += 200.0
    pose.loc[bad, "keypoint_confidence"] = 0.1
    blocks = [SessionBlock("grocery", 0.0, 30.0, "#999")]

    stats = compute_per_block_metrics(pose, blocks, frame_rate=5.0)["block_stats"][0]

    assert stats["body_scale_px"] == pytest.approx(60.0)


def test_save_sync_outputs_replaces_stale_files_when_no_windows(tmp_path):
    from video_analysis.pose_metrics import save_sync_outputs

    (tmp_path / "pose_metrics_sync.csv").write_text("old", encoding="utf-8")
    (tmp_path / "pose_synchrony.png").write_bytes(b"old")
    (tmp_path / "pose_synchrony.svg").write_bytes(b"old")
    blocks = [SessionBlock("grocery", 0.0, 10.0, "#999")]

    save_sync_outputs(tmp_path, pd.DataFrame(), [{"block": "grocery"}], blocks)

    written = pd.read_csv(tmp_path / "pose_metrics_sync.csv")
    assert written.empty
    assert {"window_start_s", "window_end_s", "sync_r", "block"} <= set(written.columns)
    assert not (tmp_path / "pose_synchrony.png").exists()
    assert not (tmp_path / "pose_synchrony.svg").exists()


def test_block_stats_record_chance_shifts_and_version():
    from video_analysis.pose_metrics import METRICS_VERSION

    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.normal(0, 4, 1500))
    pose = _upright_pose(1500, 300.0, 600.0, parent_torso=80.0, child_torso=40.0)
    pose["x_m"] = pose["x_m"] + walk[pose["frame_idx"].to_numpy()]
    blocks = [SessionBlock("grocery", 0.0, 300.0, "#999")]

    stats = compute_per_block_metrics(pose, blocks, frame_rate=5.0)["block_stats"][0]

    assert stats["metrics_version"] == METRICS_VERSION
    assert stats["sync_chance_n_shifts"] >= 3
    assert "sync_chance_sd" in stats


def test_ruler_needs_enough_confident_frames():
    """Fewer than MIN_RULER_FRAMES confident torsos: no normalized distance."""
    from video_analysis.pose_metrics import MIN_RULER_FRAMES

    pose = _upright_pose(MIN_RULER_FRAMES - 1, 300.0, 600.0, 80.0, 40.0)
    blocks = [SessionBlock("grocery", 0.0, 100.0, "#999")]

    stats = compute_per_block_metrics(pose, blocks, frame_rate=5.0)["block_stats"][0]

    assert "proximity_median" in stats
    assert "proximity_scaled_median" not in stats
