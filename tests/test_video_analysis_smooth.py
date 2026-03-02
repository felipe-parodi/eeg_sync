"""Tests for temporal EMA smoothing and confidence-gated keypoint infill."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analysis.temporal_smooth import (  # noqa: E402
    SmoothingConfig,
    infill_low_confidence_keypoints,
    smooth_camera_outputs,
    smooth_pose_3d,
    smooth_tracks_2d,
)

# tau that gives effective_alpha ≈ 0.5 when dt = 1.0s:
#   alpha = 1 - exp(-dt / tau) = 0.5  =>  tau = -1 / ln(0.5) ≈ 1.4427
_TAU_HALF = -1.0 / np.log(0.5)


def _make_pose_df(
    track_id: int = 0,
    track_label: str = "parent",
    keypoint_name: str = "kp_000",
    x_values: list[float] | None = None,
    y_values: list[float] | None = None,
    z_values: list[float] | None = None,
    confidences: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal pose_3d DataFrame for one keypoint series."""
    n = len(x_values) if x_values is not None else 5
    x_values = x_values or [float(i) for i in range(n)]
    y_values = y_values or [float(i) for i in range(n)]
    z_values = z_values or [0.0] * n
    confidences = confidences or [0.9] * n
    return pd.DataFrame(
        {
            "frame_idx": list(range(n)),
            "timestamp_s": [float(i) for i in range(n)],
            "track_id": [track_id] * n,
            "track_label": [track_label] * n,
            "keypoint_name": [keypoint_name] * n,
            "x_m": x_values,
            "y_m": y_values,
            "z_m": z_values,
            "keypoint_confidence": confidences,
        }
    )


def _make_tracks_df(
    track_id: int = 0,
    track_label: str = "parent",
    x1_values: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal tracks_2d DataFrame for one track series."""
    n = len(x1_values) if x1_values is not None else 5
    x1_values = x1_values or [float(i) for i in range(n)]
    return pd.DataFrame(
        {
            "frame_idx": list(range(n)),
            "timestamp_s": [float(i) for i in range(n)],
            "track_id": [track_id] * n,
            "track_label": [track_label] * n,
            "bbox_x1": x1_values,
            "bbox_y1": [10.0] * n,
            "bbox_x2": [100.0] * n,
            "bbox_y2": [200.0] * n,
            "track_confidence": [0.9] * n,
        }
    )


# --- EMA smoothing: pose_3d ---


def test_ema_constant_series():
    """Constant input should produce identical output regardless of tau."""
    df = _make_pose_df(x_values=[5.0, 5.0, 5.0, 5.0, 5.0])
    result = smooth_pose_3d(df, tau=0.5)
    np.testing.assert_array_almost_equal(result["x_m"].values, [5.0] * 5)


def test_ema_step_change():
    """Verify bidirectional EMA on a step input: 0,0,0,10,10 at dt=1s."""
    df = _make_pose_df(x_values=[0.0, 0.0, 0.0, 10.0, 10.0])
    result = smooth_pose_3d(df, tau=_TAU_HALF)
    # Bidirectional EMA (fwd+bwd averaged) with alpha≈0.5 at dt=1s:
    expected = [0.625, 1.25, 2.5, 7.5, 8.75]
    np.testing.assert_array_almost_equal(result["x_m"].values, expected, decimal=2)


def test_ema_tau_zero_passthrough():
    """tau=0 means no smoothing (alpha=1.0, full weight on current value)."""
    raw = [1.0, 5.0, 3.0, 8.0, 2.0]
    df = _make_pose_df(x_values=raw)
    result = smooth_pose_3d(df, tau=0.0)
    np.testing.assert_array_almost_equal(result["x_m"].values, raw)


def test_ema_does_not_smooth_confidence():
    """Confidence column should pass through unchanged."""
    confs = [0.9, 0.3, 0.8, 0.1, 0.7]
    df = _make_pose_df(confidences=confs)
    result = smooth_pose_3d(df, tau=0.5)
    np.testing.assert_array_almost_equal(
        result["keypoint_confidence"].values, confs
    )


def test_ema_per_track_independence():
    """Smoothing track 0 should not affect track 1."""
    df0 = _make_pose_df(track_id=0, x_values=[0.0, 0.0, 10.0])
    df1 = _make_pose_df(track_id=1, x_values=[100.0, 100.0, 100.0])
    combined = pd.concat([df0, df1], ignore_index=True)
    result = smooth_pose_3d(combined, tau=_TAU_HALF)
    track1 = result[result["track_id"] == 1]
    np.testing.assert_array_almost_equal(
        track1["x_m"].values, [100.0, 100.0, 100.0]
    )


def test_ema_single_frame_passthrough():
    """Single-frame group should pass through unchanged."""
    df = _make_pose_df(x_values=[42.0])
    result = smooth_pose_3d(df, tau=0.5)
    assert result["x_m"].iloc[0] == 42.0


def test_smooth_preserves_schema_columns():
    """Output columns must exactly match input columns."""
    df = _make_pose_df()
    result = smooth_pose_3d(df, tau=0.5)
    assert list(result.columns) == list(df.columns)
    assert len(result) == len(df)


# --- EMA smoothing: tracks_2d ---


def test_ema_tracks_step_change():
    """Verify bidirectional EMA on tracks bbox column."""
    df = _make_tracks_df(x1_values=[0.0, 0.0, 10.0, 10.0])
    result = smooth_tracks_2d(df, tau=_TAU_HALF)
    # Bidirectional EMA: [1.25, 2.5, 7.5, 8.75]
    expected = [1.25, 2.5, 7.5, 8.75]
    np.testing.assert_array_almost_equal(result["bbox_x1"].values, expected, decimal=2)


def test_ema_tracks_does_not_smooth_confidence():
    """Track confidence should pass through unchanged."""
    df = _make_tracks_df()
    df["track_confidence"] = [0.9, 0.3, 0.8, 0.1, 0.7]
    result = smooth_tracks_2d(df, tau=0.5)
    np.testing.assert_array_almost_equal(
        result["track_confidence"].values, [0.9, 0.3, 0.8, 0.1, 0.7]
    )


# --- Confidence-gated infill ---


def test_infill_replaces_low_confidence():
    """Low-confidence keypoint should be replaced by interpolated neighbors."""
    df = _make_pose_df(
        x_values=[10.0, 999.0, 20.0],
        y_values=[10.0, 999.0, 20.0],
        confidences=[0.9, 0.05, 0.9],
    )
    result = infill_low_confidence_keypoints(df, conf_threshold=0.3)
    # Middle frame should be interpolated to 15.0
    assert abs(result["x_m"].iloc[1] - 15.0) < 0.01
    assert abs(result["y_m"].iloc[1] - 15.0) < 0.01


def test_infill_preserves_high_confidence():
    """High-confidence keypoints should not be modified."""
    df = _make_pose_df(
        x_values=[10.0, 15.0, 20.0],
        confidences=[0.9, 0.8, 0.9],
    )
    result = infill_low_confidence_keypoints(df, conf_threshold=0.3)
    np.testing.assert_array_almost_equal(result["x_m"].values, [10.0, 15.0, 20.0])


def test_infill_no_high_conf_neighbors_unchanged():
    """If all observations are low-confidence, leave as-is."""
    df = _make_pose_df(
        x_values=[1.0, 2.0, 3.0],
        confidences=[0.1, 0.1, 0.1],
    )
    result = infill_low_confidence_keypoints(df, conf_threshold=0.3)
    np.testing.assert_array_almost_equal(result["x_m"].values, [1.0, 2.0, 3.0])


def test_infill_caps_confidence_at_threshold():
    """Infilled rows should have confidence capped at the threshold value."""
    df = _make_pose_df(
        x_values=[10.0, 999.0, 20.0],
        confidences=[0.9, 0.05, 0.9],
    )
    result = infill_low_confidence_keypoints(df, conf_threshold=0.3)
    assert result["keypoint_confidence"].iloc[1] <= 0.3


def test_infill_per_track_independence():
    """Infill on track 0 should not use data from track 1."""
    df0 = _make_pose_df(
        track_id=0,
        x_values=[10.0, 999.0, 20.0],
        confidences=[0.9, 0.05, 0.9],
    )
    df1 = _make_pose_df(
        track_id=1,
        x_values=[100.0, 100.0, 100.0],
        confidences=[0.9, 0.9, 0.9],
    )
    combined = pd.concat([df0, df1], ignore_index=True)
    result = infill_low_confidence_keypoints(combined, conf_threshold=0.3)
    track0 = result[result["track_id"] == 0]
    assert abs(track0["x_m"].iloc[1] - 15.0) < 0.01


# --- Gap-aware EMA ---


def test_ema_resets_across_gap():
    """EMA should reset when timestamps have a gap > max_gap."""
    # Timestamps: 0, 1, 2, 10, 11 — gap of 8s between frame 2 and 3.
    df = _make_pose_df(x_values=[0.0, 0.0, 0.0, 10.0, 10.0])
    df["timestamp_s"] = [0.0, 1.0, 2.0, 10.0, 11.0]
    result = smooth_pose_3d(df, tau=_TAU_HALF, max_gap=2.0)
    # After the gap, EMA should reset: frame at t=10 should be 10.0, not blended.
    vals = result.sort_values("timestamp_s")["x_m"].values
    assert vals[3] == 10.0, f"Expected EMA reset at gap, got {vals[3]}"
    # Frame at t=11 should be EMA of 10 and 10 → 10.0.
    assert vals[4] == 10.0


def test_ema_no_reset_within_gap():
    """EMA should NOT reset when timestamps are within max_gap."""
    df = _make_pose_df(x_values=[0.0, 0.0, 10.0])
    df["timestamp_s"] = [0.0, 1.0, 2.0]
    result = smooth_pose_3d(df, tau=_TAU_HALF, max_gap=2.0)
    vals = result.sort_values("timestamp_s")["x_m"].values
    # Bidirectional: forward=[0, 0, 5], backward=[2.5, 5, 10], avg=[1.25, 2.5, 7.5]
    np.testing.assert_array_almost_equal(vals, [1.25, 2.5, 7.5], decimal=2)


def test_ema_bidirectional_no_lag():
    """Bidirectional EMA should center smoothing around the step, not lag."""
    # Symmetric step: 5 zeros then 5 tens.
    df = _make_pose_df(x_values=[0.0] * 5 + [10.0] * 5)
    df["timestamp_s"] = [float(i) for i in range(10)]
    result = smooth_pose_3d(df, tau=_TAU_HALF)
    vals = result.sort_values("timestamp_s")["x_m"].values
    # The midpoint (between index 4 and 5) should be ~5.0 (centered).
    mid = (vals[4] + vals[5]) / 2.0
    assert abs(mid - 5.0) < 0.1, f"Midpoint should be ~5.0, got {mid}"
    # Symmetry: vals[4] + vals[5] should sum to ~10.
    assert abs(vals[4] + vals[5] - 10.0) < 0.1


# --- Infill run-length limiting ---


def test_infill_skips_long_low_conf_runs():
    """Runs of low-confidence frames longer than max_infill_run are left as-is."""
    # 7 frames: high, low, low, low, low, low, high — run of 5 low-conf.
    df = _make_pose_df(
        x_values=[10.0, 999.0, 999.0, 999.0, 999.0, 999.0, 20.0],
        y_values=[10.0, 999.0, 999.0, 999.0, 999.0, 999.0, 20.0],
        confidences=[0.9, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9],
    )
    result = infill_low_confidence_keypoints(
        df, conf_threshold=0.3, max_infill_run=3
    )
    # Run length is 5 > max_infill_run=3, so all 999.0 values should remain.
    middle = result["x_m"].values[1:6]
    np.testing.assert_array_equal(middle, [999.0] * 5)


def test_infill_fills_short_low_conf_runs():
    """Runs of low-confidence frames <= max_infill_run ARE infilled."""
    # 5 frames: high, low, low, low, high — run of 3 (== max_infill_run).
    df = _make_pose_df(
        x_values=[10.0, 999.0, 999.0, 999.0, 20.0],
        y_values=[10.0, 999.0, 999.0, 999.0, 20.0],
        confidences=[0.9, 0.05, 0.05, 0.05, 0.9],
    )
    result = infill_low_confidence_keypoints(
        df, conf_threshold=0.3, max_infill_run=3
    )
    # Run length is 3 == max_infill_run, so values should be interpolated.
    middle = result["x_m"].values[1:4]
    # Linear interp from 10 to 20: 12.5, 15.0, 17.5
    np.testing.assert_array_almost_equal(middle, [12.5, 15.0, 17.5])


# --- CLI integration ---


def test_cli_writes_output_files(tmp_path: Path):
    """End-to-end CLI: reads interpolated CSVs, writes smooth CSVs + summary."""
    camera_dir = tmp_path / "camera_a"
    camera_dir.mkdir()

    # Create interpolated input files (what the interpolator would produce)
    pose = _make_pose_df(x_values=[0.0, 0.0, 10.0, 10.0, 10.0])
    tracks = _make_tracks_df(x1_values=[0.0, 0.0, 10.0, 10.0, 10.0])
    pose.to_csv(camera_dir / "pose_3d_interpolated.csv", index=False)
    tracks.to_csv(camera_dir / "tracks_2d_interpolated.csv", index=False)

    summary = smooth_camera_outputs(
        SmoothingConfig(
            camera_dir=str(camera_dir),
            tau=0.5,
            conf_threshold=0.3,
            conf_gate=0.3,
        )
    )

    assert (camera_dir / "pose_3d_smooth.csv").exists()
    assert (camera_dir / "tracks_2d_smooth.csv").exists()
    assert (camera_dir / "smoothing_summary.json").exists()

    summary_data = json.loads(
        (camera_dir / "smoothing_summary.json").read_text(encoding="utf-8")
    )
    assert summary_data["tau"] == 0.5
    assert summary_data["conf_threshold"] == 0.3
    assert summary_data["conf_gate"] == 0.3
    assert summary_data["pose_rows"] == 5
    assert summary_data["tracks_rows"] == 5


# --- Confidence-gated smoothing ---


def test_conf_gate_preserves_high_confidence_positions():
    """High-confidence rows should keep their raw positions when conf_gate > 0."""
    raw = [1.0, 5.0, 3.0, 8.0, 2.0]
    df = _make_pose_df(x_values=raw, confidences=[0.9, 0.9, 0.9, 0.9, 0.9])
    result = smooth_pose_3d(df, tau=_TAU_HALF, conf_gate=0.3)
    # All confidences >= 0.3, so all positions should be unchanged.
    np.testing.assert_array_almost_equal(result["x_m"].values, raw)


def test_conf_gate_smooths_low_confidence_positions():
    """Low-confidence rows should be smoothed when conf_gate > 0."""
    # Frame 2 has low confidence and a spike value.
    df = _make_pose_df(
        x_values=[10.0, 10.0, 50.0, 10.0, 10.0],
        confidences=[0.9, 0.9, 0.05, 0.9, 0.9],
    )
    result = smooth_pose_3d(df, tau=_TAU_HALF, conf_gate=0.3)
    vals = result["x_m"].values
    # High-confidence frames should be untouched.
    assert vals[0] == 10.0
    assert vals[1] == 10.0
    assert vals[3] == 10.0
    assert vals[4] == 10.0
    # Low-confidence frame should be smoothed (pulled toward neighbors).
    assert vals[2] < 50.0


def test_conf_gate_zero_disables_gating():
    """conf_gate=0 should smooth everything (backward compat)."""
    raw = [0.0, 0.0, 10.0, 10.0, 10.0]
    df = _make_pose_df(x_values=raw, confidences=[0.9, 0.9, 0.9, 0.9, 0.9])
    result = smooth_pose_3d(df, tau=_TAU_HALF, conf_gate=0.0)
    # With gating disabled, high-confidence values SHOULD be smoothed.
    # A step change from 0→10 with bidirectional EMA won't match raw.
    assert not np.allclose(result["x_m"].values, raw)
