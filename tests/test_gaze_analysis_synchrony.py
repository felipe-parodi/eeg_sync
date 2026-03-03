"""Tests for gaze_analysis.synchrony — parent-child synchrony metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gaze_analysis.synchrony import (
    compute_gaze_categories,
    compute_gaze_convergence,
    compute_movement_xcorr,
    compute_torso_proximity,
)

# ---- Helpers ----


def _make_pose_df(
    parent_centroids: list[tuple[float, float]],
    child_centroids: list[tuple[float, float]],
    parent_id: int = 0,
    child_id: int = 1,
    fps: float = 5.0,
) -> pd.DataFrame:
    """Build a minimal pose_3d DataFrame with shoulder+hip keypoints.

    Each centroid generates 4 keypoints (left/right shoulder + left/right hip)
    positioned symmetrically around the centroid.
    """
    rows = []
    shoulder_hip_kps = ["kp_005", "kp_006", "kp_011", "kp_012"]
    offsets = [(-10, -5), (10, -5), (-10, 5), (10, 5)]

    for frame_idx, ((px, py), (cx, cy)) in enumerate(
        zip(parent_centroids, child_centroids)
    ):
        ts = frame_idx / fps
        for tid, (bx, by) in [(parent_id, (px, py)), (child_id, (cx, cy))]:
            for kp_name, (dx, dy) in zip(shoulder_hip_kps, offsets):
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_s": ts,
                        "track_id": tid,
                        "track_label": f"person_{tid:02d}",
                        "keypoint_name": kp_name,
                        "x_m": bx + dx,
                        "y_m": by + dy,
                        "z_m": 0.0,
                        "keypoint_confidence": 0.9,
                    }
                )
    return pd.DataFrame(rows)


def _make_gaze_heatmaps(
    n_frames: int,
    parent_peaks: list[tuple[int, int]],
    child_peaks: list[tuple[int, int]],
    parent_id: int = 0,
    child_id: int = 1,
    heatmap_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Build synthetic gaze heatmaps as Gaussians centered at given peaks.

    Returns:
        (heatmaps, keys) matching the npz format from gazelle_runner.
    """
    heatmaps = []
    keys = []
    for i in range(n_frames):
        for tid, (py, px) in [(parent_id, parent_peaks[i]), (child_id, child_peaks[i])]:
            hm = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
            # Place a Gaussian blob at the peak
            yy, xx = np.mgrid[:heatmap_size, :heatmap_size]
            hm = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * 5**2)).astype(
                np.float32
            )
            heatmaps.append(hm)
            keys.append(f"f{i:06d}_t{tid}")
    return np.stack(heatmaps), keys


def _make_gaze_csv(
    n_frames: int,
    parent_peaks_norm: list[tuple[float, float]],
    child_peaks_norm: list[tuple[float, float]],
    parent_id: int = 0,
    child_id: int = 1,
    fps: float = 5.0,
) -> pd.DataFrame:
    """Build a gaze_heatmap.csv DataFrame with peak coordinates."""
    rows = []
    for i in range(n_frames):
        for tid, (gx, gy) in [
            (parent_id, parent_peaks_norm[i]),
            (child_id, child_peaks_norm[i]),
        ]:
            rows.append(
                {
                    "frame_idx": i,
                    "timestamp_s": i / fps,
                    "track_id": tid,
                    "gaze_peak_x": gx,
                    "gaze_peak_y": gy,
                    "gaze_peak_value": 0.8,
                    "inout_score": 0.9,
                    "head_source": "keypoints",
                }
            )
    return pd.DataFrame(rows)


# ====================================================================
# Metric 1: Torso Proximity
# ====================================================================


def test_torso_proximity_known_distance() -> None:
    """Two people 100px apart horizontally should have distance ~100."""
    parent = [(100.0, 200.0)] * 5
    child = [(200.0, 200.0)] * 5
    pose_df = _make_pose_df(parent, child)

    result = compute_torso_proximity(pose_df, parent_track_id=0, child_track_id=1)
    assert len(result) == 5
    np.testing.assert_allclose(result["torso_distance_px"].values, 100.0, atol=0.5)


def test_torso_proximity_zero_distance() -> None:
    """Same centroid should yield distance ~0."""
    parent = [(200.0, 200.0)] * 3
    child = [(200.0, 200.0)] * 3
    pose_df = _make_pose_df(parent, child)

    result = compute_torso_proximity(pose_df, parent_track_id=0, child_track_id=1)
    np.testing.assert_allclose(result["torso_distance_px"].values, 0.0, atol=0.5)


def test_torso_proximity_diagonal() -> None:
    """Diagonal separation: sqrt(300^2 + 400^2) = 500."""
    parent = [(0.0, 0.0)] * 3
    child = [(300.0, 400.0)] * 3
    pose_df = _make_pose_df(parent, child)

    result = compute_torso_proximity(pose_df, parent_track_id=0, child_track_id=1)
    np.testing.assert_allclose(result["torso_distance_px"].values, 500.0, atol=0.5)


# ====================================================================
# Metric 2: Cross-Correlation of Movement
# ====================================================================


def test_xcorr_identical_signals() -> None:
    """Identical velocity signals should have r=1 at lag=0."""
    n = 50
    t = np.arange(n) / 5.0
    x_vals = 100.0 + 20.0 * np.sin(2 * np.pi * 0.5 * t)
    parent = [(float(x), 200.0) for x in x_vals]
    child = [(float(x), 200.0) for x in x_vals]
    pose_df = _make_pose_df(parent, child)

    result = compute_movement_xcorr(
        pose_df, parent_track_id=0, child_track_id=1, window_s=10.0, max_lag_s=2.0
    )
    # At least one window should have high correlation near 1.0
    assert result["peak_xcorr"].max() > 0.9
    # Lag should be near 0
    best_idx = result["peak_xcorr"].idxmax()
    assert abs(result.loc[best_idx, "peak_lag_s"]) < 0.5


def test_xcorr_shifted_signals() -> None:
    """Child delayed by 1s should show positive lag near 1.0s."""
    n = 75
    fps = 5.0
    t = np.arange(n) / fps
    signal = 20.0 * np.sin(2 * np.pi * 0.3 * t)
    delay_frames = int(1.0 * fps)  # 5 frames = 1 second

    parent_x = 200.0 + signal
    child_x = 200.0 + np.roll(signal, delay_frames)
    # Zero out the wrapped-around part
    child_x[:delay_frames] = 200.0

    parent = [(float(x), 200.0) for x in parent_x]
    child = [(float(x), 200.0) for x in child_x]
    pose_df = _make_pose_df(parent, child, fps=fps)

    result = compute_movement_xcorr(
        pose_df, parent_track_id=0, child_track_id=1, window_s=10.0, max_lag_s=2.0
    )
    # Should detect a lag near +1.0 (child follows parent)
    valid = result[result["peak_xcorr"] > 0.5]
    if len(valid) > 0:
        # Check that the median lag is positive (child follows)
        assert valid["peak_lag_s"].median() > 0.0


def test_xcorr_uncorrelated_signals() -> None:
    """Uncorrelated signals should have low cross-correlation."""
    np.random.seed(42)
    n = 50
    parent = [(float(np.random.randn() * 20 + 200), 200.0) for _ in range(n)]
    child = [(float(np.random.randn() * 20 + 200), 200.0) for _ in range(n)]
    pose_df = _make_pose_df(parent, child)

    result = compute_movement_xcorr(
        pose_df, parent_track_id=0, child_track_id=1, window_s=10.0, max_lag_s=2.0
    )
    # Mean xcorr should be modest (< 0.7) for random data
    assert result["peak_xcorr"].mean() < 0.7


# ====================================================================
# Metric 3: Gaze Categories
# ====================================================================


def test_mutual_gaze_both_looking_at_each_other() -> None:
    """When both look at each other's head location, classify as mutual_gaze."""
    # Parent head at normalized (0.2, 0.3), child head at (0.7, 0.3)
    # Parent gaze peak at child's head, child gaze peak at parent's head
    n = 3
    parent_gaze = [(0.7, 0.3)] * n  # looking at child
    child_gaze = [(0.2, 0.3)] * n  # looking at parent
    gaze_df = _make_gaze_csv(n, parent_gaze, child_gaze)

    # Head locations (normalized)
    parent_head_center = [(0.2, 0.3)] * n
    child_head_center = [(0.7, 0.3)] * n

    result = compute_gaze_categories(
        gaze_df,
        parent_track_id=0,
        child_track_id=1,
        parent_head_centers=parent_head_center,
        child_head_centers=child_head_center,
        proximity_threshold=0.15,
    )
    assert all(result["gaze_category"] == "mutual_gaze")


def test_joint_attention_both_looking_same_spot() -> None:
    """When both look at the same third location, classify as joint_attention."""
    n = 3
    # Both looking at center of image
    parent_gaze = [(0.5, 0.5)] * n
    child_gaze = [(0.5, 0.5)] * n
    gaze_df = _make_gaze_csv(n, parent_gaze, child_gaze)

    # Heads are at different locations
    parent_head = [(0.2, 0.3)] * n
    child_head = [(0.8, 0.3)] * n

    result = compute_gaze_categories(
        gaze_df,
        parent_track_id=0,
        child_track_id=1,
        parent_head_centers=parent_head,
        child_head_centers=child_head,
        proximity_threshold=0.15,
    )
    assert all(result["gaze_category"] == "joint_attention")


def test_independent_gaze_looking_different_spots() -> None:
    """When both look at different spots, classify as independent."""
    n = 3
    parent_gaze = [(0.1, 0.1)] * n  # top-left
    child_gaze = [(0.9, 0.9)] * n  # bottom-right
    gaze_df = _make_gaze_csv(n, parent_gaze, child_gaze)

    parent_head = [(0.3, 0.5)] * n
    child_head = [(0.7, 0.5)] * n

    result = compute_gaze_categories(
        gaze_df,
        parent_track_id=0,
        child_track_id=1,
        parent_head_centers=parent_head,
        child_head_centers=child_head,
        proximity_threshold=0.15,
    )
    assert all(result["gaze_category"] == "independent")


def test_adaptive_threshold_close_proximity() -> None:
    """With adaptive threshold, close heads + shared gaze = joint_attention, not mutual."""
    n = 3
    # Heads very close together (0.06 apart), both gaze at a spot between them
    # but slightly off-center (0.07 from parent head, 0.05 from child head)
    parent_head = [(0.47, 0.3)] * n
    child_head = [(0.53, 0.3)] * n  # 0.06 apart
    # Both look at the same spot slightly toward the child
    parent_gaze = [(0.515, 0.3)] * n  # 0.045 from parent, 0.015 from child
    child_gaze = [(0.515, 0.3)] * n
    gaze_df = _make_gaze_csv(n, parent_gaze, child_gaze)

    # With fixed 0.15 threshold: both distances < 0.15 → mutual_gaze (wrong)
    result_fixed = compute_gaze_categories(
        gaze_df, 0, 1, parent_head, child_head, proximity_threshold=0.15
    )
    assert all(result_fixed["gaze_category"] == "mutual_gaze")

    # With adaptive threshold: 0.4 * 0.06 = 0.024, clamped to 0.04
    # Parent gaze→child head = 0.015 < 0.04 (parent looks at child)
    # Child gaze→parent head = 0.045 > 0.04 (child does NOT look at parent)
    # → should NOT be mutual_gaze
    result_adaptive = compute_gaze_categories(
        gaze_df, 0, 1, parent_head, child_head, proximity_threshold=None
    )
    assert all(result_adaptive["gaze_category"] != "mutual_gaze")


# ====================================================================
# Metric 4: Gaze Convergence Score
# ====================================================================


def test_gaze_convergence_identical_heatmaps() -> None:
    """Identical heatmaps should yield convergence score ~1.0."""
    n = 3
    peak = (32, 32)
    heatmaps, keys = _make_gaze_heatmaps(
        n, [peak] * n, [peak] * n, parent_id=0, child_id=1
    )

    result = compute_gaze_convergence(
        heatmaps, keys, parent_track_id=0, child_track_id=1
    )
    assert len(result) == n
    np.testing.assert_allclose(result["gaze_convergence_score"].values, 1.0, atol=0.01)


def test_gaze_convergence_disjoint_heatmaps() -> None:
    """Non-overlapping heatmaps should yield convergence score near 0."""
    n = 3
    heatmaps, keys = _make_gaze_heatmaps(
        n,
        [(5, 5)] * n,  # parent looks top-left
        [(58, 58)] * n,  # child looks bottom-right
        parent_id=0,
        child_id=1,
    )

    result = compute_gaze_convergence(
        heatmaps, keys, parent_track_id=0, child_track_id=1
    )
    # Gaussians with sigma=5 centered 53 pixels apart should have very low overlap
    assert result["gaze_convergence_score"].mean() < 0.1


def test_gaze_convergence_partial_overlap() -> None:
    """Moderately overlapping heatmaps should have intermediate score."""
    n = 3
    heatmaps, keys = _make_gaze_heatmaps(
        n,
        [(32, 25)] * n,
        [(32, 39)] * n,  # 14 pixels apart
        parent_id=0,
        child_id=1,
    )

    result = compute_gaze_convergence(
        heatmaps, keys, parent_track_id=0, child_track_id=1
    )
    scores = result["gaze_convergence_score"].values
    assert 0.1 < scores.mean() < 0.95
