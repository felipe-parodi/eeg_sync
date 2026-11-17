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


def _head_center_dict(
    n: int, value: tuple[float, float]
) -> dict[int, tuple[float, float]]:
    """Build a ``{frame_idx: (x, y)}`` dict for n consecutive frames."""
    return {i: value for i in range(n)}


W, H = 960, 540


def _heads_and_tracks(frames, parent_head, child_head, missing_child_head=()):
    """Normalized head boxes and pixel body boxes for parent (0) and child (1)."""
    heads, tracks = [], []
    for f in frames:
        for tid, (hx, hy) in ((0, parent_head), (1, child_head)):
            if tid == 1 and f in missing_child_head:
                continue
            heads.append(
                {
                    "frame_idx": f,
                    "track_id": tid,
                    "head_x1": hx - 0.03,
                    "head_y1": hy - 0.05,
                    "head_x2": hx + 0.03,
                    "head_y2": hy + 0.05,
                }
            )
        for tid, (hx, hy) in ((0, parent_head), (1, child_head)):
            tracks.append(
                {
                    "frame_idx": f,
                    "track_id": tid,
                    "bbox_x1": (hx - 0.06) * W,
                    "bbox_y1": (hy - 0.06) * H,
                    "bbox_x2": (hx + 0.06) * W,
                    "bbox_y2": (hy + 0.5) * H,
                }
            )
    return pd.DataFrame(heads), pd.DataFrame(tracks)


def _categories(parent_gaze, child_gaze, parent_head, child_head, n=3, **kw):
    gaze_df = _make_gaze_csv(n, parent_gaze, child_gaze)
    gaze_df["inout_score"] = 0.9
    heads, tracks = _heads_and_tracks(range(n), parent_head, child_head, **kw)
    return compute_gaze_categories(gaze_df, heads, tracks, (W, H), 0, 1)


def test_mutual_gaze_both_looking_at_each_other() -> None:
    result = _categories([(0.7, 0.3)] * 3, [(0.2, 0.3)] * 3, (0.2, 0.3), (0.7, 0.3))
    assert all(result["gaze_category"] == "mutual_gaze")


def test_joint_attention_both_looking_same_spot() -> None:
    result = _categories([(0.5, 0.9)] * 3, [(0.5, 0.9)] * 3, (0.2, 0.3), (0.8, 0.3))
    assert all(result["gaze_category"] == "joint_attention")


def test_independent_gaze_looking_different_spots() -> None:
    result = _categories([(0.45, 0.05)] * 3, [(0.5, 0.95)] * 3, (0.2, 0.3), (0.8, 0.3))
    assert all(result["gaze_category"] == "independent")


def test_shared_spot_near_the_child_is_watching_not_joint_attention() -> None:
    """Old rule bug: gaze on the partner was absorbed into joint attention."""
    result = _categories([(0.8, 0.3)] * 3, [(0.8, 0.31)] * 3, (0.2, 0.3), (0.8, 0.3))
    assert all(result["gaze_category"] == "parent_watching")


def test_missing_head_box_aligns_by_frame_idx() -> None:
    """A frame without the child's head box does not borrow another frame's."""
    result = _categories(
        [(0.7, 0.3)] * 5,
        [(0.2, 0.3)] * 5,
        (0.2, 0.3),
        (0.7, 0.3),
        n=5,
        missing_child_head=(2, 3),
    )
    by_frame = dict(zip(result["frame_idx"], result["gaze_category"]))
    assert by_frame[0] == by_frame[1] == by_frame[4] == "mutual_gaze"
    # Without the child's head box the parent's gaze still lands on the
    # child's body box: watching each other, but not face to face.
    assert by_frame[2] == by_frame[3] == "both_watching"


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


# ---- Gap-aware, time-based windows (metrics-validity plan, phase 1) ----


def _make_pose_df_at(
    timestamps: np.ndarray,
    parent_x: np.ndarray,
    child_x: np.ndarray,
    fps: float = 6.0,
) -> pd.DataFrame:
    """Build a pose table with explicit timestamps, so tests can insert gaps."""
    rows = []
    kps = ["kp_005", "kp_006", "kp_011", "kp_012"]
    offsets = [(-10, -5), (10, -5), (-10, 5), (10, 5)]
    for ts, px, cx in zip(timestamps, parent_x, child_x):
        frame_idx = int(round(ts * fps))
        for tid, bx in ((0, px), (1, cx)):
            for kp, (dx, dy) in zip(kps, offsets):
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_s": float(ts),
                        "track_id": tid,
                        "track_label": f"person_{tid:02d}",
                        "keypoint_name": kp,
                        "x_m": float(bx) + dx,
                        "y_m": 200.0 + dy,
                        "z_m": 0.0,
                        "keypoint_confidence": 0.9,
                    }
                )
    return pd.DataFrame(rows)


def _two_runs_with_gap(fps: float = 6.0) -> tuple[np.ndarray, float, float]:
    """30 s of frames, a 60 s gap, then 30 s more. Returns (ts, gap_start, gap_end)."""
    first = np.arange(0, 30, 1 / fps)
    second = np.arange(90, 120, 1 / fps)
    return np.concatenate([first, second]), float(first[-1]), float(second[0])


def test_xcorr_windows_never_span_a_gap() -> None:
    """No window may start before a gap and end after it."""
    rng = np.random.default_rng(0)
    ts, gap_start, gap_end = _two_runs_with_gap()
    signal = 200.0 + np.cumsum(rng.normal(0, 3, len(ts)))
    pose_df = _make_pose_df_at(ts, signal, signal + 50.0)

    result = compute_movement_xcorr(pose_df, 0, 1, window_s=5.0, max_lag_s=2.0)

    assert not result.empty
    crosses = (result["window_start_s"] <= gap_start) & (
        result["window_end_s"] >= gap_end
    )
    assert not crosses.any()
    span = result["window_end_s"] - result["window_start_s"]
    assert (span <= 5.0 + 1 / 6.0 + 1e-9).all()


def test_xcorr_reports_segment_id_per_run() -> None:
    """Windows from separate runs carry different segment ids."""
    rng = np.random.default_rng(1)
    ts, gap_start, _ = _two_runs_with_gap()
    signal = 200.0 + np.cumsum(rng.normal(0, 3, len(ts)))
    pose_df = _make_pose_df_at(ts, signal, signal)

    result = compute_movement_xcorr(pose_df, 0, 1, window_s=5.0, max_lag_s=2.0)

    assert "segment_id" in result.columns
    before = result[result["window_end_s"] <= gap_start]["segment_id"].unique()
    after = result[result["window_start_s"] > gap_start]["segment_id"].unique()
    assert len(before) == 1 and len(after) == 1
    assert before[0] != after[0]


def test_xcorr_run_shorter_than_window_gives_no_window() -> None:
    """A 3 s run cannot hold a 5 s window, so it yields nothing."""
    ts = np.arange(0, 3, 1 / 6.0)
    x = 200.0 + 10.0 * np.sin(ts)
    pose_df = _make_pose_df_at(ts, x, x)

    result = compute_movement_xcorr(pose_df, 0, 1, window_s=5.0, max_lag_s=2.0)

    assert result.empty
    assert {"window_start_s", "window_end_s", "peak_xcorr", "peak_lag_s"} <= set(
        result.columns
    )


def test_xcorr_position_jump_across_gap_is_not_movement() -> None:
    """Both people standing still, but at new spots after a gap: no synchrony.

    Computing velocity across the gap would create a matching speed spike in
    both people and a spurious high correlation.
    """
    ts, gap_start, _ = _two_runs_with_gap()
    parent = np.where(ts <= gap_start, 100.0, 600.0)
    child = np.where(ts <= gap_start, 150.0, 700.0)
    pose_df = _make_pose_df_at(ts, parent, child)

    result = compute_movement_xcorr(pose_df, 0, 1, window_s=5.0, max_lag_s=2.0)

    assert not result.empty
    assert (result["peak_xcorr"] == 0.0).all()


def test_xcorr_lag_search_stays_within_max_lag() -> None:
    """At 6 fps a 0.3 s limit allows 1 frame (0.167 s), never 2 (0.333 s)."""
    rng = np.random.default_rng(3)
    ts = np.arange(0, 60, 1 / 6.0)
    parent = 200.0 + np.cumsum(rng.normal(0, 3, len(ts)))
    child = np.roll(parent, 2)  # child follows by 2 frames = 0.333 s
    pose_df = _make_pose_df_at(ts, parent, child)

    result = compute_movement_xcorr(pose_df, 0, 1, window_s=5.0, max_lag_s=0.3)

    assert not result.empty
    assert (result["peak_lag_s"].abs() <= 0.3 + 1e-9).all()


# ---- Chance baseline (metrics-validity plan, phase 2) ----


def test_xcorr_chance_is_well_below_real_for_shared_movement() -> None:
    """When the child copies the parent, real synchrony beats the shifted baseline."""
    from gaze_analysis.synchrony import compute_movement_xcorr_chance

    rng = np.random.default_rng(5)
    ts = np.arange(0, 180, 1 / 6.0)
    parent = 200.0 + np.cumsum(rng.normal(0, 3, len(ts)))
    pose_df = _make_pose_df_at(ts, parent, parent + 40.0)

    real = compute_movement_xcorr(pose_df, 0, 1, window_s=5.0, max_lag_s=2.0)
    chance = compute_movement_xcorr_chance(pose_df, 0, 1, window_s=5.0, max_lag_s=2.0)

    assert real["peak_xcorr"].mean() - chance["peak_xcorr"].mean() > 0.3


def test_xcorr_chance_matches_real_for_independent_movement() -> None:
    """Unrelated movement: real and shifted baseline are about equal."""
    from gaze_analysis.synchrony import compute_movement_xcorr_chance

    rng = np.random.default_rng(6)
    ts = np.arange(0, 600, 1 / 6.0)
    parent = 200.0 + np.cumsum(rng.normal(0, 3, len(ts)))
    child = 400.0 + np.cumsum(rng.normal(0, 3, len(ts)))
    pose_df = _make_pose_df_at(ts, parent, child)

    real = compute_movement_xcorr(pose_df, 0, 1, window_s=5.0, max_lag_s=2.0)
    chance = compute_movement_xcorr_chance(pose_df, 0, 1, window_s=5.0, max_lag_s=2.0)

    assert abs(real["peak_xcorr"].mean() - chance["peak_xcorr"].mean()) < 0.05


def test_xcorr_chance_short_block_uses_half_its_length() -> None:
    """A 40 s block cannot shift by 60 s; it shifts by half its length instead."""
    from gaze_analysis.synchrony import compute_movement_xcorr_chance

    rng = np.random.default_rng(7)
    ts = np.arange(0, 40, 1 / 6.0)
    parent = 200.0 + np.cumsum(rng.normal(0, 3, len(ts)))
    pose_df = _make_pose_df_at(ts, parent, parent)

    chance = compute_movement_xcorr_chance(pose_df, 0, 1, window_s=5.0, max_lag_s=2.0)

    assert not chance.empty
    assert chance["peak_xcorr"].mean() < 0.9


# ---- Whole-body movement synchrony (adopted measure) ----


def _make_body_pose(
    ts: np.ndarray,
    parent_dx: np.ndarray,
    child_dx: np.ndarray,
    child_conf: float = 0.9,
    low_conf_jitter: float = 0.0,
) -> pd.DataFrame:
    """17 keypoints per person; each moves by the person's dx over time.

    Keypoint kp_016 of the parent gets low confidence and optional jitter, so
    tests can check it is ignored.
    """
    rng = np.random.default_rng(11)
    rows = []
    for i, t in enumerate(ts):
        fidx = int(round(t * 5))
        for tid, dx, conf in ((0, parent_dx[i], 0.9), (1, child_dx[i], child_conf)):
            for k in range(17):
                low = tid == 0 and k == 16
                x = 100.0 * tid + 5.0 * k + dx
                if low:
                    x += low_conf_jitter * rng.normal()
                rows.append(
                    {
                        "frame_idx": fidx,
                        "timestamp_s": float(t),
                        "track_id": tid,
                        "track_label": f"person_{tid:02d}",
                        "keypoint_name": f"kp_{k:03d}",
                        "x_m": x,
                        "y_m": 200.0 + 3.0 * k,
                        "z_m": 0.0,
                        "keypoint_confidence": 0.1 if low else conf,
                    }
                )
    return pd.DataFrame(rows)


def _random_walk(n: int, seed: int) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(0, 4, n))


def test_body_sync_shared_movement_beats_chance() -> None:
    from gaze_analysis.synchrony import compute_movement_sync

    ts = np.round(np.arange(0, 240, 0.2), 6)
    walk = _random_walk(len(ts), 1)
    pose = _make_body_pose(ts, walk, walk)

    real, chance = compute_movement_sync(pose, 0, 1)

    assert real["sync_r"].mean() > 0.9
    assert real["sync_r"].mean() - chance["sync_r"].mean() > 0.5


def test_body_sync_independent_movement_near_zero() -> None:
    from gaze_analysis.synchrony import compute_movement_sync

    ts = np.round(np.arange(0, 600, 0.2), 6)
    pose = _make_body_pose(ts, _random_walk(len(ts), 2), _random_walk(len(ts), 3))

    real, chance = compute_movement_sync(pose, 0, 1)

    assert abs(real["sync_r"].mean()) < 0.1
    assert abs(real["sync_r"].mean() - chance["sync_r"].mean()) < 0.1


def test_body_sync_ignores_low_confidence_keypoints() -> None:
    """Wild jitter on a low-confidence keypoint must not change the result."""
    from gaze_analysis.synchrony import compute_movement_sync

    ts = np.round(np.arange(0, 120, 0.2), 6)
    walk = _random_walk(len(ts), 4)
    clean, _ = compute_movement_sync(_make_body_pose(ts, walk, walk), 0, 1)
    noisy, _ = compute_movement_sync(
        _make_body_pose(ts, walk, walk, low_conf_jitter=500.0), 0, 1
    )

    np.testing.assert_allclose(clean["sync_r"], noisy["sync_r"], atol=1e-9)


def test_body_sync_windows_are_20s_inside_gap_free_runs() -> None:
    from gaze_analysis.synchrony import compute_movement_sync

    ts = np.round(np.concatenate([np.arange(0, 50, 0.2), np.arange(110, 125, 0.2)]), 6)
    walk = _random_walk(len(ts), 5)
    real, _ = compute_movement_sync(_make_body_pose(ts, walk, walk), 0, 1)

    span = real["window_end_s"] - real["window_start_s"]
    assert (span <= 20.0 + 1e-6).all()
    assert (real["window_end_s"] < 50).all()  # the 15 s run is too short
    assert set(real.columns) >= {
        "segment_id",
        "window_start_s",
        "window_end_s",
        "sync_r",
    }


def test_body_sync_chance_averages_several_shifts() -> None:
    """Long blocks average shifts at 60 s steps (at most 10); short ones use half."""
    from gaze_analysis.synchrony import compute_movement_sync

    ts = np.round(np.arange(0, 300, 0.2), 6)  # 5 min: shifts 60, 120, 180, 240 s
    walk = _random_walk(len(ts), 6)
    _, chance = compute_movement_sync(_make_body_pose(ts, walk, walk), 0, 1)
    assert sorted(chance["shift_s"].unique()) == [60.0, 120.0, 180.0, 240.0]

    short = np.round(np.arange(0, 90, 0.2), 6)
    walk = _random_walk(len(short), 7)
    _, chance = compute_movement_sync(_make_body_pose(short, walk, walk), 0, 1)
    assert list(chance["shift_s"].unique()) == [45.0]
