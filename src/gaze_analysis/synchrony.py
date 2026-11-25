"""Parent-child synchrony metrics from pose and gaze data.

Four metrics:
1. Torso proximity — Euclidean distance between torso centroids
2. Movement cross-correlation — windowed xcorr of velocity signals
3. Gaze categories — mutual gaze, joint attention, parent/child watching, independent
4. Gaze convergence — cosine similarity of gaze heatmaps
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from gaze_analysis.config import (
    SessionConfig,
    add_time_range_args,
    filter_by_time_range,
    filter_heatmaps_by_time_range,
    frame_image_size,
    load_session_config,
    resolve_time_range,
)
from gaze_analysis.gaze_categories import (
    MIN_CHANCE_SHIFT_S,
    build_frame_table,
    gaze_categories,
    heatmap_overlap,
)
from gaze_analysis.head_bbox import extract_head_bboxes

# COCO torso keypoints: left/right shoulder (5,6) + left/right hip (11,12)
_TORSO_KEYPOINTS = ["kp_005", "kp_006", "kp_011", "kp_012"]


def _torso_centroid(pose_df: pd.DataFrame, track_id: int) -> pd.DataFrame:
    """Compute per-frame torso centroid for one person.

    Args:
        pose_df: Pose DataFrame with keypoint_name, x_m, y_m columns.
        track_id: Which track to compute centroids for.

    Returns:
        DataFrame with frame_idx, timestamp_s, cx, cy columns.
    """
    subset = pose_df[
        (pose_df["track_id"] == track_id)
        & (pose_df["keypoint_name"].isin(_TORSO_KEYPOINTS))
    ].copy()

    grouped = (
        subset.groupby("frame_idx", sort=True)
        .agg(
            timestamp_s=("timestamp_s", "first"),
            cx=("x_m", "mean"),
            cy=("y_m", "mean"),
        )
        .reset_index()
    )

    return grouped


TORSO_RULER_CONF_MIN = 0.5


def _torso_length(pose_df: pd.DataFrame, track_id: int) -> pd.DataFrame:
    """Per-frame torso length: mid-shoulder to mid-hip distance in pixels.

    Used as a ruler for torso-normalized distance: the same person filmed
    from further away has a shorter torso on screen, so dividing by it reduces
    zoom differences between cameras. It does not correct viewing angle
    (distance toward or away from the camera stays invisible).

    Args:
        pose_df: Pose DataFrame with keypoint_name, x_m, y_m columns.
        track_id: Which track to measure.

    Returns:
        DataFrame with frame_idx and torso_length_px. Frames are left out
        unless all four torso keypoints are present with confidence >=
        ``TORSO_RULER_CONF_MIN`` (no torso is built from fewer joints).
    """
    subset = pose_df[
        (pose_df["track_id"] == track_id)
        & (pose_df["keypoint_name"].isin(_TORSO_KEYPOINTS))
    ]
    if "keypoint_confidence" in subset.columns:
        subset = subset[subset["keypoint_confidence"] >= TORSO_RULER_CONF_MIN]
    wide = subset.pivot_table(
        index="frame_idx", columns="keypoint_name", values=["x_m", "y_m"]
    ).dropna()
    if wide.empty or len(wide.columns) < 8:
        return pd.DataFrame({"frame_idx": [], "torso_length_px": []})
    x, y = wide["x_m"], wide["y_m"]
    dx = (x["kp_005"] + x["kp_006"] - x["kp_011"] - x["kp_012"]) / 2
    dy = (y["kp_005"] + y["kp_006"] - y["kp_011"] - y["kp_012"]) / 2
    return pd.DataFrame(
        {"frame_idx": wide.index, "torso_length_px": np.hypot(dx, dy).to_numpy()}
    )


# ====================================================================
# Metric 1: Torso Proximity
# ====================================================================


def compute_torso_proximity(
    pose_df: pd.DataFrame,
    parent_track_id: int = 0,
    child_track_id: int = 1,
) -> pd.DataFrame:
    """Compute per-frame Euclidean distance between parent and child torso centroids.

    Args:
        pose_df: Pose DataFrame (pose_3d schema).
        parent_track_id: Track ID for the parent.
        child_track_id: Track ID for the child.

    Returns:
        DataFrame with frame_idx, timestamp_s, torso_distance_px, and each
        person's torso length (parent_torso_px, child_torso_px; NaN when a
        shoulder or hip is missing), used to express the distance in body
        sizes.
    """
    parent = _torso_centroid(pose_df, parent_track_id)
    child = _torso_centroid(pose_df, child_track_id)

    merged = parent.merge(child, on="frame_idx", suffixes=("_p", "_c"))

    merged["torso_distance_px"] = np.sqrt(
        (merged["cx_p"] - merged["cx_c"]) ** 2 + (merged["cy_p"] - merged["cy_c"]) ** 2
    )

    lengths = _torso_length(pose_df, parent_track_id).merge(
        _torso_length(pose_df, child_track_id),
        on="frame_idx",
        how="outer",
        suffixes=("_p", "_c"),
    )
    merged = merged.merge(lengths, on="frame_idx", how="left")

    return merged[
        [
            "frame_idx",
            "timestamp_s_p",
            "torso_distance_px",
            "torso_length_px_p",
            "torso_length_px_c",
        ]
    ].rename(
        columns={
            "timestamp_s_p": "timestamp_s",
            "torso_length_px_p": "parent_torso_px",
            "torso_length_px_c": "child_torso_px",
        }
    )


# ====================================================================
# Metric 2: Cross-Correlation of Movement
# ====================================================================


def _compute_velocity(
    centroid_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute velocity via central differences.

    Args:
        centroid_df: DataFrame with frame_idx, timestamp_s, cx, cy.

    Returns:
        DataFrame with frame_idx, timestamp_s, vx, vy, speed columns.
    """
    df = centroid_df.sort_values("timestamp_s").reset_index(drop=True)
    n = len(df)
    if n < 3:
        df["vx"] = 0.0
        df["vy"] = 0.0
        df["speed"] = 0.0
        return df

    cx = df["cx"].values.astype(float)
    cy = df["cy"].values.astype(float)
    ts = df["timestamp_s"].values.astype(float)

    vx = np.zeros(n)
    vy = np.zeros(n)

    # Central differences for interior points
    for i in range(1, n - 1):
        dt = ts[i + 1] - ts[i - 1]
        if dt > 0:
            vx[i] = (cx[i + 1] - cx[i - 1]) / dt
            vy[i] = (cy[i + 1] - cy[i - 1]) / dt

    # Forward/backward differences at boundaries
    dt0 = ts[1] - ts[0]
    if dt0 > 0:
        vx[0] = (cx[1] - cx[0]) / dt0
        vy[0] = (cy[1] - cy[0]) / dt0
    dtn = ts[-1] - ts[-2]
    if dtn > 0:
        vx[-1] = (cx[-1] - cx[-2]) / dtn
        vy[-1] = (cy[-1] - cy[-2]) / dtn

    df["vx"] = vx
    df["vy"] = vy
    df["speed"] = np.sqrt(vx**2 + vy**2)

    return df


XCORR_COLUMNS = [
    "segment_id",
    "window_start_s",
    "window_end_s",
    "peak_xcorr",
    "peak_lag_s",
]

# A jump in time larger than this many frame intervals ends a contiguous run.
RUN_GAP_FACTOR = 1.5


def _frame_interval(pose_df: pd.DataFrame) -> float:
    """Estimate the nominal frame interval from all timestamps in the table.

    Uses every frame either person appears in, so the estimate is the frame
    grid spacing even when both-people frames are sparse.

    Args:
        pose_df: Pose DataFrame with a ``timestamp_s`` column.

    Returns:
        Median positive spacing between unique timestamps, in seconds
        (0.2 s when it cannot be estimated).
    """
    ts = np.unique(pose_df["timestamp_s"].to_numpy(float))
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 0.2


def contiguous_runs(timestamps: np.ndarray, dt: float) -> np.ndarray:
    """Label contiguous runs of frames, splitting wherever a gap occurs.

    Args:
        timestamps: Sorted frame timestamps in seconds.
        dt: Nominal frame interval in seconds.

    Returns:
        Integer run id per timestamp, starting at 0.
    """
    if len(timestamps) == 0:
        return np.zeros(0, dtype=int)
    breaks = np.diff(timestamps) > RUN_GAP_FACTOR * dt
    return np.concatenate([[0], np.cumsum(breaks)]).astype(int)


def _window_xcorr(
    p_win: np.ndarray, c_win: np.ndarray, max_lag_frames: int
) -> tuple[float, int]:
    """Peak normalized cross-correlation of two equal-length speed windows.

    Args:
        p_win: Parent speed in the window.
        c_win: Child speed in the window.
        max_lag_frames: Largest lag searched, in frames.

    Returns:
        ``(peak_r, lag_frames)``; ``(0.0, 0)`` when either signal is flat.
        Positive lag means the child follows the parent.
    """
    p_norm = p_win - p_win.mean()
    c_norm = c_win - c_win.mean()
    if np.std(p_norm) < 1e-8 or np.std(c_norm) < 1e-8:
        return 0.0, 0

    best_r = -1.0
    best_lag = 0
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag >= 0:
            p_seg = p_norm[: len(p_norm) - lag]
            c_seg = c_norm[lag:]
        else:
            p_seg = p_norm[-lag:]
            c_seg = c_norm[: len(c_norm) + lag]
        if len(p_seg) < 2:
            continue
        r = float(np.dot(p_seg, c_seg)) / (
            np.sqrt(np.dot(p_seg, p_seg)) * np.sqrt(np.dot(c_seg, c_seg)) + 1e-12
        )
        if r > best_r:
            best_r = r
            best_lag = lag
    return max(0.0, best_r), best_lag


# Chance baseline: shift the child's movement by this much (or half the
# both-tracked data when shorter), matching the gaze chance convention.
XCORR_CHANCE_SHIFT_S = 60.0


@dataclass
class _SpeedRuns:
    """Parent and child speed inside each contiguous both-tracked run."""

    run_ids: list[int]
    timestamps: list[np.ndarray]
    parent_speed: list[np.ndarray]
    child_speed: list[np.ndarray]
    dt: float
    window_frames: int
    max_lag_frames: int


def _speed_runs(
    pose_df: pd.DataFrame,
    parent_track_id: int,
    child_track_id: int,
    window_s: float,
    max_lag_s: float,
) -> _SpeedRuns | None:
    """Split both-tracked frames into gap-free runs and compute speed in each.

    Args:
        pose_df: Pose DataFrame (pose_3d schema).
        parent_track_id: Track ID for the parent.
        child_track_id: Track ID for the child.
        window_s: Window duration in seconds.
        max_lag_s: Maximum lag in seconds.

    Returns:
        Speed series for runs at least one window long, or None when there
        are no both-tracked frames.
    """
    if pose_df.empty:
        return None
    parent_c = _torso_centroid(pose_df, parent_track_id)
    child_c = _torso_centroid(pose_df, child_track_id)
    both = (
        parent_c.merge(child_c, on="frame_idx", suffixes=("_p", "_c"))
        .sort_values("timestamp_s_p")
        .reset_index(drop=True)
    )
    if both.empty:
        return None

    dt = _frame_interval(pose_df)
    # Floor (with float tolerance) so the search never exceeds max_lag_s.
    max_lag_frames = max(1, int(np.floor(max_lag_s / dt + 1e-9)))
    window_frames = max(2 * max_lag_frames + 1, int(round(window_s / dt)))
    runs = _SpeedRuns([], [], [], [], dt, window_frames, max_lag_frames)

    run_ids = contiguous_runs(both["timestamp_s_p"].to_numpy(float), dt)
    columns = ["frame_idx", "timestamp_s", "cx", "cy"]
    for run_id in np.unique(run_ids):
        run = both[run_ids == run_id]
        if len(run) < window_frames:
            continue
        parent_v = _compute_velocity(
            run[["frame_idx", "timestamp_s_p", "cx_p", "cy_p"]].set_axis(
                columns, axis=1
            )
        )
        child_v = _compute_velocity(
            run[["frame_idx", "timestamp_s_p", "cx_c", "cy_c"]].set_axis(
                columns, axis=1
            )
        )
        runs.run_ids.append(int(run_id))
        runs.timestamps.append(parent_v["timestamp_s"].to_numpy(float))
        runs.parent_speed.append(parent_v["speed"].to_numpy(float))
        runs.child_speed.append(child_v["speed"].to_numpy(float))
    return runs


def _windowed_xcorr(runs: _SpeedRuns, child_speed: list[np.ndarray]) -> pd.DataFrame:
    """Slide half-overlapping windows inside each run and score each one.

    Args:
        runs: Speed runs from :func:`_speed_runs`.
        child_speed: Child speed per run (the real series, or a shifted one).

    Returns:
        One row per window with the :data:`XCORR_COLUMNS` columns.
    """
    step = max(1, runs.window_frames // 2)
    results = []
    for run_id, ts, speed_p, speed_c in zip(
        runs.run_ids, runs.timestamps, runs.parent_speed, child_speed
    ):
        for start_idx in range(0, len(ts) - runs.window_frames + 1, step):
            end_idx = start_idx + runs.window_frames
            peak_r, lag = _window_xcorr(
                speed_p[start_idx:end_idx],
                speed_c[start_idx:end_idx],
                runs.max_lag_frames,
            )
            results.append(
                {
                    "segment_id": run_id,
                    "window_start_s": ts[start_idx],
                    "window_end_s": ts[end_idx - 1],
                    "peak_xcorr": peak_r,
                    "peak_lag_s": lag * runs.dt,
                }
            )
    return pd.DataFrame(results, columns=XCORR_COLUMNS)


def compute_movement_xcorr(
    pose_df: pd.DataFrame,
    parent_track_id: int = 0,
    child_track_id: int = 1,
    window_s: float = 5.0,
    max_lag_s: float = 2.0,
) -> pd.DataFrame:
    """Compute windowed normalized cross-correlation of movement speed.

    Only frames where both people are tracked are used. Those frames are split
    into contiguous runs wherever a gap appears; speed is computed inside each
    run and windows are placed inside each run, so no window or speed estimate
    spans a gap. Runs shorter than one window are skipped.

    Args:
        pose_df: Pose DataFrame (pose_3d schema).
        parent_track_id: Track ID for the parent.
        child_track_id: Track ID for the child.
        window_s: Window duration in seconds.
        max_lag_s: Maximum lag in seconds for cross-correlation.

    Returns:
        DataFrame with segment_id, window_start_s, window_end_s, peak_xcorr,
        peak_lag_s (positive lag = child follows parent).
    """
    runs = _speed_runs(pose_df, parent_track_id, child_track_id, window_s, max_lag_s)
    if runs is None:
        return pd.DataFrame(columns=XCORR_COLUMNS)
    return _windowed_xcorr(runs, runs.child_speed)


def compute_movement_xcorr_chance(
    pose_df: pd.DataFrame,
    parent_track_id: int = 0,
    child_track_id: int = 1,
    window_s: float = 5.0,
    max_lag_s: float = 2.0,
    shift_s: float = XCORR_CHANCE_SHIFT_S,
) -> pd.DataFrame:
    """Movement cross-correlation expected by chance.

    Same windows as :func:`compute_movement_xcorr`, but the child's speed is
    shifted circularly by ``shift_s`` (or half the usable data when that is
    shorter) across the concatenated runs, so it is paired with the parent's
    movement from a different moment. Real synchrony should clearly exceed
    this level. Speeds are computed within runs before shifting, so the
    splice points between runs do not add spurious movement.

    Args:
        pose_df: Pose DataFrame (pose_3d schema).
        parent_track_id: Track ID for the parent.
        child_track_id: Track ID for the child.
        window_s: Window duration in seconds.
        max_lag_s: Maximum lag in seconds for cross-correlation.
        shift_s: Time shift applied to the child, in seconds.

    Returns:
        DataFrame with the :data:`XCORR_COLUMNS` columns (window times are
        the parent's).
    """
    runs = _speed_runs(pose_df, parent_track_id, child_track_id, window_s, max_lag_s)
    if runs is None or not runs.run_ids:
        return pd.DataFrame(columns=XCORR_COLUMNS)
    child_all = np.concatenate(runs.child_speed)
    n = len(child_all)
    shift = min(int(round(shift_s / runs.dt)), n // 2)
    shifted = np.roll(child_all, -shift)
    bounds = np.cumsum([len(s) for s in runs.child_speed])[:-1]
    return _windowed_xcorr(runs, np.split(shifted, bounds))


# ====================================================================
# Metric 2b: Whole-body movement synchrony (adopted measure)
# ====================================================================

SYNC_COLUMNS = ["segment_id", "window_start_s", "window_end_s", "sync_r"]
SYNC_WINDOW_S = 20.0
SYNC_CONF_MIN = 0.3
# Chance level: shifts every 10 s (>= 20 s from zero and a full turn), at
# most 10, and none when fewer than 3 fit.
SYNC_SHIFT_STEP_S = 10.0
SYNC_MIN_SHIFTS = 3


def _keypoint_arrays(
    pose_df: pd.DataFrame, track_id: int, frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, y and confidence per frame (rows) and keypoint (columns) for one person.

    Args:
        pose_df: Pose DataFrame (pose_3d schema).
        track_id: Person to extract.
        frames: frame_idx values to return, in order.

    Returns:
        Three (len(frames), n_keypoints) arrays; missing entries are NaN.
    """
    sub = pose_df[pose_df["track_id"] == track_id]
    names = sorted(pose_df["keypoint_name"].unique())

    def grid(column: str) -> np.ndarray:
        table = sub.pivot_table(
            index="frame_idx", columns="keypoint_name", values=column, aggfunc="first"
        )
        return table.reindex(index=frames, columns=names).to_numpy(dtype=float)

    return grid("x_m"), grid("y_m"), grid("keypoint_confidence")


def _body_energy(
    x: np.ndarray, y: np.ndarray, conf: np.ndarray, t: np.ndarray, conf_min: float
) -> np.ndarray:
    """Mean keypoint speed per frame, over keypoints tracked confidently.

    Speed is a central difference in time within one gap-free run, using only
    positions with confidence >= ``conf_min``: a speed is kept only when every
    position it is computed from is confident (or present). A frame with no
    usable keypoint takes the run's mean energy; a run with no usable
    keypoint at all is all NaN (unmeasured, not motionless).

    Returns:
        Motion energy per frame, in pixels per second.
    """
    if len(t) < 2:
        return np.full(len(t), np.nan)
    # Blank unreliable positions before differencing, so a low-confidence
    # glitch cannot leak into the speed of its confident neighbours.
    unreliable = ~(conf >= conf_min)
    x = np.where(unreliable, np.nan, x)
    y = np.where(unreliable, np.nan, y)
    speed = np.hypot(np.gradient(x, t, axis=0), np.gradient(y, t, axis=0))
    speed[unreliable | ~np.isfinite(speed)] = np.nan
    with np.errstate(all="ignore"):
        energy = np.nanmean(speed, axis=1)
    if not np.isfinite(energy).any():
        return np.full(len(t), np.nan)
    fill = float(np.nanmean(energy))
    return np.where(np.isfinite(energy), energy, fill)


def _lag0_windows(
    run_ids: list[int],
    times: list[np.ndarray],
    parent: list[np.ndarray],
    child: list[np.ndarray],
    window_frames: int,
) -> pd.DataFrame:
    """Pearson correlation at lag 0 in half-overlapping windows inside each run.

    Windows where either person's energy is flat or unmeasured have no
    defined correlation and are left out (not counted as zero synchrony).
    """
    step = max(1, window_frames // 2)
    rows = []
    for run_id, ts, p, c in zip(run_ids, times, parent, child):
        for start in range(0, len(ts) - window_frames + 1, step):
            pw = (
                p[start : start + window_frames]
                - p[start : start + window_frames].mean()
            )
            cw = (
                c[start : start + window_frames]
                - c[start : start + window_frames].mean()
            )
            p_norm, c_norm = np.sqrt(pw @ pw), np.sqrt(cw @ cw)
            # Flat (or unmeasured) energy in either person: undefined r.
            # 1e-6 px/s is far below any real movement but above float noise.
            if not (np.isfinite(p_norm) and np.isfinite(c_norm)):
                continue
            if p_norm < 1e-6 or c_norm < 1e-6:
                continue
            denom = p_norm * c_norm
            rows.append(
                {
                    "segment_id": run_id,
                    "window_start_s": ts[start],
                    "window_end_s": ts[start + window_frames - 1],
                    "sync_r": float(pw @ cw / denom),
                }
            )
    return pd.DataFrame(rows, columns=SYNC_COLUMNS)


def compute_movement_sync(
    pose_df: pd.DataFrame,
    parent_track_id: int = 0,
    child_track_id: int = 1,
    window_s: float = SYNC_WINDOW_S,
    conf_min: float = SYNC_CONF_MIN,
    shift_step_s: float = SYNC_SHIFT_STEP_S,
    max_shifts: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Whole-body movement synchrony and its chance level.

    Each person's motion energy is the mean speed of their confidently tracked
    keypoints. Parent and child energies are correlated at lag 0 in 20 s
    windows (50% overlap) placed inside gap-free runs of frames where both are
    tracked. The chance level repeats this with the child's energy shifted
    circularly across the concatenated runs, so the child is paired with the
    parent's movement from another moment. One rule for every block length:
    shifts every ``shift_step_s`` seconds, at least 20 s from no shift and
    from a full turn, thinned evenly to at most ``max_shifts``; with fewer
    than ``SYNC_MIN_SHIFTS`` possible shifts there is no chance level (a
    single shift is too arbitrary: on 1018 B grocery_part1 one 53 s shift gave
    -0.21, while shifts of 20-80 s averaged -0.03). This is a time-shift
    reference level, not a significance test.

    On sessions 1011v3-1020v3 this separated real from shifted data (mean
    difference +0.17, effect size 1.7), survived removing background motion
    and frames with possibly shared keypoints, whereas peak torso-speed
    cross-correlation in 5 s windows did not (+0.01).

    Args:
        pose_df: Pose DataFrame (pose_3d schema).
        parent_track_id: Track ID for the parent.
        child_track_id: Track ID for the child.
        window_s: Window length in seconds.
        conf_min: Minimum keypoint confidence for a keypoint to count.
        shift_step_s: Step between chance-level shifts of the child, seconds.
        max_shifts: Most shifts pooled for the chance level.

    Returns:
        ``(real, chance)`` tables with segment_id, window_start_s,
        window_end_s, sync_r (window times are the parent's); ``chance`` also
        has ``shift_s``.
    """
    empty = pd.DataFrame(columns=SYNC_COLUMNS)
    if pose_df.empty:
        return empty, empty.copy()
    times_by_frame = pose_df.groupby("frame_idx")["timestamp_s"].first()
    tracked = pose_df.groupby("track_id")["frame_idx"].apply(set)
    if parent_track_id not in tracked or child_track_id not in tracked:
        return empty, empty.copy()
    frames = np.array(sorted(tracked[parent_track_id] & tracked[child_track_id]))
    if len(frames) == 0:
        return empty, empty.copy()
    t_all = times_by_frame.reindex(frames).to_numpy(dtype=float)
    order = np.argsort(t_all)
    frames, t_all = frames[order], t_all[order]

    dt = _frame_interval(pose_df)
    window_frames = max(3, int(round(window_s / dt)))
    px, py, pc = _keypoint_arrays(pose_df, parent_track_id, frames)
    cx, cy, cc = _keypoint_arrays(pose_df, child_track_id, frames)

    run_ids: list[int] = []
    times: list[np.ndarray] = []
    parent: list[np.ndarray] = []
    child: list[np.ndarray] = []
    labels = contiguous_runs(t_all, dt)
    for run_id in np.unique(labels):
        sel = labels == run_id
        if sel.sum() < window_frames:
            continue
        ts = t_all[sel]
        run_ids.append(int(run_id))
        times.append(ts)
        parent.append(_body_energy(px[sel], py[sel], pc[sel], ts, conf_min))
        child.append(_body_energy(cx[sel], cy[sel], cc[sel], ts, conf_min))
    if not run_ids:
        return empty, empty.copy()

    real = _lag0_windows(run_ids, times, parent, child, window_frames)
    if real.empty:
        return real, empty.copy()
    real_keys = real[["segment_id", "window_start_s"]]
    child_all = np.concatenate(child)
    bounds = np.cumsum([len(c) for c in child])[:-1]
    chance_tables = []
    for shift in _chance_shifts(len(child_all), dt, shift_step_s, max_shifts):
        shifted = np.split(np.roll(child_all, -shift), bounds)
        table = _lag0_windows(run_ids, times, parent, shifted, window_frames)
        # Same moments as the real windows, so real and reference compare
        # like with like (a window undefined in either is left out of both
        # only through this shift's mean).
        table = table.merge(real_keys, on=["segment_id", "window_start_s"])
        if not table.empty:
            chance_tables.append(table.assign(shift_s=round(shift * dt, 6)))
    if not chance_tables:
        return real, empty.copy()
    chance = pd.concat(chance_tables, ignore_index=True)
    return real, chance


def _chance_shifts(n: int, dt: float, step_s: float, max_shifts: int) -> list[int]:
    """Circular shifts (in frames) for the chance level.

    Every ``step_s`` from ``MIN_CHANCE_SHIFT_S`` up to ``MIN_CHANCE_SHIFT_S``
    short of a full turn, thinned evenly to at most ``max_shifts``. Returns
    no shifts when fewer than ``SYNC_MIN_SHIFTS`` fit, because one or two
    shifts give an arbitrary chance level in short blocks.
    """
    lo = int(np.ceil(MIN_CHANCE_SHIFT_S / dt - 1e-9))
    step = max(1, int(round(step_s / dt)))
    shifts = list(range(lo, n - lo + 1, step))
    if len(shifts) < SYNC_MIN_SHIFTS:
        return []
    if len(shifts) > max_shifts:
        keep = np.linspace(0, len(shifts) - 1, max_shifts).round().astype(int)
        shifts = [shifts[i] for i in sorted(set(keep))]
    return shifts if len(shifts) >= SYNC_MIN_SHIFTS else []


def chance_summary(chance: pd.DataFrame) -> dict:
    """Reference level from shifted windows: each shift counts equally.

    Args:
        chance: Shifted windows with ``shift_s`` and ``sync_r``.

    Returns:
        ``sync_chance_mean`` (mean of per-shift means), ``sync_chance_sd``
        (spread across shifts) and ``sync_chance_n_shifts``; empty when fewer
        than ``SYNC_MIN_SHIFTS`` shifts have windows.
    """
    if chance.empty:
        return {}
    per_shift = chance.groupby("shift_s")["sync_r"].mean().dropna()
    if len(per_shift) < SYNC_MIN_SHIFTS:
        return {}
    return {
        "sync_chance_mean": float(per_shift.mean()),
        "sync_chance_sd": float(per_shift.std()),
        "sync_chance_n_shifts": int(len(per_shift)),
    }


# ====================================================================
# Metric 3: Gaze Categories
# ====================================================================


def compute_gaze_categories(
    gaze_df: pd.DataFrame,
    head_boxes_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    image_size: tuple[int, int],
    parent_track_id: int = 0,
    child_track_id: int = 1,
) -> pd.DataFrame:
    """Classify each frame's parent-child gaze pattern.

    See :mod:`gaze_analysis.gaze_categories` for the category rules.

    Args:
        gaze_df: Gaze CSV rows (gaze point as image fractions, in/out score).
        head_boxes_df: Normalized head boxes (``extract_head_bboxes``).
        tracks_df: Tracks with pixel body boxes.
        image_size: (width, height) of the frames in pixels.
        parent_track_id: Track ID for the parent.
        child_track_id: Track ID for the child.

    Returns:
        DataFrame with frame_idx, timestamp_s, gaze_category columns.
    """
    frames = build_frame_table(
        gaze_df, head_boxes_df, tracks_df, image_size, parent_track_id, child_track_id
    )
    if frames.empty:
        return pd.DataFrame(columns=["frame_idx", "timestamp_s", "gaze_category"])
    return gaze_categories(frames)


# ====================================================================
# Metric 4: Gaze Convergence Score
# ====================================================================


def compute_gaze_convergence(
    heatmaps: np.ndarray,
    keys: list[str],
    parent_track_id: int = 0,
    child_track_id: int = 1,
) -> pd.DataFrame:
    """Compute the overlap of parent and child gaze heatmaps per frame.

    Uses :func:`gaze_analysis.gaze_categories.heatmap_overlap` (background
    removed, maps normalized to sum to 1, summed elementwise minimum): 1 means
    identical gaze distributions and 0 disjoint ones.

    Args:
        heatmaps: Array of shape [N, 64, 64] from gaze_heatmaps.npz.
        keys: List of keys like "f000001_t0" matching heatmap indices.
        parent_track_id: Track ID for the parent.
        child_track_id: Track ID for the child.

    Returns:
        DataFrame with frame_idx, gaze_convergence_score columns.
    """
    # Build lookup: key → index
    key_to_idx = {k: i for i, k in enumerate(keys)}

    # Find all frames where both parent and child have heatmaps
    parent_suffix = f"_t{parent_track_id}"
    child_suffix = f"_t{child_track_id}"

    parent_keys = {k: key_to_idx[k] for k in keys if k.endswith(parent_suffix)}
    child_keys = {k: key_to_idx[k] for k in keys if k.endswith(child_suffix)}

    # Extract frame indices
    parent_frames = {k.split("_t")[0]: idx for k, idx in parent_keys.items()}
    child_frames = {k.split("_t")[0]: idx for k, idx in child_keys.items()}

    common_frames = sorted(set(parent_frames.keys()) & set(child_frames.keys()))

    rows = []
    for frame_key in common_frames:
        cos_sim = heatmap_overlap(
            heatmaps[parent_frames[frame_key]], heatmaps[child_frames[frame_key]]
        )

        # Extract frame index from key like "f000001"
        frame_idx = int(frame_key[1:])
        rows.append(
            {
                "frame_idx": frame_idx,
                "gaze_convergence_score": cos_sim,
            }
        )

    return pd.DataFrame(rows)


# ====================================================================
# Aggregate: Block Summaries
# ====================================================================


def compute_block_summary(
    metrics_df: pd.DataFrame,
    session_config: SessionConfig,
    metric_column: str,
    camera_id: str | None = None,
) -> pd.DataFrame:
    """Compute per-block summary statistics for a metric column.

    Args:
        metrics_df: DataFrame with timestamp_s and the metric column.
        session_config: Session config with block definitions.
        metric_column: Column name to summarize.
        camera_id: Use this camera's own blocks (default: shared blocks).

    Returns:
        DataFrame with block_name, mean, std, median, count columns.
    """
    rows = []
    blocks = (
        session_config.blocks_for_camera(camera_id)
        if camera_id is not None
        else session_config.session_blocks
    )
    for block in blocks:
        mask = (metrics_df["timestamp_s"] >= block.start_s) & (
            metrics_df["timestamp_s"] <= block.end_s
        )
        block_data = metrics_df.loc[mask, metric_column]
        rows.append(
            {
                "block_name": block.name,
                "block_color": block.color,
                "mean": float(block_data.mean()) if len(block_data) > 0 else 0.0,
                "std": float(block_data.std()) if len(block_data) > 1 else 0.0,
                "median": float(block_data.median()) if len(block_data) > 0 else 0.0,
                "count": len(block_data),
            }
        )
    return pd.DataFrame(rows)


# ====================================================================
# CLI
# ====================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for gaze-synchrony."""
    parser = argparse.ArgumentParser(
        prog="gaze-synchrony",
        description="Compute parent-child synchrony metrics from pose and gaze data.",
    )
    parser.add_argument("--camera-dir", required=True, type=str)
    parser.add_argument("--session-config", required=True, type=str)
    parser.add_argument("--camera-id", default="camera_a", type=str)
    parser.add_argument("--pose-input", default="pose_3d.csv", type=str)
    parser.add_argument("--tracks-input", default="tracks_2d.csv", type=str)
    parser.add_argument("--gaze-csv", default="gaze_heatmap.csv", type=str)
    parser.add_argument("--gaze-npz", default="gaze_heatmaps.npz", type=str)
    parser.add_argument("--output-csv", default="synchrony_metrics.csv", type=str)
    parser.add_argument("--output-json", default="synchrony_summary.json", type=str)
    add_time_range_args(parser)
    return parser


def main() -> None:
    """CLI entry point for gaze-synchrony."""
    parser = build_arg_parser()
    args = parser.parse_args()

    camera_dir = Path(args.camera_dir)
    session_config = load_session_config(args.session_config)
    mapping = session_config.get_camera_mapping(args.camera_id)

    pid = mapping.parent_track_id
    cid = mapping.child_track_id

    # Resolve time range
    t_start, t_end = resolve_time_range(
        session_config, args.start_s, args.end_s, args.time_block
    )
    time_limited = t_start > 0.0 or t_end < float("inf")
    if time_limited:
        print(f"Time range: {t_start:.1f}s – {t_end:.1f}s")

    # Load data
    pose_df = pd.read_csv(camera_dir / args.pose_input)
    gaze_df = pd.read_csv(camera_dir / args.gaze_csv)

    npz_path = camera_dir / args.gaze_npz
    if npz_path.exists():
        npz_data = np.load(npz_path)
        heatmaps = npz_data["heatmaps"]
        hm_keys = list(npz_data["keys"])
    else:
        heatmaps = np.array([])
        hm_keys = []

    # Apply time-range filter
    if time_limited:
        pose_df = filter_by_time_range(pose_df, t_start, t_end)
        gaze_df = filter_by_time_range(gaze_df, t_start, t_end)
        if heatmaps.size > 0:
            # Build frame→timestamp lookup from pose_df
            frame_ts = dict(
                zip(pose_df["frame_idx"].values, pose_df["timestamp_s"].values)
            )
            heatmaps, hm_keys = filter_heatmaps_by_time_range(
                heatmaps, hm_keys, frame_ts, t_start, t_end
            )

    # Metric 1: Torso proximity
    proximity_df = compute_torso_proximity(pose_df, pid, cid)

    # Metric 2: Movement cross-correlation
    xcorr_df = compute_movement_xcorr(pose_df, pid, cid)

    # Metric 3: Gaze categories, from each person's gaze point relative to
    # the other's head and body boxes (in pixels of the real frame size).
    tracks_df = pd.read_csv(camera_dir / args.tracks_input)
    if time_limited:
        tracks_df = filter_by_time_range(tracks_df, t_start, t_end)
    image_size = frame_image_size(
        camera_dir, (session_config.image_width, session_config.image_height)
    )
    head_boxes_df = extract_head_bboxes(pose_df, tracks_df, *image_size)
    gaze_cat_df = compute_gaze_categories(
        gaze_df, head_boxes_df, tracks_df, image_size, pid, cid
    )

    # Metric 4: Gaze convergence
    if heatmaps.size > 0:
        convergence_df = compute_gaze_convergence(heatmaps, hm_keys, pid, cid)
    else:
        convergence_df = pd.DataFrame(columns=["frame_idx", "gaze_convergence_score"])

    # Merge all frame-level metrics
    frame_metrics = proximity_df.copy()
    if not convergence_df.empty:
        frame_metrics = frame_metrics.merge(convergence_df, on="frame_idx", how="left")
    frame_metrics = frame_metrics.merge(
        gaze_cat_df[["frame_idx", "gaze_category"]], on="frame_idx", how="left"
    )

    # Write outputs
    frame_metrics.to_csv(camera_dir / args.output_csv, index=False)

    # Block summaries
    summary: Dict[str, Any] = {"camera_id": args.camera_id}
    if not proximity_df.empty:
        summary["torso_proximity_blocks"] = compute_block_summary(
            proximity_df, session_config, "torso_distance_px", args.camera_id
        ).to_dict(orient="records")

    if not xcorr_df.empty:
        summary["xcorr_windows"] = len(xcorr_df)
        summary["mean_peak_xcorr"] = float(xcorr_df["peak_xcorr"].mean())

    if not gaze_cat_df.empty:
        summary["gaze_category_counts"] = (
            gaze_cat_df["gaze_category"].value_counts().to_dict()
        )

    if not convergence_df.empty:
        summary["mean_gaze_convergence"] = float(
            convergence_df["gaze_convergence_score"].mean()
        )

    summary_path = camera_dir / args.output_json
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(frame_metrics)} metric rows to {camera_dir / args.output_csv}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
