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
        DataFrame with frame_idx, timestamp_s, torso_distance_px columns.
    """
    parent = _torso_centroid(pose_df, parent_track_id)
    child = _torso_centroid(pose_df, child_track_id)

    merged = parent.merge(child, on="frame_idx", suffixes=("_p", "_c"))

    merged["torso_distance_px"] = np.sqrt(
        (merged["cx_p"] - merged["cx_c"]) ** 2 + (merged["cy_p"] - merged["cy_c"]) ** 2
    )

    return merged[["frame_idx", "timestamp_s_p", "torso_distance_px"]].rename(
        columns={"timestamp_s_p": "timestamp_s"}
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


def compute_movement_xcorr(
    pose_df: pd.DataFrame,
    parent_track_id: int = 0,
    child_track_id: int = 1,
    window_s: float = 5.0,
    max_lag_s: float = 2.0,
) -> pd.DataFrame:
    """Compute windowed normalized cross-correlation of movement speed.

    Args:
        pose_df: Pose DataFrame (pose_3d schema).
        parent_track_id: Track ID for the parent.
        child_track_id: Track ID for the child.
        window_s: Window duration in seconds.
        max_lag_s: Maximum lag in seconds for cross-correlation.

    Returns:
        DataFrame with window_start_s, window_end_s, peak_xcorr, peak_lag_s.
    """
    parent_c = _torso_centroid(pose_df, parent_track_id)
    child_c = _torso_centroid(pose_df, child_track_id)

    parent_v = _compute_velocity(parent_c)
    child_v = _compute_velocity(child_c)

    # Merge on frame_idx to align
    merged = (
        parent_v[["frame_idx", "timestamp_s", "speed"]]
        .merge(
            child_v[["frame_idx", "speed"]],
            on="frame_idx",
            suffixes=("_p", "_c"),
        )
        .sort_values("timestamp_s")
        .reset_index(drop=True)
    )

    timestamps = merged["timestamp_s"].values.astype(float)
    speed_p = merged["speed_p"].values.astype(float)
    speed_c = merged["speed_c"].values.astype(float)

    if len(timestamps) == 0:
        return pd.DataFrame(
            columns=["window_start_s", "window_end_s", "peak_xcorr", "peak_lag_s"]
        )

    # Estimate frame interval
    if len(timestamps) > 1:
        dt = float(np.median(np.diff(timestamps)))
    else:
        dt = 0.2  # default 5 fps

    max_lag_frames = max(1, int(max_lag_s / dt))
    window_frames = max(2 * max_lag_frames + 1, int(window_s / dt))

    results = []
    for start_idx in range(0, len(timestamps) - window_frames + 1, window_frames // 2):
        end_idx = start_idx + window_frames
        if end_idx > len(timestamps):
            break

        p_win = speed_p[start_idx:end_idx]
        c_win = speed_c[start_idx:end_idx]

        # Normalize
        p_norm = p_win - p_win.mean()
        c_norm = c_win - c_win.mean()

        p_std = np.std(p_norm)
        c_std = np.std(c_norm)

        if p_std < 1e-8 or c_std < 1e-8:
            results.append(
                {
                    "window_start_s": timestamps[start_idx],
                    "window_end_s": timestamps[end_idx - 1],
                    "peak_xcorr": 0.0,
                    "peak_lag_s": 0.0,
                }
            )
            continue

        # Compute cross-correlation at lags [-max_lag, +max_lag]
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

        results.append(
            {
                "window_start_s": timestamps[start_idx],
                "window_end_s": timestamps[end_idx - 1],
                "peak_xcorr": max(0.0, best_r),
                "peak_lag_s": best_lag * dt,
            }
        )

    return pd.DataFrame(results)


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
