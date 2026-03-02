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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from gaze_analysis.config import (
    SessionConfig,
    add_time_range_args,
    filter_by_time_range,
    filter_heatmaps_by_time_range,
    load_session_config,
    resolve_time_range,
)

# COCO torso keypoints: left/right shoulder (5,6) + left/right hip (11,12)
_TORSO_KEYPOINTS = ["kp_005", "kp_006", "kp_011", "kp_012"]


def _torso_centroid(
    pose_df: pd.DataFrame, track_id: int
) -> pd.DataFrame:
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

    grouped = subset.groupby("frame_idx", sort=True).agg(
        timestamp_s=("timestamp_s", "first"),
        cx=("x_m", "mean"),
        cy=("y_m", "mean"),
    ).reset_index()

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

    merged = parent.merge(
        child, on="frame_idx", suffixes=("_p", "_c")
    )

    merged["torso_distance_px"] = np.sqrt(
        (merged["cx_p"] - merged["cx_c"]) ** 2
        + (merged["cy_p"] - merged["cy_c"]) ** 2
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
    merged = parent_v[["frame_idx", "timestamp_s", "speed"]].merge(
        child_v[["frame_idx", "speed"]],
        on="frame_idx",
        suffixes=("_p", "_c"),
    ).sort_values("timestamp_s").reset_index(drop=True)

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
    parent_track_id: int = 0,
    child_track_id: int = 1,
    parent_head_centers: list[tuple[float, float]] | None = None,
    child_head_centers: list[tuple[float, float]] | None = None,
    proximity_threshold: float = 0.15,
) -> pd.DataFrame:
    """Classify gaze patterns per frame into categories.

    Categories:
    - mutual_gaze: Both look at each other's head
    - joint_attention: Both look at the same spot (not each other)
    - parent_watching: Only parent looks at child
    - child_watching: Only child looks at parent
    - independent: Neither looks at the other or a shared spot

    Args:
        gaze_df: Gaze CSV DataFrame with gaze_peak_x, gaze_peak_y per person.
        parent_track_id: Track ID for the parent.
        child_track_id: Track ID for the child.
        parent_head_centers: List of (x, y) normalized head centers per frame.
        child_head_centers: List of (x, y) normalized head centers per frame.
        proximity_threshold: Distance threshold (normalized) for "looking at".

    Returns:
        DataFrame with frame_idx, timestamp_s, gaze_category columns.
    """
    parent_gaze = gaze_df[gaze_df["track_id"] == parent_track_id].sort_values(
        "frame_idx"
    ).reset_index(drop=True)
    child_gaze = gaze_df[gaze_df["track_id"] == child_track_id].sort_values(
        "frame_idx"
    ).reset_index(drop=True)

    # Align by frame_idx
    merged = parent_gaze[["frame_idx", "timestamp_s", "gaze_peak_x", "gaze_peak_y"]].merge(
        child_gaze[["frame_idx", "gaze_peak_x", "gaze_peak_y"]],
        on="frame_idx",
        suffixes=("_p", "_c"),
    ).sort_values("frame_idx").reset_index(drop=True)

    n = len(merged)
    categories = []

    for i in range(n):
        p_gx = merged.iloc[i]["gaze_peak_x_p"]
        p_gy = merged.iloc[i]["gaze_peak_y_p"]
        c_gx = merged.iloc[i]["gaze_peak_x_c"]
        c_gy = merged.iloc[i]["gaze_peak_y_c"]

        # Head centers for this frame
        if parent_head_centers is not None and i < len(parent_head_centers):
            p_hx, p_hy = parent_head_centers[i]
        else:
            p_hx, p_hy = 0.0, 0.0

        if child_head_centers is not None and i < len(child_head_centers):
            c_hx, c_hy = child_head_centers[i]
        else:
            c_hx, c_hy = 0.0, 0.0

        # Distance from parent's gaze to child's head
        p_looks_at_child = np.sqrt((p_gx - c_hx) ** 2 + (p_gy - c_hy) ** 2) < proximity_threshold
        # Distance from child's gaze to parent's head
        c_looks_at_parent = np.sqrt((c_gx - p_hx) ** 2 + (c_gy - p_hy) ** 2) < proximity_threshold
        # Distance between gaze peaks
        gaze_close = np.sqrt((p_gx - c_gx) ** 2 + (p_gy - c_gy) ** 2) < proximity_threshold

        if p_looks_at_child and c_looks_at_parent:
            cat = "mutual_gaze"
        elif gaze_close and not p_looks_at_child and not c_looks_at_parent:
            cat = "joint_attention"
        elif gaze_close:
            # Both looking at same area which happens to be near one's head
            cat = "joint_attention"
        elif p_looks_at_child:
            cat = "parent_watching"
        elif c_looks_at_parent:
            cat = "child_watching"
        else:
            cat = "independent"

        categories.append(cat)

    merged["gaze_category"] = categories
    return merged[["frame_idx", "timestamp_s", "gaze_category"]]


# ====================================================================
# Metric 4: Gaze Convergence Score
# ====================================================================


def compute_gaze_convergence(
    heatmaps: np.ndarray,
    keys: list[str],
    parent_track_id: int = 0,
    child_track_id: int = 1,
) -> pd.DataFrame:
    """Compute cosine similarity of parent and child gaze heatmaps per frame.

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
        p_hm = heatmaps[parent_frames[frame_key]].flatten().astype(float)
        c_hm = heatmaps[child_frames[frame_key]].flatten().astype(float)

        p_norm = np.linalg.norm(p_hm)
        c_norm = np.linalg.norm(c_hm)

        if p_norm < 1e-12 or c_norm < 1e-12:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(p_hm, c_hm) / (p_norm * c_norm))

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
) -> pd.DataFrame:
    """Compute per-block summary statistics for a metric column.

    Args:
        metrics_df: DataFrame with timestamp_s and the metric column.
        session_config: Session config with block definitions.
        metric_column: Column name to summarize.

    Returns:
        DataFrame with block_name, mean, std, median, count columns.
    """
    rows = []
    for block in session_config.session_blocks:
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

    # Metric 3: Gaze categories (derive head centers from gaze CSV or head_bbox)
    head_bbox_path = camera_dir / "head_bboxes.csv"
    if head_bbox_path.exists():
        head_bbox_df = pd.read_csv(head_bbox_path)
        p_heads = head_bbox_df[head_bbox_df["track_id"] == pid].sort_values("frame_idx")
        c_heads = head_bbox_df[head_bbox_df["track_id"] == cid].sort_values("frame_idx")
        parent_head_centers = list(
            zip(
                ((p_heads["head_x1"] + p_heads["head_x2"]) / 2).values,
                ((p_heads["head_y1"] + p_heads["head_y2"]) / 2).values,
            )
        )
        child_head_centers = list(
            zip(
                ((c_heads["head_x1"] + c_heads["head_x2"]) / 2).values,
                ((c_heads["head_y1"] + c_heads["head_y2"]) / 2).values,
            )
        )
    else:
        parent_head_centers = None
        child_head_centers = None

    gaze_cat_df = compute_gaze_categories(
        gaze_df, pid, cid,
        parent_head_centers=parent_head_centers,
        child_head_centers=child_head_centers,
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
            proximity_df, session_config, "torso_distance_px"
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
