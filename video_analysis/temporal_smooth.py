"""Temporal EMA smoothing and confidence-gated keypoint infill."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

POSE_POSITION_COLUMNS = ["x_m", "y_m", "z_m"]
TRACK_POSITION_COLUMNS = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


@dataclass
class SmoothingConfig:
    """Configuration for temporal smoothing on one camera output directory."""

    camera_dir: str
    tau: float = 0.15
    conf_threshold: float = 0.3
    conf_gate: float = 0.3
    # Longest run of low-confidence frames to infill, in seconds (time-based so
    # it is independent of the sampling rate). Converted to a frame count from
    # the observed timestamp spacing. Infilled rows are flagged in the
    # ``interpolated`` column.
    max_infill_gap_s: float = 1.0
    pose_input: str = "pose_3d_interpolated.csv"
    tracks_input: str = "tracks_2d_interpolated.csv"
    pose_output: str = "pose_3d_smooth.csv"
    tracks_output: str = "tracks_2d_smooth.csv"
    summary_output: str = "smoothing_summary.json"


def _apply_ema_forward(
    values: np.ndarray,
    tau: float,
    timestamps: np.ndarray | None = None,
    max_gap: float = 0.0,
) -> np.ndarray:
    """Apply causal (forward-only) time-aware EMA to a 1-D array."""
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        if (
            max_gap > 0
            and timestamps is not None
            and abs(timestamps[i] - timestamps[i - 1]) > max_gap
        ):
            out[i] = values[i]
        else:
            dt = (
                abs(float(timestamps[i] - timestamps[i - 1]))
                if timestamps is not None
                else 1.0
            )
            alpha = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _apply_ema(
    values: np.ndarray,
    tau: float,
    timestamps: np.ndarray | None = None,
    max_gap: float = 0.0,
) -> np.ndarray:
    """Apply zero-phase bidirectional EMA to a 1-D array.

    Runs a forward EMA and a backward EMA, then averages the two. This
    eliminates the temporal lag inherent in causal filtering, making it
    ideal for offline post-processing.

    Uses a time constant ``tau`` (seconds) so that the effective smoothing
    adapts to the actual frame rate:

        effective_alpha = 1 - exp(-dt / tau)

    Args:
        values: Raw signal values.
        tau: Time constant in seconds. Controls smoothing strength.
            Smaller = more responsive, larger = heavier smoothing.
        timestamps: Timestamps for each value. Required for time-aware
            smoothing; if None, assumes dt=1 for every step.
        max_gap: If > 0, reset EMA when consecutive timestamp gap exceeds this.

    Returns:
        Smoothed array of same length with zero phase lag.
    """
    fwd = _apply_ema_forward(values, tau, timestamps, max_gap)
    ts_rev = timestamps[::-1].copy() if timestamps is not None else None
    bwd = _apply_ema_forward(values[::-1].copy(), tau, ts_rev, max_gap)[::-1]
    return (fwd + bwd) / 2.0


def smooth_pose_3d(
    pose_df: pd.DataFrame,
    tau: float = 0.15,
    max_gap: float = 2.0,
    conf_gate: float = 0.0,
) -> pd.DataFrame:
    """Apply per-keypoint time-aware EMA smoothing to pose position columns.

    Smooths x_m, y_m, z_m independently for each (track_id, keypoint_name)
    group. Does NOT smooth keypoint_confidence. Resets EMA state when
    consecutive observations are separated by more than max_gap seconds.

    When ``conf_gate`` > 0, only rows with keypoint_confidence below that
    threshold are smoothed; high-confidence rows keep their raw positions.

    Args:
        pose_df: DataFrame with pose_3d schema columns.
        tau: EMA time constant in seconds. Smaller = more responsive.
        max_gap: Reset EMA when timestamp gap exceeds this (seconds).
        conf_gate: Confidence threshold for gating. Rows with confidence
            >= this value keep their raw positions. 0 disables gating.

    Returns:
        Smoothed DataFrame with identical columns and row count.
    """
    if pose_df.empty:
        return pose_df.copy()

    groups: List[pd.DataFrame] = []
    for _, group in pose_df.groupby(["track_id", "keypoint_name"], sort=False):
        group = group.copy()
        group = group.sort_values("timestamp_s").reset_index(drop=True)
        if len(group) > 1:
            ts = group["timestamp_s"].values.astype(float)
            if conf_gate > 0:
                high_conf = group["keypoint_confidence"].values >= conf_gate
            for col in POSE_POSITION_COLUMNS:
                original = group[col].values.astype(float).copy()
                smoothed = _apply_ema(original.copy(), tau, ts, max_gap)
                if conf_gate > 0:
                    smoothed[high_conf] = original[high_conf]
                group[col] = smoothed
        groups.append(group)

    result = pd.concat(groups, ignore_index=True)
    return result[pose_df.columns.tolist()]


def smooth_tracks_2d(
    tracks_df: pd.DataFrame,
    tau: float = 0.15,
    max_gap: float = 2.0,
) -> pd.DataFrame:
    """Apply per-track time-aware EMA smoothing to bounding box position columns.

    Smooths bbox_x1, bbox_y1, bbox_x2, bbox_y2 independently for each
    track_id group. Does NOT smooth track_confidence. Resets EMA state
    when consecutive observations are separated by more than max_gap seconds.

    Args:
        tracks_df: DataFrame with tracks_2d schema columns.
        tau: EMA time constant in seconds. Smaller = more responsive.
        max_gap: Reset EMA when timestamp gap exceeds this (seconds).

    Returns:
        Smoothed DataFrame with identical columns and row count.
    """
    if tracks_df.empty:
        return tracks_df.copy()

    groups: List[pd.DataFrame] = []
    for _, group in tracks_df.groupby("track_id", sort=False):
        group = group.copy()
        group = group.sort_values("timestamp_s").reset_index(drop=True)
        if len(group) > 1:
            ts = group["timestamp_s"].values.astype(float)
            for col in TRACK_POSITION_COLUMNS:
                group[col] = _apply_ema(
                    group[col].values.astype(float), tau, ts, max_gap
                )
        groups.append(group)

    result = pd.concat(groups, ignore_index=True)
    return result[tracks_df.columns.tolist()]


def _run_lengths(mask: np.ndarray) -> List[int]:
    """Return lengths of consecutive True runs in a boolean array."""
    if mask.size == 0:
        return []
    runs: List[int] = []
    current = 0
    for val in mask:
        if val:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return runs


def _short_run_mask(mask: np.ndarray, max_run: int) -> np.ndarray:
    """Return a mask that is True only where ``mask`` has runs <= max_run."""
    out = np.zeros_like(mask, dtype=bool)
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            length = i - start
            if length <= max_run:
                out[start:i] = True
        else:
            i += 1
    return out


def infill_low_confidence_keypoints(
    pose_df: pd.DataFrame,
    conf_threshold: float = 0.3,
    max_infill_run: int = 3,
) -> pd.DataFrame:
    """Replace low-confidence keypoint positions with interpolated neighbors.

    For each (track_id, keypoint_name) time series, rows where
    keypoint_confidence < conf_threshold have their x_m, y_m, z_m replaced
    by linear interpolation from the nearest high-confidence observations.
    Only runs of consecutive low-confidence frames that are <= max_infill_run
    long are infilled; longer runs are left unchanged (their positions are
    too unreliable to interpolate). If no high-confidence neighbors exist,
    values are left unchanged.

    Args:
        pose_df: DataFrame with pose_3d schema columns.
        conf_threshold: Confidence below which keypoints are replaced.
        max_infill_run: Maximum consecutive low-confidence frames to infill.
            Runs longer than this are left as-is.

    Returns:
        DataFrame with infilled positions and capped confidence.
    """
    if pose_df.empty:
        return pose_df.copy()

    # Always report an ``interpolated`` flag, even when the input predates the
    # interpolation stage (e.g. a direct unit-test call).
    out_columns = list(pose_df.columns)
    if "interpolated" not in out_columns:
        out_columns.append("interpolated")

    groups: List[pd.DataFrame] = []
    for _, group in pose_df.groupby(["track_id", "keypoint_name"], sort=False):
        group = group.copy()
        group = group.sort_values("timestamp_s").reset_index(drop=True)
        if "interpolated" not in group.columns:
            group["interpolated"] = False

        # Infill within unbroken stretches only, interpolating in time, so a
        # low-confidence keypoint is never pulled toward where the person was
        # after a gap (another block, or minutes later).
        times = group["timestamp_s"].to_numpy(dtype=float)
        diffs = np.diff(times)
        step = float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 0.0
        stretch = (
            np.concatenate([[0], np.cumsum(diffs > 1.5 * step)])
            if step
            else (np.zeros(len(group), dtype=int))
        )
        for stretch_id in np.unique(stretch):
            rows = np.flatnonzero(stretch == stretch_id)
            part = group.iloc[rows]
            low_mask = (part["keypoint_confidence"] < conf_threshold).to_numpy()
            high_mask = ~low_mask
            # Only infill short runs of low-confidence frames.
            infill_mask = _short_run_mask(low_mask, max_infill_run)
            if not (infill_mask.any() and high_mask.any()):
                continue
            index = group.index[rows]
            for col in POSE_POSITION_COLUMNS:
                series = pd.Series(part[col].to_numpy(dtype=float), index=times[rows])
                series[infill_mask] = np.nan
                series = series.interpolate(method="index", limit_direction="both")
                group.loc[index, col] = series.to_numpy()
            group.loc[index[infill_mask], "keypoint_confidence"] = min(
                conf_threshold,
                part.loc[high_mask, "keypoint_confidence"].min(),
            )
            group.loc[index[infill_mask], "interpolated"] = True

        groups.append(group)

    result = pd.concat(groups, ignore_index=True)
    return result[out_columns]


def smooth_camera_outputs(config: SmoothingConfig) -> Dict[str, object]:
    """Run infill + smoothing on a camera output directory.

    Args:
        config: Smoothing configuration.

    Returns:
        Summary dict with row counts and settings.
    """
    camera_dir = Path(config.camera_dir)
    if not camera_dir.exists():
        raise FileNotFoundError(f"camera_dir does not exist: {camera_dir}")

    pose_path = camera_dir / config.pose_input
    tracks_path = camera_dir / config.tracks_input
    if not pose_path.exists():
        raise FileNotFoundError(f"pose input not found: {pose_path}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks input not found: {tracks_path}")

    pose_df = pd.read_csv(pose_path)
    tracks_df = pd.read_csv(tracks_path)

    # Convert the time-based infill gap to a frame count using the observed
    # timestamp spacing, so the same wall-clock gap applies at any frame rate.
    unique_ts = np.sort(pose_df["timestamp_s"].astype(float).unique())
    median_dt = float(np.median(np.diff(unique_ts))) if unique_ts.size > 1 else 0.0
    max_infill_run = (
        max(1, int(round(config.max_infill_gap_s / median_dt))) if median_dt > 0 else 3
    )

    # Infill low-confidence keypoints first, then smooth.
    pose_df = infill_low_confidence_keypoints(
        pose_df, config.conf_threshold, max_infill_run
    )
    pose_smooth = smooth_pose_3d(pose_df, config.tau, conf_gate=config.conf_gate)
    tracks_smooth = smooth_tracks_2d(tracks_df, config.tau)

    pose_smooth.to_csv(camera_dir / config.pose_output, index=False)
    tracks_smooth.to_csv(camera_dir / config.tracks_output, index=False)

    summary: Dict[str, object] = {
        "config": asdict(config),
        "tau": config.tau,
        "conf_threshold": config.conf_threshold,
        "conf_gate": config.conf_gate,
        "pose_rows": len(pose_smooth),
        "tracks_rows": len(tracks_smooth),
    }
    (camera_dir / config.summary_output).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for video-smooth."""
    parser = argparse.ArgumentParser(
        prog="video-smooth",
        description="Apply temporal EMA smoothing and confidence-gated infill.",
    )
    parser.add_argument("--camera-dir", required=True, type=str)
    parser.add_argument("--tau", default=0.15, type=float)
    parser.add_argument("--conf-threshold", default=0.3, type=float)
    parser.add_argument(
        "--conf-gate",
        default=0.3,
        type=float,
        help="Only smooth keypoints below this confidence; high-confidence "
        "positions pass through unchanged. 0 disables gating.",
    )
    parser.add_argument(
        "--max-infill-gap-s",
        default=1.0,
        type=float,
        help="Longest run of low-confidence frames to infill, in seconds.",
    )
    parser.add_argument("--pose-input", default="pose_3d_interpolated.csv", type=str)
    parser.add_argument(
        "--tracks-input", default="tracks_2d_interpolated.csv", type=str
    )
    parser.add_argument("--pose-output", default="pose_3d_smooth.csv", type=str)
    parser.add_argument("--tracks-output", default="tracks_2d_smooth.csv", type=str)
    parser.add_argument("--summary-output", default="smoothing_summary.json", type=str)
    return parser


def main() -> None:
    """CLI entry point for video-smooth."""
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = SmoothingConfig(
        camera_dir=args.camera_dir,
        tau=args.tau,
        conf_threshold=args.conf_threshold,
        conf_gate=args.conf_gate,
        max_infill_gap_s=args.max_infill_gap_s,
        pose_input=args.pose_input,
        tracks_input=args.tracks_input,
        pose_output=args.pose_output,
        tracks_output=args.tracks_output,
        summary_output=args.summary_output,
    )
    smooth_camera_outputs(cfg)


if __name__ == "__main__":
    main()
