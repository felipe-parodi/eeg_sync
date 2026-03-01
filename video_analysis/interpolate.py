"""Interpolate low-FPS video inference outputs to a higher target FPS."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

TRACK_NUMERIC_COLUMNS = [
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "track_confidence",
]
POSE_NUMERIC_COLUMNS = ["x_m", "y_m", "z_m", "keypoint_confidence"]


@dataclass
class InterpolationConfig:
    """Interpolation config for one camera output directory."""

    camera_dir: str
    target_fps: float = 8.0
    tracks_input: str = "tracks_2d.csv"
    pose_input: str = "pose_3d.csv"
    tracks_output: str = "tracks_2d_interpolated.csv"
    pose_output: str = "pose_3d_interpolated.csv"
    summary_output: str = "interpolation_summary.json"


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _build_target_timestamps(
    tracks_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    target_fps: float,
) -> np.ndarray:
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    times: List[float] = []
    if not tracks_df.empty:
        times.extend(tracks_df["timestamp_s"].astype(float).tolist())
    if not pose_df.empty:
        times.extend(pose_df["timestamp_s"].astype(float).tolist())

    if not times:
        return np.asarray([], dtype=float)

    start = float(min(times))
    end = float(max(times))
    step = 1.0 / target_fps
    return np.arange(start, end + (step * 0.5), step, dtype=float)


def _infer_source_fps(df: pd.DataFrame) -> float:
    if df.empty or "timestamp_s" not in df.columns:
        return 0.0
    unique_ts = np.sort(df["timestamp_s"].astype(float).unique())
    if unique_ts.size < 2:
        return 0.0
    diffs = np.diff(unique_ts)
    diffs = diffs[diffs > 1e-9]
    if diffs.size == 0:
        return 0.0
    return float(1.0 / np.median(diffs))


def _mode_or_default(series: pd.Series, default: str) -> str:
    mode = series.mode(dropna=True)
    if mode.empty:
        return default
    return str(mode.iloc[0])


def _interpolate_group(
    group_df: pd.DataFrame,
    target_timestamps: np.ndarray,
    numeric_columns: List[str],
) -> pd.DataFrame:
    indexed = (
        group_df.sort_values("timestamp_s")
        .drop_duplicates(subset=["timestamp_s"], keep="last")
        .set_index("timestamp_s")
    )
    expanded = indexed.reindex(target_timestamps)
    expanded[numeric_columns] = expanded[numeric_columns].interpolate(
        method="linear",
        limit_direction="both",
        limit_area="inside",
    )
    expanded["timestamp_s"] = expanded.index.astype(float)
    return expanded.reset_index(drop=True)


def interpolate_tracks_2d(
    tracks_df: pd.DataFrame,
    target_timestamps: np.ndarray,
    target_fps: float,
) -> pd.DataFrame:
    _validate_required_columns(
        tracks_df,
        [
            "frame_idx",
            "timestamp_s",
            "track_id",
            "track_label",
            *TRACK_NUMERIC_COLUMNS,
        ],
        "tracks_2d",
    )
    if tracks_df.empty or target_timestamps.size == 0:
        return tracks_df.copy()

    rows: List[pd.DataFrame] = []
    for track_id, group in tracks_df.groupby("track_id", dropna=False):
        group = group.copy()
        out = _interpolate_group(group, target_timestamps, TRACK_NUMERIC_COLUMNS)
        out["track_id"] = int(track_id)
        out["track_label"] = _mode_or_default(group["track_label"], "unknown")
        out["frame_idx"] = np.rint(out["timestamp_s"] * target_fps).astype(int)
        out = out.dropna(subset=TRACK_NUMERIC_COLUMNS)
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=tracks_df.columns)

    merged = pd.concat(rows, ignore_index=True)
    merged = merged[
        [
            "frame_idx",
            "timestamp_s",
            "track_id",
            "track_label",
            *TRACK_NUMERIC_COLUMNS,
        ]
    ].sort_values(["frame_idx", "track_id"], ignore_index=True)
    return merged


def interpolate_pose_3d(
    pose_df: pd.DataFrame,
    target_timestamps: np.ndarray,
    target_fps: float,
) -> pd.DataFrame:
    _validate_required_columns(
        pose_df,
        [
            "frame_idx",
            "timestamp_s",
            "track_id",
            "track_label",
            "keypoint_name",
            *POSE_NUMERIC_COLUMNS,
        ],
        "pose_3d",
    )
    if pose_df.empty or target_timestamps.size == 0:
        return pose_df.copy()

    rows: List[pd.DataFrame] = []
    grouped = pose_df.groupby(["track_id", "keypoint_name"], dropna=False)
    for (track_id, keypoint_name), group in grouped:
        group = group.copy()
        out = _interpolate_group(group, target_timestamps, POSE_NUMERIC_COLUMNS)
        out["track_id"] = int(track_id)
        out["track_label"] = _mode_or_default(group["track_label"], "unknown")
        out["keypoint_name"] = str(keypoint_name)
        out["frame_idx"] = np.rint(out["timestamp_s"] * target_fps).astype(int)
        out = out.dropna(subset=POSE_NUMERIC_COLUMNS)
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=pose_df.columns)

    merged = pd.concat(rows, ignore_index=True)
    merged = merged[
        [
            "frame_idx",
            "timestamp_s",
            "track_id",
            "track_label",
            "keypoint_name",
            *POSE_NUMERIC_COLUMNS,
        ]
    ].sort_values(["frame_idx", "track_id", "keypoint_name"], ignore_index=True)
    return merged


def interpolate_camera_outputs(config: InterpolationConfig) -> Dict[str, object]:
    camera_dir = Path(config.camera_dir)
    if not camera_dir.exists():
        raise FileNotFoundError(f"camera_dir does not exist: {camera_dir}")

    tracks_path = camera_dir / config.tracks_input
    pose_path = camera_dir / config.pose_input
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks input not found: {tracks_path}")
    if not pose_path.exists():
        raise FileNotFoundError(f"pose input not found: {pose_path}")

    tracks_df = pd.read_csv(tracks_path)
    pose_df = pd.read_csv(pose_path)
    source_fps = max(_infer_source_fps(tracks_df), _infer_source_fps(pose_df))
    target_timestamps = _build_target_timestamps(
        tracks_df=tracks_df,
        pose_df=pose_df,
        target_fps=config.target_fps,
    )

    tracks_interp = interpolate_tracks_2d(
        tracks_df=tracks_df,
        target_timestamps=target_timestamps,
        target_fps=config.target_fps,
    )
    pose_interp = interpolate_pose_3d(
        pose_df=pose_df,
        target_timestamps=target_timestamps,
        target_fps=config.target_fps,
    )

    tracks_out = camera_dir / config.tracks_output
    pose_out = camera_dir / config.pose_output
    summary_out = camera_dir / config.summary_output
    tracks_interp.to_csv(tracks_out, index=False)
    pose_interp.to_csv(pose_out, index=False)

    summary = {
        "config": asdict(config),
        "source_fps_estimate": source_fps,
        "target_fps": config.target_fps,
        "tracks_rows_input": int(len(tracks_df)),
        "tracks_rows_output": int(len(tracks_interp)),
        "pose_rows_input": int(len(pose_df)),
        "pose_rows_output": int(len(pose_interp)),
        "timestamps_output_count": int(target_timestamps.size),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-interpolate",
        description="Interpolate low-FPS inference outputs to a higher FPS timeline.",
    )
    parser.add_argument("--camera-dir", required=True, type=str)
    parser.add_argument("--target-fps", default=8.0, type=float)
    parser.add_argument("--tracks-input", default="tracks_2d.csv", type=str)
    parser.add_argument("--pose-input", default="pose_3d.csv", type=str)
    parser.add_argument(
        "--tracks-output",
        default="tracks_2d_interpolated.csv",
        type=str,
    )
    parser.add_argument(
        "--pose-output",
        default="pose_3d_interpolated.csv",
        type=str,
    )
    parser.add_argument(
        "--summary-output",
        default="interpolation_summary.json",
        type=str,
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = InterpolationConfig(
        camera_dir=args.camera_dir,
        target_fps=args.target_fps,
        tracks_input=args.tracks_input,
        pose_input=args.pose_input,
        tracks_output=args.tracks_output,
        pose_output=args.pose_output,
        summary_output=args.summary_output,
    )
    interpolate_camera_outputs(cfg)


if __name__ == "__main__":
    main()
