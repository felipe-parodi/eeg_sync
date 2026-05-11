"""Render 3D pose skeletons from pose_3d CSV using matplotlib."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from video_analysis.visualize_pose_tracks import COCO17_EDGES, TRACK_COLORS

matplotlib.use("Agg")  # non-interactive backend for headless rendering


@dataclass
class Visualize3DConfig:
    """Configuration for 3D pose skeleton rendering."""

    pose_csv: str
    output_dir: str
    start_frame: int = 0
    max_frames: Optional[int] = None
    snapshot_interval: int = 1
    output_video: Optional[str] = None
    output_fps: float = 6.0
    keypoint_conf_thresh: float = 0.2
    elev: float = 20.0
    azim: float = -60.0
    rotate: bool = False
    figsize: Tuple[float, float] = (8.0, 6.0)
    dpi: int = 100


def _bgr_to_rgb_normalized(bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert BGR (0-255) to RGB (0.0-1.0) for matplotlib."""
    return (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)


def _color_for_track_mpl(track_id: int) -> Tuple[float, float, float]:
    """Return matplotlib RGB color for a track_id."""
    bgr = TRACK_COLORS.get(track_id, (180, 180, 180))
    return _bgr_to_rgb_normalized(bgr)


def _keypoint_name_to_index(name: str) -> int:
    """Map keypoint name string to COCO-17 integer index.

    Args:
        name: Keypoint name string (e.g. 'kp_000' or 'kp_016').

    Returns:
        Integer index 0-16.

    Raises:
        ValueError: If name is not recognized.
    """
    if name.startswith("kp_"):
        return int(name[3:])
    raise ValueError(f"Unknown keypoint name: {name!r}")


def load_pose_data(
    pose_csv: str,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> pd.DataFrame:
    """Load and filter pose_3d CSV data.

    Args:
        pose_csv: Path to the pose CSV file.
        start_frame: First frame_idx to include.
        max_frames: Maximum number of distinct frames to include.

    Returns:
        Filtered DataFrame with pose data.

    Raises:
        FileNotFoundError: If pose_csv does not exist.
    """
    path = Path(pose_csv)
    if not path.exists():
        raise FileNotFoundError(f"Pose CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"frame_idx", "track_id", "keypoint_name", "x_m", "y_m", "z_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if start_frame > 0:
        df = df[df["frame_idx"] >= start_frame]

    if max_frames is not None:
        unique_frames = sorted(df["frame_idx"].unique())
        keep_frames = set(unique_frames[:max_frames])
        df = df[df["frame_idx"].isin(keep_frames)]

    return df.reset_index(drop=True)


def compute_axis_limits(
    pose_df: pd.DataFrame,
    padding_frac: float = 0.1,
) -> Dict[str, Tuple[float, float]]:
    """Compute fixed axis limits across all frames.

    Args:
        pose_df: Filtered pose DataFrame.
        padding_frac: Fractional padding to add to each axis range.

    Returns:
        Dict with keys 'x', 'y', 'z', each mapping to (min, max) tuple.
    """
    limits = {}
    for col, key in [("x_m", "x"), ("y_m", "y"), ("z_m", "z")]:
        vmin = float(pose_df[col].min())
        vmax = float(pose_df[col].max())
        span = max(vmax - vmin, 1e-6)
        pad = span * padding_frac
        limits[key] = (vmin - pad, vmax + pad)
    return limits


def _render_frame_3d(
    frame_df: pd.DataFrame,
    frame_idx: int,
    axis_limits: Dict[str, Tuple[float, float]],
    config: Visualize3DConfig,
    azim_override: Optional[float] = None,
) -> plt.Figure:
    """Render a single frame's 3D skeleton plot.

    Args:
        frame_df: Pose rows for a single frame_idx.
        frame_idx: The frame index (for title display).
        axis_limits: Fixed axis limits from compute_axis_limits.
        config: Visualization configuration.
        azim_override: Override azimuth (for rotation mode).

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Lighten grid lines and pane backgrounds.
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    ax.xaxis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.15)
    ax.yaxis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.15)
    ax.zaxis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.15)

    # Draw ground plane at y_max (bottom of image = floor).
    # Axis mapping: mpl_X = x_m, mpl_Y = z_m (depth), mpl_Z = -y_m (up).
    import numpy as np

    # Flipped Y limits: negate and swap so min < max.
    y_up_lo = -axis_limits["y"][1]
    y_up_hi = -axis_limits["y"][0]

    y_floor = y_up_lo  # bottom of scene (max pixel y, negated)
    xx = np.array([axis_limits["x"][0], axis_limits["x"][1]])
    zz_depth = np.array([axis_limits["z"][0], axis_limits["z"][1]])
    xx_grid, zz_grid = np.meshgrid(xx, zz_depth)
    yy_grid = np.full_like(xx_grid, y_floor)
    ax.plot_surface(xx_grid, zz_grid, yy_grid, alpha=0.08, color="gray")

    for track_id, group in frame_df.groupby("track_id"):
        color = _color_for_track_mpl(int(track_id))

        # Build keypoint index -> (x, y_up, z_depth, conf) mapping.
        # mpl_X = x_m, mpl_Y = z_m (depth), mpl_Z = -y_m (up).
        kp_map: Dict[int, Tuple[float, float, float, float]] = {}
        for _, row in group.iterrows():
            try:
                kp_idx = _keypoint_name_to_index(str(row["keypoint_name"]))
            except ValueError:
                continue
            conf = float(row.get("keypoint_confidence", 1.0))
            if conf < config.keypoint_conf_thresh:
                continue
            kp_map[kp_idx] = (
                float(row["x_m"]),
                float(row["z_m"]),
                -float(row["y_m"]),
                conf,
            )

        # Draw skeleton edges.
        for i, j in COCO17_EDGES:
            if i in kp_map and j in kp_map:
                p1, p2 = kp_map[i], kp_map[j]
                ax.plot3D(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color,
                    linewidth=2,
                )

        # Draw keypoint scatter.
        if kp_map:
            xs = [v[0] for v in kp_map.values()]
            ys = [v[1] for v in kp_map.values()]
            zs = [v[2] for v in kp_map.values()]
            ax.scatter3D(xs, ys, zs, color=color, s=20)

    ax.set_xlim(axis_limits["x"])
    ax.set_ylim(axis_limits["z"])
    ax.set_zlim(y_up_lo, y_up_hi)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Z (depth)")
    ax.set_zlabel("Y (up)")
    ax.set_title(f"Frame {frame_idx}", fontsize=12, fontweight="bold")
    ax.view_init(elev=config.elev, azim=azim_override or config.azim)

    fig.tight_layout()
    return fig


def render_pose_3d(config: Visualize3DConfig) -> Dict[str, Any]:
    """Render 3D pose skeletons from pose CSV.

    Produces PNG snapshots at configurable intervals. Optionally stitches
    them into a video using cv2.VideoWriter.

    Args:
        config: Visualization configuration.

    Returns:
        Summary dict with frame counts and output paths.
    """
    pose_df = load_pose_data(
        config.pose_csv,
        start_frame=config.start_frame,
        max_frames=config.max_frames,
    )

    if pose_df.empty:
        return {"rendered_frames": 0, "output_dir": config.output_dir}

    axis_limits = compute_axis_limits(pose_df)
    unique_frames = sorted(pose_df["frame_idx"].unique())

    # Apply snapshot interval.
    selected_frames = unique_frames[:: config.snapshot_interval]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    total = len(selected_frames)
    saved_paths: List[str] = []

    for i, frame_idx in enumerate(selected_frames):
        frame_df = pose_df[pose_df["frame_idx"] == frame_idx]

        azim_override = None
        if config.rotate and total > 1:
            azim_override = config.azim + (i / total) * 360.0

        fig = _render_frame_3d(frame_df, frame_idx, axis_limits, config, azim_override)
        png_path = output_dir / f"pose3d_frame_{frame_idx:06d}.png"
        fig.savefig(str(png_path), dpi=config.dpi)
        plt.close(fig)
        saved_paths.append(str(png_path))
        rendered += 1

        if rendered % 50 == 0:
            print(f"[video-visualize-3d] rendered {rendered}/{total} frames")

    summary: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "rendered_frames": rendered,
        "total_unique_frames": len(unique_frames),
        "snapshot_interval": config.snapshot_interval,
    }

    # Optionally stitch into video.
    if config.output_video and saved_paths:
        import cv2

        first_img = cv2.imread(saved_paths[0])
        if first_img is not None:
            h, w = first_img.shape[:2]
            video_path = Path(config.output_video)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                config.output_fps,
                (w, h),
            )
            for png_path in saved_paths:
                img = cv2.imread(png_path)
                if img is not None:
                    writer.write(img)
            writer.release()
            summary["output_video"] = str(video_path)

    print(f"[video-visualize-3d] done: {rendered} frames rendered")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for video-visualize-3d."""
    parser = argparse.ArgumentParser(
        prog="video-visualize-3d",
        description="Render 3D pose skeletons from pose_3d CSV data.",
    )
    parser.add_argument("--pose-csv", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--start-frame", default=0, type=int)
    parser.add_argument("--max-frames", default=None, type=int)
    parser.add_argument("--snapshot-interval", default=1, type=int)
    parser.add_argument("--output-video", default=None, type=str)
    parser.add_argument("--output-fps", default=6.0, type=float)
    parser.add_argument("--keypoint-conf-thresh", default=0.2, type=float)
    parser.add_argument("--elev", default=20.0, type=float)
    parser.add_argument("--azim", default=-60.0, type=float)
    parser.add_argument("--rotate", action="store_true", default=False)
    parser.add_argument("--figsize", default="8,6", type=str)
    parser.add_argument("--dpi", default=100, type=int)
    return parser


def main() -> None:
    """CLI entry point for video-visualize-3d."""
    parser = build_arg_parser()
    args = parser.parse_args()
    figsize = tuple(float(v) for v in args.figsize.split(","))
    cfg = Visualize3DConfig(
        pose_csv=args.pose_csv,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        snapshot_interval=args.snapshot_interval,
        output_video=args.output_video,
        output_fps=args.output_fps,
        keypoint_conf_thresh=args.keypoint_conf_thresh,
        elev=args.elev,
        azim=args.azim,
        rotate=args.rotate,
        figsize=figsize,
        dpi=args.dpi,
    )
    summary = render_pose_3d(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
