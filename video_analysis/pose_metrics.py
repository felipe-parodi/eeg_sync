"""Per-block pose metrics: torso proximity and movement synchrony.

Computes torso-centroid distance and movement cross-correlation for
parent-child dyads, then generates per-block comparison plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaze_analysis.config import SessionBlock, load_session_config
from gaze_analysis.synchrony import (
    compute_movement_xcorr,
    compute_torso_proximity,
)

# ---------------------------------------------------------------------------
# Nature journal style
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Colorblind-safe block colors
BLOCK_COLORS = {
    "grocery": "#2196F3",
    "synchrony": "#4CAF50",
    "storybook": "#FF9800",
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_per_block_metrics(
    pose_df: pd.DataFrame,
    blocks: List[SessionBlock],
    parent_id: int = 0,
    child_id: int = 1,
    xcorr_window_s: float = 5.0,
    xcorr_max_lag_s: float = 2.0,
) -> Dict[str, Any]:
    """Compute torso proximity and movement xcorr per block.

    Args:
        pose_df: Smoothed pose DataFrame.
        blocks: Session block definitions.
        parent_id: Parent track ID.
        child_id: Child track ID.
        xcorr_window_s: Cross-correlation window size in seconds.
        xcorr_max_lag_s: Maximum lag to search in seconds.

    Returns:
        Dict with keys: proximity_frames, xcorr_windows, block_stats.
    """
    all_proximity: List[pd.DataFrame] = []
    all_xcorr: List[pd.DataFrame] = []
    block_stats: List[Dict[str, Any]] = []

    for block in blocks:
        block_pose = pose_df[
            (pose_df["timestamp_s"] >= block.start_s)
            & (pose_df["timestamp_s"] <= block.end_s)
        ].copy()

        if block_pose.empty:
            continue

        # Torso proximity
        prox = compute_torso_proximity(block_pose, parent_id, child_id)
        if not prox.empty:
            prox["block"] = block.name
            all_proximity.append(prox)

        # Movement cross-correlation
        xcorr = compute_movement_xcorr(
            block_pose, parent_id, child_id, xcorr_window_s, xcorr_max_lag_s
        )
        if not xcorr.empty:
            xcorr["block"] = block.name
            all_xcorr.append(xcorr)

        # Per-block summary
        stats: Dict[str, Any] = {"block": block.name}
        if not prox.empty:
            dist = prox["torso_distance_px"]
            stats["proximity_mean"] = float(dist.mean())
            stats["proximity_std"] = float(dist.std())
            stats["proximity_median"] = float(dist.median())
            stats["proximity_n_frames"] = len(dist)
        if not xcorr.empty:
            stats["xcorr_mean"] = float(xcorr["peak_xcorr"].mean())
            stats["xcorr_std"] = float(xcorr["peak_xcorr"].std())
            stats["lag_mean_s"] = float(xcorr["peak_lag_s"].mean())
            stats["lag_std_s"] = float(xcorr["peak_lag_s"].std())
            stats["xcorr_n_windows"] = len(xcorr)
        block_stats.append(stats)

    proximity_df = pd.concat(all_proximity, ignore_index=True) if all_proximity else pd.DataFrame()
    xcorr_df = pd.concat(all_xcorr, ignore_index=True) if all_xcorr else pd.DataFrame()

    return {
        "proximity_df": proximity_df,
        "xcorr_df": xcorr_df,
        "block_stats": block_stats,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_proximity(
    proximity_df: pd.DataFrame,
    blocks: List[SessionBlock],
    output_path: Path,
) -> None:
    """Figure 1: Torso proximity — boxplot comparison + per-block time series.

    Args:
        proximity_df: Per-frame proximity with 'block' column.
        blocks: Session block definitions.
        output_path: Base path for output (without extension).
    """
    block_names = [b.name for b in blocks if b.name in proximity_df["block"].unique()]
    n_blocks = len(block_names)
    if n_blocks == 0:
        return

    fig, axes = plt.subplots(1, n_blocks + 1, figsize=(4 + 3.5 * n_blocks, 4),
                             gridspec_kw={"width_ratios": [1.2] + [1] * n_blocks})

    # Left panel: boxplot comparison
    ax_box = axes[0]
    box_data = [
        proximity_df[proximity_df["block"] == name]["torso_distance_px"].values
        for name in block_names
    ]
    colors = [BLOCK_COLORS.get(name, "#999") for name in block_names]
    bp = ax_box.boxplot(box_data, labels=block_names, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_box.set_ylabel("Torso distance (px)")
    ax_box.set_title("Per-block comparison")

    # Right panels: per-block time series
    y_max = proximity_df["torso_distance_px"].quantile(0.99)
    for i, name in enumerate(block_names):
        ax = axes[i + 1]
        block_data = proximity_df[proximity_df["block"] == name].sort_values("timestamp_s")
        color = BLOCK_COLORS.get(name, "#999")

        # Plot relative time (seconds into block)
        block_start = blocks[[b.name for b in blocks].index(name)].start_s
        t_rel = block_data["timestamp_s"] - block_start

        ax.plot(t_rel, block_data["torso_distance_px"], color=color, alpha=0.3, linewidth=0.5)
        # Rolling mean for readability
        window = min(90, len(block_data) // 4) if len(block_data) > 10 else 1
        if window > 1:
            rolling = block_data["torso_distance_px"].rolling(window, center=True).mean()
            ax.plot(t_rel, rolling, color=color, linewidth=1.5)
        ax.set_ylim(0, y_max * 1.05)
        ax.set_xlabel("Time in block (s)")
        ax.set_title(name.capitalize())
        if i > 0:
            ax.set_yticklabels([])

    fig.suptitle("Parent-Child Torso Proximity", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_path) + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[pose-metrics] saved {output_path}.png/.svg")


def plot_synchrony(
    xcorr_df: pd.DataFrame,
    blocks: List[SessionBlock],
    output_path: Path,
) -> None:
    """Figure 2: Movement synchrony — bar charts of xcorr strength + lag per block.

    Args:
        xcorr_df: Per-window xcorr results with 'block' column.
        blocks: Session block definitions.
        output_path: Base path for output (without extension).
    """
    block_names = [b.name for b in blocks if b.name in xcorr_df["block"].unique()]
    if not block_names:
        return

    fig, (ax_xcorr, ax_lag) = plt.subplots(1, 2, figsize=(8, 4))

    x = np.arange(len(block_names))
    colors = [BLOCK_COLORS.get(name, "#999") for name in block_names]

    # Left: mean peak xcorr per block
    means = []
    stds = []
    for name in block_names:
        block_data = xcorr_df[xcorr_df["block"] == name]["peak_xcorr"]
        means.append(float(block_data.mean()))
        stds.append(float(block_data.std()))

    ax_xcorr.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5, edgecolor="black", linewidth=0.5)
    ax_xcorr.set_xticks(x)
    ax_xcorr.set_xticklabels([n.capitalize() for n in block_names])
    ax_xcorr.set_ylabel("Peak cross-correlation")
    ax_xcorr.set_title("Movement synchrony")
    ax_xcorr.set_ylim(0, 1)

    # Right: mean lag per block
    lag_means = []
    lag_stds = []
    for name in block_names:
        block_data = xcorr_df[xcorr_df["block"] == name]["peak_lag_s"]
        lag_means.append(float(block_data.mean()))
        lag_stds.append(float(block_data.std()))

    bar_colors = ["#e57373" if m > 0 else "#64B5F6" for m in lag_means]
    ax_lag.bar(x, lag_means, yerr=lag_stds, color=bar_colors, alpha=0.7, capsize=5, edgecolor="black", linewidth=0.5)
    ax_lag.set_xticks(x)
    ax_lag.set_xticklabels([n.capitalize() for n in block_names])
    ax_lag.set_ylabel("Mean lag (s)")
    ax_lag.set_title("Lead-follow (+ = child follows)")
    ax_lag.axhline(0, color="black", linewidth=0.5, linestyle="--")

    fig.suptitle("Parent-Child Movement Synchrony", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_path) + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[pose-metrics] saved {output_path}.png/.svg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="video-pose-metrics",
        description="Compute per-block torso proximity and movement synchrony.",
    )
    parser.add_argument("--camera-dir", required=True, type=str)
    parser.add_argument("--session-config", required=True, type=str)
    parser.add_argument("--camera-id", default="camera_a", type=str)
    parser.add_argument("--pose-input", default="pose_3d_smooth.csv", type=str)
    parser.add_argument("--xcorr-window", default=5.0, type=float,
                        help="Cross-correlation window size in seconds.")
    parser.add_argument("--xcorr-max-lag", default=2.0, type=float,
                        help="Max lag to search in seconds.")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    camera_dir = Path(args.camera_dir)
    config = load_session_config(args.session_config)
    mapping = config.get_camera_mapping(args.camera_id)
    blocks = config.session_blocks

    pose_df = pd.read_csv(camera_dir / args.pose_input)
    print(f"[pose-metrics] loaded {len(pose_df)} pose rows from {args.pose_input}")

    results = compute_per_block_metrics(
        pose_df,
        blocks,
        parent_id=mapping.parent_track_id,
        child_id=mapping.child_track_id,
        xcorr_window_s=args.xcorr_window,
        xcorr_max_lag_s=args.xcorr_max_lag,
    )

    # Save CSVs
    if not results["proximity_df"].empty:
        prox_path = camera_dir / "pose_metrics_proximity.csv"
        results["proximity_df"].to_csv(prox_path, index=False)
        print(f"[pose-metrics] wrote {prox_path} ({len(results['proximity_df'])} rows)")

    if not results["xcorr_df"].empty:
        xcorr_path = camera_dir / "pose_metrics_xcorr.csv"
        results["xcorr_df"].to_csv(xcorr_path, index=False)
        print(f"[pose-metrics] wrote {xcorr_path} ({len(results['xcorr_df'])} rows)")

    # Save summary JSON
    summary_path = camera_dir / "pose_metrics_summary.json"
    summary_path.write_text(
        json.dumps(results["block_stats"], indent=2), encoding="utf-8"
    )
    print(f"[pose-metrics] wrote {summary_path}")

    # Print summary table
    for stats in results["block_stats"]:
        block = stats["block"]
        prox_mean = stats.get("proximity_mean", 0)
        prox_std = stats.get("proximity_std", 0)
        xcorr_mean = stats.get("xcorr_mean", 0)
        lag_mean = stats.get("lag_mean_s", 0)
        print(
            f"  {block:12s}: proximity={prox_mean:.1f} +/- {prox_std:.1f} px, "
            f"xcorr={xcorr_mean:.3f}, lag={lag_mean:.3f}s"
        )

    # Generate plots
    if not results["proximity_df"].empty:
        plot_proximity(results["proximity_df"], blocks, camera_dir / "pose_proximity")

    if not results["xcorr_df"].empty:
        plot_synchrony(results["xcorr_df"], blocks, camera_dir / "pose_synchrony")


if __name__ == "__main__":
    main()
