"""Visualization for synchrony metrics with session block coloring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaze_analysis.config import (
    SessionBlock,
    add_time_range_args,
    filter_by_time_range,
    load_session_config,
    resolve_time_range,
)


def _shade_blocks(
    ax: plt.Axes,
    blocks: List[SessionBlock],
    alpha: float = 0.15,
) -> None:
    """Add vertical shading for session blocks.

    Args:
        ax: Matplotlib axes to shade.
        blocks: Session blocks with start/end times and colors.
        alpha: Shading transparency.
    """
    for block in blocks:
        ax.axvspan(
            block.start_s,
            block.end_s,
            alpha=alpha,
            color=block.color,
            label=block.name.replace("_", " "),
        )


def plot_torso_proximity(
    proximity_df: pd.DataFrame,
    blocks: List[SessionBlock],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot torso proximity distance over time.

    Args:
        proximity_df: DataFrame with timestamp_s, torso_distance_px.
        blocks: Session blocks for background shading.
        ax: Axes to plot on (creates one if None).

    Returns:
        The axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(14, 3))

    _shade_blocks(ax, blocks)
    ax.plot(
        proximity_df["timestamp_s"],
        proximity_df["torso_distance_px"],
        linewidth=0.6,
        color="black",
        alpha=0.7,
    )
    ax.set_ylabel("Distance (px)")
    ax.set_title("Torso Proximity")
    ax.set_xlabel("Time (s)")
    return ax


def plot_movement_xcorr(
    xcorr_df: pd.DataFrame,
    blocks: List[SessionBlock],
    axes: Optional[tuple[plt.Axes, plt.Axes]] = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot movement cross-correlation: peak r and peak lag.

    Args:
        xcorr_df: DataFrame with window_start_s, peak_xcorr, peak_lag_s.
        blocks: Session blocks for background shading.
        axes: Pair of axes (creates if None).

    Returns:
        Tuple of (correlation axes, lag axes).
    """
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    ax_r, ax_lag = axes

    mid_time = (xcorr_df["window_start_s"] + xcorr_df["window_end_s"]) / 2

    _shade_blocks(ax_r, blocks)
    ax_r.bar(mid_time, xcorr_df["peak_xcorr"], width=1.0, color="steelblue", alpha=0.7)
    ax_r.set_ylabel("Peak r")
    ax_r.set_title("Movement Cross-Correlation")
    ax_r.set_ylim(0, 1)

    _shade_blocks(ax_lag, blocks)
    ax_lag.bar(mid_time, xcorr_df["peak_lag_s"], width=1.0, color="coral", alpha=0.7)
    ax_lag.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_lag.set_ylabel("Lag (s)")
    ax_lag.set_xlabel("Time (s)")
    ax_lag.set_title("Peak Lag (positive = child follows)")

    return ax_r, ax_lag


def plot_movement_sync(
    sync_df: pd.DataFrame,
    blocks: List[SessionBlock],
    chance: Optional[Dict[str, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot whole-body movement synchrony per 20 s window, with chance levels.

    Args:
        sync_df: ``pose_metrics_sync.csv`` rows (window_start_s, window_end_s,
            sync_r, block).
        blocks: Session blocks for background shading.
        chance: Chance level per block name, drawn as a dashed line across
            that block.
        ax: Axes to draw on (creates if None).

    Returns:
        The axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3))
    _shade_blocks(ax, blocks)
    if not sync_df.empty:
        mid = (sync_df["window_start_s"] + sync_df["window_end_s"]) / 2
        ax.bar(mid, sync_df["sync_r"], width=8.0, color="steelblue", alpha=0.7)
    for block in blocks:
        level = (chance or {}).get(block.name)
        if level is not None and not pd.isna(level):
            ax.hlines(
                level,
                block.start_s,
                block.end_s,
                color="black",
                linestyle="--",
                linewidth=1.2,
            )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("r")
    ax.set_title("Movement synchrony (whole body, 20 s windows; dashed = chance)")
    return ax


def plot_gaze_categories(
    gaze_cat_df: pd.DataFrame,
    blocks: List[SessionBlock],
    window_s: float = 10.0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot gaze category proportions as a stacked area chart.

    Args:
        gaze_cat_df: DataFrame with timestamp_s, gaze_category.
        blocks: Session blocks for background shading.
        window_s: Sliding window duration for proportion computation.
        ax: Axes to plot on (creates one if None).

    Returns:
        The axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(14, 3))

    from gaze_analysis.gaze_categories import CATEGORIES, CATEGORY_COLORS

    category_order = CATEGORIES
    category_colors = CATEGORY_COLORS

    df = gaze_cat_df.sort_values("timestamp_s").reset_index(drop=True)
    timestamps = df["timestamp_s"].values

    if len(timestamps) == 0:
        return ax

    # Compute rolling proportions
    t_min, t_max = timestamps[0], timestamps[-1]
    step = window_s / 2
    window_centers = np.arange(t_min + window_s / 2, t_max - window_s / 2 + step, step)

    proportions = {cat: [] for cat in category_order}
    for tc in window_centers:
        mask = (timestamps >= tc - window_s / 2) & (timestamps < tc + window_s / 2)
        window_cats = df.loc[mask, "gaze_category"]
        total = len(window_cats) if len(window_cats) > 0 else 1
        counts = window_cats.value_counts()
        for cat in category_order:
            proportions[cat].append(counts.get(cat, 0) / total)

    # Stack
    bottom = np.zeros(len(window_centers))
    _shade_blocks(ax, blocks)
    for cat in category_order:
        vals = np.array(proportions[cat])
        ax.fill_between(
            window_centers,
            bottom,
            bottom + vals,
            alpha=0.8,
            color=category_colors[cat],
            label=cat.replace("_", " "),
        )
        bottom += vals

    ax.set_ylabel("Proportion")
    ax.set_title("Gaze Categories")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7)
    return ax


def plot_gaze_convergence(
    convergence_df: pd.DataFrame,
    blocks: List[SessionBlock],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot gaze convergence score over time.

    Args:
        convergence_df: DataFrame with timestamp_s, gaze_convergence_score.
        blocks: Session blocks for background shading.
        ax: Axes to plot on (creates one if None).

    Returns:
        The axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(14, 3))

    _shade_blocks(ax, blocks)
    ax.plot(
        convergence_df["timestamp_s"],
        convergence_df["gaze_convergence_score"],
        linewidth=0.6,
        color="purple",
        alpha=0.7,
    )
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Gaze Convergence")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 1)
    return ax


def plot_dashboard(
    proximity_df: pd.DataFrame,
    xcorr_df: pd.DataFrame,
    gaze_cat_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    blocks: List[SessionBlock],
    title: str = "Parent-Child Synchrony Dashboard",
    sync_df: Optional[pd.DataFrame] = None,
    sync_chance: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Create the synchrony dashboard.

    Runs with ``pose_metrics_sync.csv`` show the whole-body synchrony windows
    (the reported measure) in one panel; older runs without it fall back to
    the earlier cross-correlation and lag panels.

    Args:
        proximity_df: Torso proximity data.
        xcorr_df: Cross-correlation data (used only without ``sync_df``).
        gaze_cat_df: Gaze category data.
        convergence_df: Gaze convergence data.
        blocks: Session blocks for background shading.
        title: Figure title.
        sync_df: Whole-body synchrony windows, if available.
        sync_chance: Chance level per block for ``sync_df``.

    Returns:
        Matplotlib Figure.
    """
    use_sync = sync_df is not None and not sync_df.empty
    n_rows = 4 if use_sync else 5
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3.6 * n_rows), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plot_torso_proximity(proximity_df, blocks, ax=axes[0])
    if use_sync:
        plot_movement_sync(sync_df, blocks, chance=sync_chance, ax=axes[1])
    else:
        plot_movement_xcorr(xcorr_df, blocks, axes=(axes[1], axes[2]))
    plot_gaze_categories(gaze_cat_df, blocks, ax=axes[n_rows - 2])

    last = axes[n_rows - 1]
    if not convergence_df.empty:
        plot_gaze_convergence(convergence_df, blocks, ax=last)
    else:
        last.text(
            0.5,
            0.5,
            "No gaze convergence data",
            ha="center",
            va="center",
            transform=last.transAxes,
        )
        last.set_title("Gaze Convergence")

    # Add block legend to bottom
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=b.color, alpha=0.3, label=b.name.replace("_", " "))
        for b in blocks
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(blocks),
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for gaze-plot."""
    parser = argparse.ArgumentParser(
        prog="gaze-plot",
        description="Plot parent-child synchrony metrics.",
    )
    parser.add_argument("--camera-dir", required=True, type=str)
    parser.add_argument("--session-config", required=True, type=str)
    parser.add_argument("--camera-id", default="camera_a", type=str)
    parser.add_argument("--metrics-csv", default="synchrony_metrics.csv", type=str)
    parser.add_argument(
        "--xcorr-csv",
        default="",
        type=str,
        help="Cross-correlation CSV (auto-detected if empty).",
    )
    parser.add_argument("--output", default="synchrony_dashboard.png", type=str)
    add_time_range_args(parser)
    return parser


def main() -> None:
    """CLI entry point for gaze-plot."""
    parser = build_arg_parser()
    args = parser.parse_args()

    camera_dir = Path(args.camera_dir)
    session_config = load_session_config(args.session_config)

    metrics_df = pd.read_csv(camera_dir / args.metrics_csv)
    blocks = session_config.blocks_for_camera(args.camera_id)

    # Apply time-range filter
    t_start, t_end = resolve_time_range(
        session_config, args.start_s, args.end_s, args.time_block
    )
    metrics_df = filter_by_time_range(metrics_df, t_start, t_end)

    # Build individual metric DataFrames from the combined CSV
    proximity_df = metrics_df[["frame_idx", "timestamp_s", "torso_distance_px"]].dropna(
        subset=["torso_distance_px"]
    )

    # Load xcorr if available
    xcorr_csv = camera_dir / (args.xcorr_csv or "movement_xcorr.csv")
    if xcorr_csv.exists():
        xcorr_df = pd.read_csv(xcorr_csv)
        xcorr_df = filter_by_time_range(
            xcorr_df, t_start, t_end, time_column="window_start_s"
        )
    else:
        xcorr_df = pd.DataFrame(
            columns=["window_start_s", "window_end_s", "peak_xcorr", "peak_lag_s"]
        )

    # Whole-body synchrony from the pose-metrics step, when it has run.
    sync_df = None
    sync_chance: Dict[str, float] = {}
    sync_csv = camera_dir / "pose_metrics_sync.csv"
    if sync_csv.exists():
        sync_df = filter_by_time_range(
            pd.read_csv(sync_csv), t_start, t_end, time_column="window_start_s"
        )
        summary = camera_dir / "pose_metrics_summary.json"
        if summary.exists():
            sync_chance = {
                s["block"]: s["sync_chance_mean"]
                for s in json.loads(summary.read_text(encoding="utf-8"))
                if s.get("sync_chance_mean") is not None
            }

    gaze_cat_df = metrics_df[["frame_idx", "timestamp_s", "gaze_category"]].dropna(
        subset=["gaze_category"]
    )

    if "gaze_convergence_score" in metrics_df.columns:
        convergence_df = metrics_df[
            ["frame_idx", "timestamp_s", "gaze_convergence_score"]
        ].dropna(subset=["gaze_convergence_score"])
    else:
        convergence_df = pd.DataFrame(
            columns=["frame_idx", "timestamp_s", "gaze_convergence_score"]
        )

    fig = plot_dashboard(
        proximity_df,
        xcorr_df,
        gaze_cat_df,
        convergence_df,
        blocks,
        title=f"Synchrony: {session_config.session_id} / {args.camera_id}",
        sync_df=sync_df,
        sync_chance=sync_chance,
    )

    output_path = camera_dir / args.output
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved to {output_path}")


if __name__ == "__main__":
    main()
