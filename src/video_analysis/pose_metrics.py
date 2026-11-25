"""Per-block pose metrics: torso proximity and movement synchrony.

Computes torso-centroid distance and movement cross-correlation for
parent-child dyads, then generates per-block comparison plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaze_analysis.config import SessionBlock, load_session_config
from gaze_analysis.synchrony import (
    XCORR_COLUMNS,
    chance_summary,
    compute_movement_sync,
    compute_movement_xcorr,
    compute_movement_xcorr_chance,
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


# Blocks where both people are tracked in fewer than this share of frames are
# flagged: their metrics describe only part of the block.
LOW_COVERAGE_THRESHOLD = 0.5

# Thresholds behind ``quality_verdict``; "usable" is the recommended inclusion
# rule for a task (pieces joined). Basis (2026-11-16, 1011v3-1020v3): across
# dyads, distance from a 2 min sample correlates r = 0.85 with the full-task
# value, and synchrony from 10 windows r ~ 0.87 (3 windows only r = 0.66).
# Need at least this much time with both people tracked.
MIN_VALID_SECONDS = 120.0
# Movement synchrony needs at least this many 20 s windows (about 2 minutes of
# unbroken tracking); with fewer it is not reported.
MIN_SYNC_WINDOWS = 10
# Torso ruler needs at least this many confident frames per person (10 s).
MIN_RULER_FRAMES = 50
# Bump when metric definitions change; stored with every block's metrics so
# tables never silently mix definitions. 2026-11-16: 5 fps even frame timing,
# short-gap-only filling, whole-body synchrony, task table, verdict.
# 2026-11-25: torso-normalized distance, one chance-shift rule for all block
# lengths (>= 3 shifts, at the real windows), flat windows left out, verdict
# fails closed.
METRICS_VERSION = "2026-11-25"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def block_coverage(
    block_pose: pd.DataFrame,
    block: SessionBlock,
    frame_rate: float,
    parent_id: int = 0,
    child_id: int = 1,
) -> Dict[str, Any]:
    """Count how many frames of a block have the parent, the child, and both.

    Args:
        block_pose: Pose rows inside the block.
        block: The block.
        frame_rate: Analysis frame rate (frames per second).
        parent_id: Parent track ID.
        child_id: Child track ID.

    Returns:
        Coverage counts, the share of block frames with both people, and a
        ``low_coverage`` flag.
    """
    block_frames = int(round((block.end_s - block.start_s) * frame_rate)) + 1
    times = block_pose.groupby("track_id")["timestamp_s"].apply(
        lambda t: set(np.round(t.to_numpy(float) * frame_rate).astype(int))
    )
    parent = times.get(parent_id, set())
    child = times.get(child_id, set())
    both_frames = np.array(sorted(parent & child))
    both = len(both_frames)
    coverage = min(1.0, both / block_frames) if block_frames > 0 else 0.0
    if both:
        run_lengths = np.diff(
            np.flatnonzero(np.diff(np.r_[-10, both_frames, both_frames[-1] + 10]) > 1)
        )
        n_segments = int(len(run_lengths))
        longest = float(run_lengths.max()) / frame_rate
    else:
        n_segments, longest = 0, 0.0
    return {
        "block_start_s": block.start_s,
        "block_end_s": block.end_s,
        "block_frames": block_frames,
        "frames_with_parent": len(parent),
        "frames_with_child": len(child),
        "frames_with_both": both,
        "coverage_both": round(coverage, 4),
        "low_coverage": bool(coverage < LOW_COVERAGE_THRESHOLD),
        "valid_seconds_both": round(both / frame_rate, 2),
        "n_segments": n_segments,
        "longest_segment_s": round(longest, 2),
    }


def _finite_or_none(value: Any) -> Optional[float]:
    """``float(value)`` when it is a finite number, else None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def quality_verdict(stats: Dict[str, Any]) -> tuple[str, str]:
    """Data check for one block or task: is there enough data behind the numbers?

    ``insufficient`` when both people are tracked for less than
    ``MIN_VALID_SECONDS``; ``check`` when coverage is below
    ``LOW_COVERAGE_THRESHOLD`` or there are fewer than ``MIN_SYNC_WINDOWS``
    synchrony windows; otherwise ``usable``. ``usable`` is the recommended
    inclusion rule, together with a check of the overlay clips.

    Args:
        stats: Row with ``valid_seconds_both``, ``coverage_both`` and
            (optionally) ``sync_n_windows``.

    Returns:
        ``(verdict, reasons)``; reasons are plain sentences joined by "; ".
    """
    seconds = _finite_or_none(stats.get("valid_seconds_both"))
    coverage = _finite_or_none(stats.get("coverage_both"))
    windows = _finite_or_none(stats.get("sync_n_windows"))
    windows = 0 if windows is None else int(windows)
    reasons = []
    if seconds is None or coverage is None:
        # Fail closed: unknown tracking time is not enough tracking time.
        return "insufficient", "how long both people were tracked is unknown"
    if seconds < MIN_VALID_SECONDS:
        reasons.append(
            f"both people tracked for only {seconds:.0f} s "
            f"(less than {MIN_VALID_SECONDS:.0f} s)"
        )
        return "insufficient", "; ".join(reasons)
    if coverage < LOW_COVERAGE_THRESHOLD:
        reasons.append(f"both people tracked in only {coverage:.0%} of the time")
    if windows < MIN_SYNC_WINDOWS:
        reasons.append(
            "too few unbroken 20-second stretches to measure movement synchrony"
        )
    return ("check" if reasons else "usable"), "; ".join(reasons)


def body_scaled_proximity(
    prox: pd.DataFrame, ruler_frames: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """Torso-normalized distance: torso distance divided by a torso ruler.

    The ruler is the average of the parent's and the child's median torso
    length (medians, so brief bends and mis-detections do not move it).
    Callers pass all of a camera's frames as ``ruler_frames``, so one camera
    has one ruler: a per-task ruler would leak posture into task contrasts
    (sitting in storybook shortens the torso on screen), whereas a single
    ruler keeps the contrast between tasks the same as in pixels. It reduces
    the overall scale difference between camera views; it is not a physical
    distance and does not correct viewing angle, people moving nearer or
    farther, or lens distortion (a 2D camera cannot see distance toward or
    away from itself). Torso lengths come only from frames where all four
    torso joints are confident (see ``_torso_length``).

    Args:
        prox: Per-frame proximity rows to summarise, with torso_distance_px
            and, from runs after 2026-11-25, parent_torso_px and
            child_torso_px.
        ruler_frames: Frames to measure the ruler on (default: ``prox``).

    Returns:
        ``body_scale_px``, ``proximity_scaled_mean`` and
        ``proximity_scaled_median``, or an empty dict when torso lengths are
        not available.
    """
    ruler = prox if ruler_frames is None else ruler_frames
    if not {"parent_torso_px", "child_torso_px"} <= set(ruler.columns):
        return {}
    support = int(
        min(
            ruler["parent_torso_px"].notna().sum(),
            ruler["child_torso_px"].notna().sum(),
        )
    )
    if support < MIN_RULER_FRAMES:
        return {}
    parent = ruler["parent_torso_px"].median()
    child = ruler["child_torso_px"].median()
    if not (np.isfinite(parent) and np.isfinite(child)) or parent <= 0 or child <= 0:
        return {}
    scale = float((parent + child) / 2)
    dist = prox["torso_distance_px"]
    return {
        "body_scale_px": scale,
        "body_scale_n_frames": support,
        "proximity_scaled_mean": float(dist.mean()) / scale,
        "proximity_scaled_median": float(dist.median()) / scale,
    }


def compute_per_block_metrics(
    pose_df: pd.DataFrame,
    blocks: List[SessionBlock],
    parent_id: int = 0,
    child_id: int = 1,
    xcorr_window_s: float = 5.0,
    xcorr_max_lag_s: float = 2.0,
    frame_rate: float = 6.0,
) -> Dict[str, Any]:
    """Compute torso proximity and movement xcorr per block.

    Args:
        pose_df: Smoothed pose DataFrame.
        blocks: Session block definitions.
        parent_id: Parent track ID.
        child_id: Child track ID.
        xcorr_window_s: Cross-correlation window size in seconds.
        xcorr_max_lag_s: Maximum lag to search in seconds.
        frame_rate: Analysis frame rate, used for tracking coverage.

    Returns:
        Dict with keys: proximity_frames, xcorr_windows, block_stats.
    """
    all_proximity: List[pd.DataFrame] = []
    all_xcorr: List[pd.DataFrame] = []
    all_sync: List[pd.DataFrame] = []
    block_stats: List[Dict[str, Any]] = []

    for block in blocks:
        block_pose = pose_df[
            (pose_df["timestamp_s"] >= block.start_s)
            & (pose_df["timestamp_s"] <= block.end_s)
        ].copy()

        coverage = block_coverage(block_pose, block, frame_rate, parent_id, child_id)
        if block_pose.empty:
            empty_stats: Dict[str, Any] = {"block": block.name, **coverage}
            empty_stats["quality_verdict"], empty_stats["quality_reasons"] = (
                quality_verdict(empty_stats)
            )
            block_stats.append(empty_stats)
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
        stats: Dict[str, Any] = {"block": block.name, **coverage}
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
            chance = compute_movement_xcorr_chance(
                block_pose, parent_id, child_id, xcorr_window_s, xcorr_max_lag_s
            )
            if not chance.empty:
                stats["xcorr_chance_mean"] = float(chance["peak_xcorr"].mean())
                stats["xcorr_minus_chance"] = (
                    stats["xcorr_mean"] - stats["xcorr_chance_mean"]
                )
        # Whole-body movement synchrony (the reported measure).
        sync, sync_chance = compute_movement_sync(block_pose, parent_id, child_id)
        if not sync.empty:
            sync["block"] = block.name
            all_sync.append(sync)
            stats["sync_mean"] = float(sync["sync_r"].mean())
            stats["sync_std"] = float(sync["sync_r"].std())
            stats["sync_n_windows"] = len(sync)
            reference = chance_summary(sync_chance)
            if reference:
                stats.update(reference)
                stats["sync_minus_chance"] = (
                    stats["sync_mean"] - stats["sync_chance_mean"]
                )
        stats["quality_verdict"], stats["quality_reasons"] = quality_verdict(stats)
        stats["metrics_version"] = METRICS_VERSION
        block_stats.append(stats)

    proximity_df = (
        pd.concat(all_proximity, ignore_index=True) if all_proximity else pd.DataFrame()
    )
    # Body-size distance: one ruler from all of this camera's frames.
    for stats in block_stats:
        if "proximity_n_frames" in stats:
            frames = proximity_df[proximity_df["block"] == stats["block"]]
            stats.update(body_scaled_proximity(frames, ruler_frames=proximity_df))
    xcorr_df = pd.concat(all_xcorr, ignore_index=True) if all_xcorr else pd.DataFrame()

    sync_df = pd.concat(all_sync, ignore_index=True) if all_sync else pd.DataFrame()

    return {
        "proximity_df": proximity_df,
        "xcorr_df": xcorr_df,
        "sync_df": sync_df,
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

    fig, axes = plt.subplots(
        1,
        n_blocks + 1,
        figsize=(4 + 3.5 * n_blocks, 4),
        gridspec_kw={"width_ratios": [1.2] + [1] * n_blocks},
    )

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
        block_data = proximity_df[proximity_df["block"] == name].sort_values(
            "timestamp_s"
        )
        color = BLOCK_COLORS.get(name, "#999")

        # Plot relative time (seconds into block)
        block_start = blocks[[b.name for b in blocks].index(name)].start_s
        t_rel = block_data["timestamp_s"] - block_start

        ax.plot(
            t_rel,
            block_data["torso_distance_px"],
            color=color,
            alpha=0.3,
            linewidth=0.5,
        )
        # Rolling mean for readability
        window = min(90, len(block_data) // 4) if len(block_data) > 10 else 1
        if window > 1:
            rolling = (
                block_data["torso_distance_px"].rolling(window, center=True).mean()
            )
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
    sync_df: pd.DataFrame,
    block_stats: List[Dict[str, Any]],
    blocks: List[SessionBlock],
    output_path: Path,
) -> None:
    """Figure 2: whole-body movement synchrony per block, against chance.

    Left: each block's mean synchrony (bar) next to its chance level (black
    line), with every 20 s window as a dot (windows overlap by half, so no
    error bars are drawn). Blocks
    with fewer than ``MIN_SYNC_WINDOWS`` windows are drawn faded and labelled,
    matching the report. Right: every window over session time, so gaps and
    drifts are visible.

    Args:
        sync_df: Per-window synchrony (``pose_metrics_sync.csv``): block,
            window_start_s, window_end_s, sync_r.
        block_stats: Per-block summaries with sync_mean, sync_chance_mean and
            sync_n_windows.
        blocks: Session block definitions (sets the order).
        output_path: Base path for output (without extension).
    """
    if sync_df.empty:
        return
    present = set(sync_df["block"].unique())
    shown = [b for b in blocks if b.name in present]
    if not shown:
        return
    stats_by_block = {s["block"]: s for s in block_stats}
    rng = np.random.default_rng(0)

    fig, (ax_bar, ax_time) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    for i, block in enumerate(shown):
        color = BLOCK_COLORS.get(block.name, block.color or "#999")
        r = sync_df.loc[sync_df["block"] == block.name, "sync_r"].to_numpy()
        enough = len(r) >= MIN_SYNC_WINDOWS
        alpha = 0.75 if enough else 0.25
        mean = float(np.mean(r))
        # No error bars: windows overlap by half and are autocorrelated, so
        # SD / sqrt(n) would overstate the precision. The dots show the spread.
        ax_bar.bar(
            i,
            mean,
            color=color,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.5,
            width=0.6,
        )
        ax_bar.scatter(
            i + rng.uniform(-0.18, 0.18, len(r)),
            r,
            s=9,
            color="black",
            alpha=0.35 if enough else 0.15,
            zorder=3,
            linewidths=0,
        )
        chance = stats_by_block.get(block.name, {}).get("sync_chance_mean")
        if chance is not None and not pd.isna(chance):
            ax_bar.hlines(
                chance, i - 0.36, i + 0.36, color="black", linewidth=2.2, zorder=4
            )
        label = f"n={len(r)}" if enough else f"n={len(r)}\ntoo few"
        ax_bar.text(
            i,
            1.02,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            transform=ax_bar.get_xaxis_transform(),
        )
        windows = sync_df[sync_df["block"] == block.name]
        mid_min = (windows["window_start_s"] + windows["window_end_s"]) / 2 / 60
        ax_time.scatter(
            mid_min,
            windows["sync_r"],
            s=14,
            color=color,
            alpha=alpha,
            label=block.name.replace("_", " "),
            linewidths=0,
        )
    ax_bar.axhline(0, color="black", linewidth=0.5)
    ax_bar.set_xticks(range(len(shown)))
    ax_bar.set_xticklabels(
        [b.name.replace("_", " ").capitalize() for b in shown],
        rotation=20 if len(shown) > 3 else 0,
    )
    ax_bar.set_ylabel("Movement synchrony (r)")
    ax_bar.set_ylim(-1, 1)
    ax_bar.set_title(
        "Per block: bar = mean, line = chance,\ndots = 20 s windows (overlapping)",
        pad=26,
        fontsize=11,
    )
    ax_time.axhline(0, color="black", linewidth=0.5)
    ax_time.set_ylim(-1, 1)
    ax_time.set_xlabel("Time in video (min)")
    ax_time.set_ylabel("Synchrony per 20 s window (r)")
    ax_time.set_title("Every window over time", pad=26, fontsize=11)
    ax_time.legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
    for ax in (ax_bar, ax_time):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Parent-child movement synchrony (whole-body movement, 20 s windows)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(str(output_path) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_path) + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[pose-metrics] saved {output_path}.png/.svg")


PROXIMITY_TABLE_COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "torso_distance_px",
    "parent_torso_px",
    "child_torso_px",
    "block",
]
SYNC_TABLE_COLUMNS = ["segment_id", "window_start_s", "window_end_s", "sync_r", "block"]


def save_sync_outputs(
    camera_dir: Path,
    sync_df: pd.DataFrame,
    block_stats: List[Dict[str, Any]],
    blocks: List[SessionBlock],
) -> None:
    """Write ``pose_metrics_sync.csv`` and ``pose_synchrony.png/.svg``.

    The table is always written (header only when there are no windows), so
    a rerun never leaves an earlier run's windows behind and the dashboard
    can tell a current run from an old one by the file being there. With no
    windows, an earlier synchrony figure is removed rather than left stale.

    Args:
        camera_dir: Camera output folder.
        sync_df: Per-window synchrony (may be empty).
        block_stats: Per-block summaries.
        blocks: Session blocks.
    """
    camera_dir = Path(camera_dir)
    table = sync_df if not sync_df.empty else pd.DataFrame(columns=SYNC_TABLE_COLUMNS)
    sync_path = camera_dir / "pose_metrics_sync.csv"
    table.to_csv(sync_path, index=False)
    print(f"[pose-metrics] wrote {sync_path} ({len(table)} rows)")
    if sync_df.empty:
        for suffix in (".png", ".svg"):
            (camera_dir / f"pose_synchrony{suffix}").unlink(missing_ok=True)
        return
    plot_synchrony(sync_df, block_stats, blocks, camera_dir / "pose_synchrony")


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
    parser.add_argument(
        "--xcorr-window",
        default=5.0,
        type=float,
        help="Cross-correlation window size in seconds.",
    )
    parser.add_argument(
        "--xcorr-max-lag", default=2.0, type=float, help="Max lag to search in seconds."
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    camera_dir = Path(args.camera_dir)
    config = load_session_config(args.session_config)
    mapping = config.get_camera_mapping(args.camera_id)
    blocks = config.blocks_for_camera(args.camera_id)

    pose_df = pd.read_csv(camera_dir / args.pose_input)
    print(f"[pose-metrics] loaded {len(pose_df)} pose rows from {args.pose_input}")

    results = compute_per_block_metrics(
        pose_df,
        blocks,
        parent_id=mapping.parent_track_id,
        child_id=mapping.child_track_id,
        xcorr_window_s=args.xcorr_window,
        xcorr_max_lag_s=args.xcorr_max_lag,
        frame_rate=config.frame_rate,
    )

    # Save CSVs (always, header-only when empty, so reruns leave nothing stale)
    prox_path = camera_dir / "pose_metrics_proximity.csv"
    prox_out = results["proximity_df"]
    if prox_out.empty:
        prox_out = pd.DataFrame(columns=PROXIMITY_TABLE_COLUMNS)
    prox_out.to_csv(prox_path, index=False)
    print(f"[pose-metrics] wrote {prox_path} ({len(prox_out)} rows)")

    xcorr_path = camera_dir / "pose_metrics_xcorr.csv"
    xcorr_out = results["xcorr_df"]
    if xcorr_out.empty:
        xcorr_out = pd.DataFrame(columns=[*XCORR_COLUMNS, "block"])
    xcorr_out.to_csv(xcorr_path, index=False)

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
        sync_mean = stats.get("sync_mean", float("nan"))
        sync_chance = stats.get("sync_chance_mean", float("nan"))
        print(
            f"  {block:12s}: proximity={prox_mean:.1f} +/- {prox_std:.1f} px, "
            f"movement synchrony={sync_mean:.2f} (chance {sync_chance:.2f}), "
            f"both people tracked in {stats['coverage_both']:.0%} of the block"
            + (" (LOW: interpret with care)" if stats["low_coverage"] else "")
        )

    # Generate plots
    if not results["proximity_df"].empty:
        plot_proximity(results["proximity_df"], blocks, camera_dir / "pose_proximity")
    else:
        for suffix in (".png", ".svg"):
            (camera_dir / f"pose_proximity{suffix}").unlink(missing_ok=True)

    save_sync_outputs(camera_dir, results["sync_df"], results["block_stats"], blocks)


if __name__ == "__main__":
    main()
