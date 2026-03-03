"""Per-block gaze metrics: gaze categories and convergence.

Computes gaze category proportions and gaze convergence scores
for parent-child dyads, then generates per-block comparison plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaze_analysis.config import SessionBlock, load_session_config
from gaze_analysis.head_bbox import extract_head_bboxes
from gaze_analysis.synchrony import compute_gaze_categories, compute_gaze_convergence

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

# Colorblind-safe block colors (match pose_metrics)
BLOCK_COLORS = {
    "grocery": "#2196F3",
    "synchrony": "#4CAF50",
    "storybook": "#FF9800",
}

# Gaze category colors
CATEGORY_COLORS = {
    "mutual_gaze": "#2ecc71",
    "joint_attention": "#3498db",
    "parent_watching": "#e67e22",
    "child_watching": "#9b59b6",
    "independent": "#95a5a6",
}

CATEGORY_ORDER = [
    "mutual_gaze",
    "joint_attention",
    "parent_watching",
    "child_watching",
    "independent",
]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_per_block_gaze_metrics(
    gaze_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    heatmaps: np.ndarray,
    hm_keys: List[str],
    blocks: List[SessionBlock],
    parent_id: int = 0,
    child_id: int = 1,
    image_width: int = 854,
    image_height: int = 480,
) -> Dict[str, Any]:
    """Compute gaze categories and convergence per block.

    Args:
        gaze_df: Gaze CSV DataFrame (gaze_heatmap.csv schema).
        pose_df: Pose DataFrame for head bbox extraction.
        tracks_df: Tracks DataFrame for head bbox fallback.
        heatmaps: Raw heatmap array [N, 64, 64] from npz.
        hm_keys: Heatmap keys matching heatmap indices.
        blocks: Session block definitions.
        parent_id: Parent track ID.
        child_id: Child track ID.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.

    Returns:
        Dict with keys: categories_df, convergence_df, block_stats.
    """
    all_categories: List[pd.DataFrame] = []
    all_convergence: List[pd.DataFrame] = []
    block_stats: List[Dict[str, Any]] = []

    # Extract head bboxes once for all frames
    head_bboxes_df = extract_head_bboxes(
        pose_df, tracks_df, image_width, image_height
    )

    for block in blocks:
        # Filter gaze data to block
        block_gaze = gaze_df[
            (gaze_df["timestamp_s"] >= block.start_s)
            & (gaze_df["timestamp_s"] <= block.end_s)
        ].copy()

        if block_gaze.empty:
            continue

        # Filter head bboxes to block frames
        block_frames = set(block_gaze["frame_idx"].unique())
        block_heads = head_bboxes_df[
            head_bboxes_df["frame_idx"].isin(block_frames)
        ].copy()

        # Build head center lists aligned with merged gaze frames
        parent_head_centers, child_head_centers = _build_head_centers(
            block_gaze, block_heads, parent_id, child_id
        )

        # Gaze categories
        cat_df = compute_gaze_categories(
            block_gaze,
            parent_id,
            child_id,
            parent_head_centers=parent_head_centers,
            child_head_centers=child_head_centers,
        )
        if not cat_df.empty:
            cat_df["block"] = block.name
            all_categories.append(cat_df)

        # Gaze convergence (from heatmaps)
        if heatmaps.size > 0:
            # Filter heatmap keys to block frames
            block_hm_indices = []
            block_hm_keys = []
            for i, key in enumerate(hm_keys):
                frame_idx = int(key.split("_t")[0][1:])
                if frame_idx in block_frames:
                    block_hm_indices.append(i)
                    block_hm_keys.append(key)

            if block_hm_indices:
                block_heatmaps = heatmaps[block_hm_indices]
                conv_df = compute_gaze_convergence(
                    block_heatmaps, block_hm_keys, parent_id, child_id
                )
                if not conv_df.empty:
                    conv_df["block"] = block.name
                    # Add timestamp_s from gaze_df
                    ts_lookup = dict(
                        zip(
                            block_gaze["frame_idx"].values,
                            block_gaze["timestamp_s"].values,
                        )
                    )
                    conv_df["timestamp_s"] = conv_df["frame_idx"].map(ts_lookup)
                    all_convergence.append(conv_df)

        # Per-block summary
        stats: Dict[str, Any] = {"block": block.name}
        if not cat_df.empty:
            counts = cat_df["gaze_category"].value_counts()
            total = len(cat_df)
            stats["n_frames"] = total
            for cat in CATEGORY_ORDER:
                stats[f"{cat}_pct"] = float(counts.get(cat, 0) / total * 100)
        if all_convergence and all_convergence[-1]["block"].iloc[0] == block.name:
            conv_vals = all_convergence[-1]["gaze_convergence_score"]
            stats["convergence_mean"] = float(conv_vals.mean())
            stats["convergence_std"] = float(conv_vals.std())
            stats["convergence_n_frames"] = len(conv_vals)
        block_stats.append(stats)

    categories_df = (
        pd.concat(all_categories, ignore_index=True) if all_categories else pd.DataFrame()
    )
    convergence_df = (
        pd.concat(all_convergence, ignore_index=True)
        if all_convergence
        else pd.DataFrame()
    )

    return {
        "categories_df": categories_df,
        "convergence_df": convergence_df,
        "block_stats": block_stats,
    }


def _build_head_centers(
    gaze_df: pd.DataFrame,
    head_bboxes_df: pd.DataFrame,
    parent_id: int,
    child_id: int,
) -> Tuple[Optional[List[Tuple[float, float]]], Optional[List[Tuple[float, float]]]]:
    """Build aligned head center lists for gaze category computation.

    Args:
        gaze_df: Gaze DataFrame with frame_idx, track_id.
        head_bboxes_df: Head bbox DataFrame.
        parent_id: Parent track ID.
        child_id: Child track ID.

    Returns:
        (parent_head_centers, child_head_centers) — lists of (x, y) or None.
    """
    if head_bboxes_df.empty:
        return None, None

    # Get frames where both parent and child have gaze data
    parent_frames = set(
        gaze_df[gaze_df["track_id"] == parent_id]["frame_idx"].values
    )
    child_frames = set(
        gaze_df[gaze_df["track_id"] == child_id]["frame_idx"].values
    )
    common_frames = sorted(parent_frames & child_frames)

    if not common_frames:
        return None, None

    # Build head center lookup per track
    p_heads = head_bboxes_df[head_bboxes_df["track_id"] == parent_id].set_index(
        "frame_idx"
    )
    c_heads = head_bboxes_df[head_bboxes_df["track_id"] == child_id].set_index(
        "frame_idx"
    )

    parent_centers = []
    child_centers = []
    for fidx in common_frames:
        if fidx in p_heads.index:
            pr = p_heads.loc[fidx]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[0]
            parent_centers.append(
                ((pr["head_x1"] + pr["head_x2"]) / 2, (pr["head_y1"] + pr["head_y2"]) / 2)
            )
        else:
            parent_centers.append((0.0, 0.0))

        if fidx in c_heads.index:
            cr = c_heads.loc[fidx]
            if isinstance(cr, pd.DataFrame):
                cr = cr.iloc[0]
            child_centers.append(
                ((cr["head_x1"] + cr["head_x2"]) / 2, (cr["head_y1"] + cr["head_y2"]) / 2)
            )
        else:
            child_centers.append((0.0, 0.0))

    return parent_centers, child_centers


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_gaze_categories_per_block(
    categories_df: pd.DataFrame,
    blocks: List[SessionBlock],
    output_path: Path,
) -> None:
    """Figure 3: Gaze categories — stacked bar chart per block.

    Args:
        categories_df: Per-frame gaze categories with 'block' column.
        blocks: Session block definitions.
        output_path: Base path for output (without extension).
    """
    block_names = [b.name for b in blocks if b.name in categories_df["block"].unique()]
    if not block_names:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    x = np.arange(len(block_names))

    # Compute proportions per block
    bottom = np.zeros(len(block_names))
    for cat in CATEGORY_ORDER:
        proportions = []
        for name in block_names:
            block_data = categories_df[categories_df["block"] == name]
            total = len(block_data)
            count = len(block_data[block_data["gaze_category"] == cat])
            proportions.append(count / total if total > 0 else 0)
        proportions = np.array(proportions)

        ax.bar(
            x,
            proportions,
            bottom=bottom,
            color=CATEGORY_COLORS[cat],
            alpha=0.85,
            label=cat.replace("_", " ").title(),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += proportions

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in block_names])
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.legend(
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )

    fig.suptitle(
        "Gaze Category Proportions by Interaction Context",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(str(output_path) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_path) + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[gaze-metrics] saved {output_path}.png/.svg")


def plot_gaze_convergence_per_block(
    convergence_df: pd.DataFrame,
    blocks: List[SessionBlock],
    output_path: Path,
) -> None:
    """Figure 4: Gaze convergence — boxplot + per-block time series.

    Args:
        convergence_df: Per-frame convergence with 'block' column.
        blocks: Session block definitions.
        output_path: Base path for output (without extension).
    """
    block_names = [b.name for b in blocks if b.name in convergence_df["block"].unique()]
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
        convergence_df[convergence_df["block"] == name][
            "gaze_convergence_score"
        ].values
        for name in block_names
    ]
    colors = [BLOCK_COLORS.get(name, "#999") for name in block_names]
    bp = ax_box.boxplot(box_data, tick_labels=block_names, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_box.set_ylabel("Gaze convergence\n(cosine similarity)")
    ax_box.set_title("Per-block comparison")
    ax_box.set_ylim(0, 1)

    # Right panels: per-block time series
    for i, name in enumerate(block_names):
        ax = axes[i + 1]
        block_data = convergence_df[convergence_df["block"] == name].sort_values(
            "timestamp_s"
        )
        color = BLOCK_COLORS.get(name, "#999")

        block_start = blocks[[b.name for b in blocks].index(name)].start_s
        t_rel = block_data["timestamp_s"] - block_start

        ax.plot(
            t_rel,
            block_data["gaze_convergence_score"],
            color=color,
            alpha=0.3,
            linewidth=0.5,
        )
        # Rolling mean
        window = min(30, len(block_data) // 4) if len(block_data) > 10 else 1
        if window > 1:
            rolling = (
                block_data["gaze_convergence_score"]
                .rolling(window, center=True)
                .mean()
            )
            ax.plot(t_rel, rolling, color=color, linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time in block (s)")
        ax.set_title(name.capitalize())
        if i > 0:
            ax.set_yticklabels([])

    fig.suptitle(
        "Parent-Child Gaze Convergence",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(str(output_path) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_path) + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[gaze-metrics] saved {output_path}.png/.svg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="video-gaze-metrics",
        description="Compute per-block gaze categories and convergence.",
    )
    parser.add_argument("--camera-dir", required=True, type=str)
    parser.add_argument("--session-config", required=True, type=str)
    parser.add_argument("--camera-id", default="camera_a", type=str)
    parser.add_argument("--gaze-csv", default="gaze_heatmap.csv", type=str)
    parser.add_argument("--gaze-npz", default="gaze_heatmaps.npz", type=str)
    parser.add_argument("--pose-input", default="pose_3d_filtered_5hz.csv", type=str)
    parser.add_argument("--tracks-input", default="tracks_2d_filtered_5hz.csv", type=str)
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    camera_dir = Path(args.camera_dir)
    config = load_session_config(args.session_config)
    mapping = config.get_camera_mapping(args.camera_id)
    blocks = config.session_blocks

    # Load data
    gaze_df = pd.read_csv(camera_dir / args.gaze_csv)
    pose_df = pd.read_csv(camera_dir / args.pose_input)
    tracks_df = pd.read_csv(camera_dir / args.tracks_input)
    print(f"[gaze-metrics] loaded {len(gaze_df)} gaze rows")

    # Load heatmaps
    npz_path = camera_dir / args.gaze_npz
    if npz_path.exists():
        npz_data = np.load(npz_path)
        heatmaps = npz_data["heatmaps"]
        hm_keys = list(npz_data["keys"])
        print(f"[gaze-metrics] loaded {len(heatmaps)} heatmaps")
    else:
        heatmaps = np.array([])
        hm_keys = []

    results = compute_per_block_gaze_metrics(
        gaze_df,
        pose_df,
        tracks_df,
        heatmaps,
        hm_keys,
        blocks,
        parent_id=mapping.parent_track_id,
        child_id=mapping.child_track_id,
        image_width=config.image_width,
        image_height=config.image_height,
    )

    # Save CSVs
    if not results["categories_df"].empty:
        cat_path = camera_dir / "gaze_metrics_categories.csv"
        results["categories_df"].to_csv(cat_path, index=False)
        print(f"[gaze-metrics] wrote {cat_path} ({len(results['categories_df'])} rows)")

    if not results["convergence_df"].empty:
        conv_path = camera_dir / "gaze_metrics_convergence.csv"
        results["convergence_df"].to_csv(conv_path, index=False)
        print(
            f"[gaze-metrics] wrote {conv_path} ({len(results['convergence_df'])} rows)"
        )

    # Save summary JSON
    summary_path = camera_dir / "gaze_metrics_summary.json"
    summary_path.write_text(
        json.dumps(results["block_stats"], indent=2), encoding="utf-8"
    )
    print(f"[gaze-metrics] wrote {summary_path}")

    # Print summary
    for stats in results["block_stats"]:
        block = stats["block"]
        n = stats.get("n_frames", 0)
        mg = stats.get("mutual_gaze_pct", 0)
        ja = stats.get("joint_attention_pct", 0)
        conv = stats.get("convergence_mean", 0)
        print(
            f"  {block:12s}: n={n}, mutual_gaze={mg:.1f}%, "
            f"joint_attn={ja:.1f}%, convergence={conv:.3f}"
        )

    # Generate plots
    if not results["categories_df"].empty:
        plot_gaze_categories_per_block(
            results["categories_df"], blocks, camera_dir / "gaze_categories"
        )

    if not results["convergence_df"].empty:
        plot_gaze_convergence_per_block(
            results["convergence_df"], blocks, camera_dir / "gaze_convergence"
        )


if __name__ == "__main__":
    main()
