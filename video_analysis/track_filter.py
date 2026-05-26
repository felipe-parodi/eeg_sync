"""Block-based track filtering for parent-child analysis.

Given task-block time ranges, keeps only the top-N most persistent tracks
per block and assigns consistent IDs (parent=0, child=1) based on average
bounding-box area.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class BlockDef:
    """A named time block within the video."""

    name: str
    start_time: str  # "MM:SS" or "MM:SS.ff"
    end_time: str


@dataclass
class TrackFilterConfig:
    """Configuration for block-based track filtering."""

    session_dir: str
    camera: str = "camera_a"
    blocks: List[BlockDef] = field(default_factory=list)
    source_fps: float = 30.0
    n_keep: int = 2
    # Area-continuity re-link: detect frames where the tracker appears to have
    # swapped the parent/child ID and correct them in place before role
    # assignment. Set anchor_seconds <= 0 to disable.
    relink_anchor_seconds: float = 4.0
    relink_min_run_seconds: float = 0.5
    relink_min_size_ratio: float = 1.3


def _parse_time(time_str: str) -> float:
    """Parse ``MM:SS`` or ``MM:SS.ff`` or ``HH:MM:SS`` to seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Cannot parse time: {time_str!r}")


def _time_to_frame(time_str: str, fps: float) -> int:
    """Convert time string to frame index."""
    return int(_parse_time(time_str) * fps)


def _bbox_area(row: pd.Series) -> float:
    """Compute bbox area from a tracks row."""
    w = max(0.0, float(row["bbox_x2"]) - float(row["bbox_x1"]))
    h = max(0.0, float(row["bbox_y2"]) - float(row["bbox_y1"]))
    return w * h


def _detect_track_swaps(
    block: pd.DataFrame,
    track_a: int,
    track_b: int,
    anchor_seconds: float,
    min_run_seconds: float,
    min_size_ratio: float,
) -> List[Tuple[float, float]]:
    """Detect timestamp ranges where two tracks look like they have swapped.

    Strategy: anchor each track's "true" bbox area from the first
    ``anchor_seconds`` of frames where it appears. For every subsequent
    frame in which both tracks are present, the assignment is "swapped" if
    re-pairing the two area observations with the opposite anchor produces
    a lower total absolute-area error than the original pairing. Single-
    frame flips are ignored; only runs of at least ``min_run_seconds``
    are kept.

    Args:
        block: Track rows for one analysis block (raw, not yet role-remapped).
        track_a: Original tracker id for the apparently larger person.
        track_b: Original tracker id for the apparently smaller person.
        anchor_seconds: Window at the start of each track used to fix its
            anchor bbox area.
        min_run_seconds: Minimum sustained swap duration to act on.
        min_size_ratio: If the two anchors are within this ratio of each
            other (e.g. 1.3 means within 30%), the size signal is too weak
            to discriminate — skip detection entirely.

    Returns:
        List of ``(start_s, end_s)`` timestamp ranges (inclusive on both
        ends) over which ``track_a`` and ``track_b`` should be swapped.
    """
    sub = block[block["track_id"].isin([track_a, track_b])].copy()
    if sub.empty:
        return []
    sub["_area"] = sub.apply(_bbox_area, axis=1)

    def _anchor(tid: int) -> float:
        rows = sub[sub["track_id"] == tid].sort_values("timestamp_s")
        if rows.empty:
            return float("nan")
        t0 = rows["timestamp_s"].iloc[0]
        window = rows[rows["timestamp_s"] <= t0 + anchor_seconds]
        return float(window["_area"].mean())

    anchor_a = _anchor(track_a)
    anchor_b = _anchor(track_b)
    if pd.isna(anchor_a) or pd.isna(anchor_b):
        return []
    larger = max(anchor_a, anchor_b)
    smaller = min(anchor_a, anchor_b)
    if smaller <= 0 or larger / smaller < min_size_ratio:
        return []

    paired = sub.pivot_table(
        index="timestamp_s",
        columns="track_id",
        values="_area",
        aggfunc="first",
    ).dropna(subset=[track_a, track_b])
    if paired.empty:
        return []

    orig_cost = (paired[track_a] - anchor_a).abs() + (paired[track_b] - anchor_b).abs()
    swap_cost = (paired[track_a] - anchor_b).abs() + (paired[track_b] - anchor_a).abs()
    is_swapped = swap_cost < orig_cost

    runs: List[Tuple[float, float]] = []
    run_start: float | None = None
    prev_ts: float | None = None
    timestamps = paired.index.tolist()
    for ts in timestamps:
        if bool(is_swapped.loc[ts]):
            if run_start is None:
                run_start = float(ts)
            prev_ts = float(ts)
        else:
            if run_start is not None and prev_ts is not None:
                if prev_ts - run_start >= min_run_seconds:
                    runs.append((run_start, prev_ts))
            run_start = None
            prev_ts = None
    if run_start is not None and prev_ts is not None:
        if prev_ts - run_start >= min_run_seconds:
            runs.append((run_start, prev_ts))

    return runs


def _apply_track_swaps(
    df: pd.DataFrame,
    track_a: int,
    track_b: int,
    swap_ranges: List[Tuple[float, float]],
) -> None:
    """In-place: swap ``track_a`` and ``track_b`` ids over the given ranges."""
    if not swap_ranges:
        return
    for start_s, end_s in swap_ranges:
        in_range = (df["timestamp_s"] >= start_s) & (df["timestamp_s"] <= end_s)
        flip_a = in_range & (df["track_id"] == track_a)
        flip_b = in_range & (df["track_id"] == track_b)
        df.loc[flip_a, "track_id"] = track_b
        df.loc[flip_b, "track_id"] = track_a


def _rank_tracks_by_persistence(
    block: pd.DataFrame,
    n_keep: int,
) -> List[Tuple[int, str, float]]:
    """Rank the most persistent tracks in one pre-filtered block."""
    if block.empty:
        return []

    counts = block.groupby("track_id")["frame_idx"].nunique()
    top_ids = counts.nlargest(n_keep).index.tolist()
    if not top_ids:
        return []

    sub = block[block["track_id"].isin(top_ids)].copy()
    sub["_area"] = sub.apply(_bbox_area, axis=1)
    avg_areas = sub.groupby("track_id")["_area"].mean()
    sorted_ids = avg_areas.sort_values(ascending=False).index.tolist()
    result = []
    for role_idx, tid in enumerate(sorted_ids):
        role = "parent" if role_idx == 0 else "child"
        result.append((int(tid), role, float(avg_areas[tid])))
    return result


def identify_top_tracks(
    tracks_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    n_keep: int = 2,
) -> List[Tuple[int, str, float]]:
    """Identify the top-N most persistent tracks in a frame range.

    Args:
        tracks_df: Full tracks DataFrame.
        start_frame: First frame of the block.
        end_frame: Last frame of the block (inclusive).
        n_keep: Number of tracks to keep.

    Returns:
        List of (track_id, role, avg_area) tuples sorted by role
        (parent=0 first, child=1 second).  Role is assigned by
        descending average bbox area.
    """
    block = tracks_df[
        (tracks_df["frame_idx"] >= start_frame) & (tracks_df["frame_idx"] <= end_frame)
    ]
    return _rank_tracks_by_persistence(block, n_keep)


def identify_top_tracks_by_time(
    tracks_df: pd.DataFrame,
    start_s: float,
    end_s: float,
    n_keep: int = 2,
) -> List[Tuple[int, str, float]]:
    """Identify the top-N most persistent tracks in a timestamp range.

    Args:
        tracks_df: Full tracks DataFrame.
        start_s: First source-video second of the block.
        end_s: End of the source-video block, exclusive.
        n_keep: Number of tracks to keep.

    Returns:
        List of (track_id, role, avg_area) tuples sorted by role
        (parent=0 first, child=1 second).
    """
    block = tracks_df[
        (tracks_df["timestamp_s"] >= start_s) & (tracks_df["timestamp_s"] < end_s)
    ]
    return _rank_tracks_by_persistence(block, n_keep)


def filter_tracks(config: TrackFilterConfig) -> Dict[str, Any]:
    """Run block-based track filtering.

    Args:
        config: Filter configuration.

    Returns:
        Summary dict with per-block mappings and output paths.
    """
    session_dir = Path(config.session_dir)
    camera_dir = session_dir / config.camera
    tracks_path = camera_dir / "tracks_2d.csv"
    pose_path = camera_dir / "pose_3d.csv"

    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_path}")

    tracks_df = pd.read_csv(tracks_path)
    pose_df = pd.read_csv(pose_path) if pose_path.exists() else None
    blocks = config.blocks or []

    # Build block time ranges and track mappings.
    block_summaries: List[Dict[str, Any]] = []
    # Collect (start_s, end_s, {old_id: new_id}) per block.
    block_remaps: List[Tuple[float, float, Dict[int, int]]] = []

    relink_enabled = config.relink_anchor_seconds > 0

    for blk in blocks:
        start_s = _parse_time(blk.start_time)
        end_s = _parse_time(blk.end_time)

        # Re-link ID swaps within this block using bbox-area continuity. We
        # operate on the top-2 tracks (parent/child); larger n_keep falls back
        # to the original identification logic without re-linking.
        swap_ranges: List[Tuple[float, float]] = []
        if relink_enabled and config.n_keep >= 2:
            pre_top = identify_top_tracks_by_time(tracks_df, start_s, end_s, n_keep=2)
            if len(pre_top) == 2:
                tid_large = pre_top[0][0]
                tid_small = pre_top[1][0]
                block_view = tracks_df[
                    (tracks_df["timestamp_s"] >= start_s)
                    & (tracks_df["timestamp_s"] < end_s)
                ]
                swap_ranges = _detect_track_swaps(
                    block_view,
                    track_a=tid_large,
                    track_b=tid_small,
                    anchor_seconds=config.relink_anchor_seconds,
                    min_run_seconds=config.relink_min_run_seconds,
                    min_size_ratio=config.relink_min_size_ratio,
                )
                if swap_ranges:
                    _apply_track_swaps(tracks_df, tid_large, tid_small, swap_ranges)
                    if pose_df is not None:
                        _apply_track_swaps(pose_df, tid_large, tid_small, swap_ranges)
                    total = sum(end - start for start, end in swap_ranges)
                    print(
                        f"[filter] {blk.name}: re-linked tracks {tid_large}<->{tid_small} "
                        f"across {len(swap_ranges)} swap(s), {total:.2f}s total"
                    )

        top = identify_top_tracks_by_time(tracks_df, start_s, end_s, config.n_keep)

        remap: Dict[int, int] = {}
        summary_entry: Dict[str, Any] = {
            "name": blk.name,
            "start_s": start_s,
            "end_s": end_s,
            "tracks": [],
            "relink_swaps": [
                {"start_s": float(s), "end_s": float(e)} for s, e in swap_ranges
            ],
        }

        for role_idx, (tid, role, avg_area) in enumerate(top):
            remap[tid] = role_idx  # parent=0, child=1
            summary_entry["tracks"].append(
                {
                    "original_id": tid,
                    "new_id": role_idx,
                    "role": role,
                    "avg_bbox_area": round(avg_area, 1),
                }
            )
            print(
                f"[filter] {blk.name}: track {tid} -> {role_idx} "
                f"({role}, avg area {avg_area:.0f})"
            )

        block_remaps.append((start_s, end_s, remap))
        block_summaries.append(summary_entry)

    # Apply filtering: keep only remapped tracks within blocks,
    # drop everything else (gaps between blocks).
    filtered_rows = []
    for start_s, end_s, remap in block_remaps:
        block_df = tracks_df[
            (tracks_df["timestamp_s"] >= start_s) & (tracks_df["timestamp_s"] < end_s)
        ].copy()
        # Keep only tracks in remap.
        keep_ids = set(remap.keys())
        block_df = block_df[block_df["track_id"].isin(keep_ids)].copy()
        # Remap IDs.
        block_df["track_id"] = block_df["track_id"].map(remap)
        block_df["track_label"] = block_df["track_id"].map({0: "parent", 1: "child"})
        filtered_rows.append(block_df)

    if filtered_rows:
        result_df = pd.concat(filtered_rows, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=tracks_df.columns)

    # Write filtered tracks.
    out_tracks = camera_dir / "tracks_2d_filtered.csv"
    result_df.to_csv(out_tracks, index=False)
    print(f"[filter] wrote {out_tracks} ({len(result_df)} rows)")

    summary: Dict[str, Any] = {
        "blocks": block_summaries,
        "tracks_output": str(out_tracks),
        "total_rows": len(result_df),
        "unique_frames": (
            int(result_df["frame_idx"].nunique()) if not result_df.empty else 0
        ),
    }

    # Also filter pose CSV if it exists. Uses the in-memory copy that already
    # has the same re-link swaps applied as tracks_df.
    if pose_df is not None:
        filtered_pose_rows = []
        for start_s, end_s, remap in block_remaps:
            block_pose = pose_df[
                (pose_df["timestamp_s"] >= start_s) & (pose_df["timestamp_s"] < end_s)
            ].copy()
            keep_ids = set(remap.keys())
            block_pose = block_pose[block_pose["track_id"].isin(keep_ids)].copy()
            block_pose["track_id"] = block_pose["track_id"].map(remap)
            block_pose["track_label"] = block_pose["track_id"].map(
                {0: "parent", 1: "child"}
            )
            filtered_pose_rows.append(block_pose)

        if filtered_pose_rows:
            result_pose = pd.concat(filtered_pose_rows, ignore_index=True)
        else:
            result_pose = pd.DataFrame(columns=pose_df.columns)

        out_pose = camera_dir / "pose_3d_filtered.csv"
        result_pose.to_csv(out_pose, index=False)
        print(f"[filter] wrote {out_pose} ({len(result_pose)} rows)")
        summary["pose_output"] = str(out_pose)

    return summary


def parse_blocks_string(blocks_str: str) -> List[BlockDef]:
    """Parse CLI blocks string like ``name,start,end;name,start,end``."""
    blocks = []
    for entry in blocks_str.split(";"):
        parts = entry.strip().split(",")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid block definition: {entry!r}. "
                "Expected format: name,start,end"
            )
        blocks.append(
            BlockDef(
                name=parts[0].strip(),
                start_time=parts[1].strip(),
                end_time=parts[2].strip(),
            )
        )
    return blocks


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="video-filter-tracks",
        description="Filter tracks by task blocks, keeping only parent+child.",
    )
    parser.add_argument("--session-dir", required=True, type=str)
    parser.add_argument("--camera", default="camera_a", type=str)
    parser.add_argument(
        "--blocks",
        required=True,
        type=str,
        help="Block definitions: name,start,end;name,start,end",
    )
    parser.add_argument("--source-fps", default=30.0, type=float)
    parser.add_argument(
        "--n-keep", default=2, type=int, help="Number of tracks to keep per block"
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    blocks = parse_blocks_string(args.blocks)
    cfg = TrackFilterConfig(
        session_dir=args.session_dir,
        camera=args.camera,
        blocks=blocks,
        source_fps=args.source_fps,
        n_keep=args.n_keep,
    )
    summary = filter_tracks(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
