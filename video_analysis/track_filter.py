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
    if block.empty:
        return []

    # Count frames per track.
    counts = block.groupby("track_id")["frame_idx"].nunique()
    top_ids = counts.nlargest(n_keep).index.tolist()

    if not top_ids:
        return []

    # Compute average bbox area per track.
    sub = block[block["track_id"].isin(top_ids)].copy()
    sub["_area"] = sub.apply(_bbox_area, axis=1)
    avg_areas = sub.groupby("track_id")["_area"].mean()

    # Sort by area descending: largest = parent (role 0), smallest = child (role 1).
    sorted_ids = avg_areas.sort_values(ascending=False).index.tolist()
    result = []
    for role_idx, tid in enumerate(sorted_ids):
        role = "parent" if role_idx == 0 else "child"
        result.append((int(tid), role, float(avg_areas[tid])))
    return result


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
    fps = config.source_fps
    blocks = config.blocks or []

    # Build block frame ranges and track mappings.
    block_summaries: List[Dict[str, Any]] = []
    # Collect (start_frame, end_frame, {old_id: new_id}) per block.
    block_remaps: List[Tuple[int, int, Dict[int, int]]] = []

    for blk in blocks:
        sf = _time_to_frame(blk.start_time, fps)
        ef = _time_to_frame(blk.end_time, fps)
        top = identify_top_tracks(tracks_df, sf, ef, config.n_keep)

        remap: Dict[int, int] = {}
        summary_entry: Dict[str, Any] = {
            "name": blk.name,
            "start_frame": sf,
            "end_frame": ef,
            "tracks": [],
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

        block_remaps.append((sf, ef, remap))
        block_summaries.append(summary_entry)

    # Apply filtering: keep only remapped tracks within blocks,
    # drop everything else (gaps between blocks).
    filtered_rows = []
    for sf, ef, remap in block_remaps:
        block_df = tracks_df[
            (tracks_df["frame_idx"] >= sf) & (tracks_df["frame_idx"] <= ef)
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

    # Also filter pose CSV if it exists.
    if pose_path.exists():
        pose_df = pd.read_csv(pose_path)
        filtered_pose_rows = []
        for sf, ef, remap in block_remaps:
            block_pose = pose_df[
                (pose_df["frame_idx"] >= sf) & (pose_df["frame_idx"] <= ef)
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
