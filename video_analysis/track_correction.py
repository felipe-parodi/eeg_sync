"""Apply track ID corrections to tracks and pose CSVs.

Reads a ``track_corrections.json`` produced by the interactive annotator
and rewrites ``tracks_2d.csv`` and ``pose_3d.csv`` with corrected IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_corrections(path: Path) -> Dict[int, Dict[int, int]]:
    """Load corrections JSON.

    Args:
        path: Path to ``track_corrections.json``.

    Returns:
        Dict mapping ``frame_idx`` → ``{old_track_id: new_track_id}``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Corrections file not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    return {int(k): {int(ok): int(nv) for ok, nv in v.items()} for k, v in raw.items()}


def apply_corrections(
    df: pd.DataFrame,
    corrections: Dict[int, Dict[int, int]],
    id_column: str = "track_id",
) -> pd.DataFrame:
    """Apply track ID remapping to a DataFrame.

    For each frame in *corrections*, the ``id_column`` values are remapped
    according to the old→new mapping.  ID swaps are handled atomically
    per frame (all remaps applied simultaneously).

    Args:
        df: DataFrame with ``frame_idx`` and *id_column* columns.
        corrections: ``{frame_idx: {old_id: new_id}}`` mapping.
        id_column: Name of the column containing track IDs.

    Returns:
        New DataFrame with corrected IDs.
    """
    df = df.copy()
    for frame_idx, remap in corrections.items():
        mask = df["frame_idx"] == frame_idx
        if not mask.any():
            continue
        # Atomic swap: map through a temporary series.
        original = df.loc[mask, id_column].copy()
        df.loc[mask, id_column] = original.map(
            lambda tid, r=remap: r.get(int(tid), int(tid))
        )
    return df


def apply_corrections_to_session(
    camera_dir: Path,
    corrections_path: Path,
    tracks_input: str = "tracks_2d.csv",
    pose_input: str = "pose_3d.csv",
    tracks_output: str = "tracks_2d_corrected.csv",
    pose_output: str = "pose_3d_corrected.csv",
) -> Dict[str, Any]:
    """Apply corrections to tracks and pose CSVs in a session.

    Args:
        camera_dir: Path to the camera directory.
        corrections_path: Path to ``track_corrections.json``.
        tracks_input: Input tracks CSV filename.
        pose_input: Input pose CSV filename.
        tracks_output: Output tracks CSV filename.
        pose_output: Output pose CSV filename.

    Returns:
        Summary dict with file paths and correction counts.
    """
    corrections = load_corrections(corrections_path)
    n_frames = len(corrections)
    n_remaps = sum(len(v) for v in corrections.values())
    print(f"[track-correction] loaded {n_remaps} remaps across {n_frames} frames")

    tracks_path = camera_dir / tracks_input
    pose_path = camera_dir / pose_input

    summary: Dict[str, Any] = {
        "corrections_file": str(corrections_path),
        "n_corrected_frames": n_frames,
        "n_remaps": n_remaps,
    }

    if tracks_path.exists():
        tracks_df = pd.read_csv(tracks_path)
        tracks_df = apply_corrections(tracks_df, corrections)
        out_path = camera_dir / tracks_output
        tracks_df.to_csv(out_path, index=False)
        summary["tracks_output"] = str(out_path)
        print(f"[track-correction] wrote {out_path}")

    if pose_path.exists():
        pose_df = pd.read_csv(pose_path)
        pose_df = apply_corrections(pose_df, corrections)
        out_path = camera_dir / pose_output
        pose_df.to_csv(out_path, index=False)
        summary["pose_output"] = str(out_path)
        print(f"[track-correction] wrote {out_path}")

    return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="video-correct-tracks",
        description="Apply track ID corrections from annotator JSON.",
    )
    parser.add_argument("--session-dir", required=True, type=str)
    parser.add_argument("--camera", default="camera_a", type=str)
    parser.add_argument(
        "--corrections",
        default="track_corrections.json",
        type=str,
        help="Corrections JSON filename",
    )
    parser.add_argument("--tracks-input", default="tracks_2d.csv", type=str)
    parser.add_argument("--pose-input", default="pose_3d.csv", type=str)
    parser.add_argument("--tracks-output", default="tracks_2d_corrected.csv", type=str)
    parser.add_argument("--pose-output", default="pose_3d_corrected.csv", type=str)
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    session_dir = Path(args.session_dir)
    camera_dir = session_dir / args.camera
    corrections_path = camera_dir / args.corrections
    summary = apply_corrections_to_session(
        camera_dir=camera_dir,
        corrections_path=corrections_path,
        tracks_input=args.tracks_input,
        pose_input=args.pose_input,
        tracks_output=args.tracks_output,
        pose_output=args.pose_output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
