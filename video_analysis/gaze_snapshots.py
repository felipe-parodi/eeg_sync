"""Generate gaze overlay snapshot grids per interaction block.

For each block, samples frames at regular intervals and draws:
- Head bboxes for parent (orange) and child (blue)
- Gaze target points (circles) where each person is looking
- Gaze vector lines from head center to gaze target
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from gaze_analysis.config import SessionBlock, load_session_config

# Track colors: parent = orange, child = blue
TRACK_COLORS = {
    0: (0, 140, 255),  # orange (BGR)
    1: (255, 140, 0),  # blue (BGR)
}
TRACK_LABELS = {0: "Parent", 1: "Child"}


def _find_frame_paths(camera_dir: Path) -> Dict[int, Path]:
    """Map frame_idx to frame file paths.

    Args:
        camera_dir: Camera output directory with frames/ subdirectory.

    Returns:
        Dict mapping frame_idx to Path.
    """
    frames_dir = camera_dir / "frames"
    if not frames_dir.exists():
        return {}
    frame_paths: Dict[int, Path] = {}
    for p in sorted(frames_dir.glob("frame_*.jpg")):
        try:
            idx = int(p.stem.split("_")[1])
            frame_paths[idx] = p
        except (IndexError, ValueError):
            continue
    return frame_paths


def draw_gaze_overlay(
    image: np.ndarray,
    gaze_rows: pd.DataFrame,
    head_rows: pd.DataFrame,
    image_width: int,
    image_height: int,
    parent_id: int = 0,
    child_id: int = 1,
) -> np.ndarray:
    """Draw gaze vectors and head bboxes on a single frame.

    Args:
        image: BGR image array.
        gaze_rows: Gaze rows for this frame (track_id, gaze_peak_x/y, inout_score).
        head_rows: Head bbox rows for this frame (track_id, head_x1/y1/x2/y2).
        image_width: Frame width.
        image_height: Frame height.
        parent_id: Parent track ID.
        child_id: Child track ID.

    Returns:
        Annotated image.
    """
    img = image.copy()
    h, w = img.shape[:2]

    for _, hrow in head_rows.iterrows():
        tid = int(hrow["track_id"])
        color = TRACK_COLORS.get(tid, (200, 200, 200))
        label = TRACK_LABELS.get(tid, f"T{tid}")

        # Head bbox (normalized -> pixel)
        x1 = int(hrow["head_x1"] * w)
        y1 = int(hrow["head_y1"] * h)
        x2 = int(hrow["head_x2"] * w)
        y2 = int(hrow["head_y2"] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label
        cv2.putText(
            img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        # Head center
        hcx = (x1 + x2) // 2
        hcy = (y1 + y2) // 2

        # Find matching gaze row
        gaze_match = gaze_rows[gaze_rows["track_id"] == tid]
        if gaze_match.empty:
            continue
        grow = gaze_match.iloc[0]

        # Gaze target (normalized -> pixel)
        gx = int(grow["gaze_peak_x"] * w)
        gy = int(grow["gaze_peak_y"] * h)

        inout = grow.get("inout_score", 1.0)

        # Draw gaze vector line
        cv2.arrowedLine(img, (hcx, hcy), (gx, gy), color, 2, tipLength=0.05)

        # Draw gaze target circle
        cv2.circle(img, (gx, gy), 8, color, 2)
        cv2.circle(img, (gx, gy), 3, color, -1)

        # Inout score label
        cv2.putText(
            img,
            f"io={inout:.2f}",
            (x2 + 3, y1 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )

    return img


def generate_block_snapshots(
    camera_dir: Path,
    gaze_df: pd.DataFrame,
    head_bboxes_df: pd.DataFrame,
    blocks: List[SessionBlock],
    output_dir: Path,
    n_samples: int = 8,
    image_width: int = 854,
    image_height: int = 480,
    parent_id: int = 0,
    child_id: int = 1,
) -> List[Path]:
    """Generate gaze overlay snapshot grids for each block.

    Args:
        camera_dir: Camera directory with frames/ subdirectory.
        gaze_df: Gaze CSV DataFrame.
        head_bboxes_df: Head bbox DataFrame.
        blocks: Session block definitions.
        output_dir: Directory to save snapshot images.
        n_samples: Number of frames to sample per block.
        image_width: Frame width.
        image_height: Frame height.
        parent_id: Parent track ID.
        child_id: Child track ID.

    Returns:
        List of saved file paths.
    """
    frame_paths = _find_frame_paths(camera_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for block in blocks:
        # Filter gaze to block
        block_gaze = gaze_df[
            (gaze_df["timestamp_s"] >= block.start_s)
            & (gaze_df["timestamp_s"] <= block.end_s)
        ]
        if block_gaze.empty:
            print(f"[gaze-snapshots] no gaze data for block '{block.name}', skipping")
            continue

        # Get unique frames with gaze data
        available_frames = sorted(
            set(block_gaze["frame_idx"].unique()) & set(frame_paths.keys())
        )
        if not available_frames:
            continue

        # Sample evenly
        n = min(n_samples, len(available_frames))
        indices = np.linspace(0, len(available_frames) - 1, n, dtype=int)
        sampled_frames = [available_frames[i] for i in indices]

        # Generate annotated frames
        annotated_images = []
        for fidx in sampled_frames:
            img = cv2.imread(str(frame_paths[fidx]))
            if img is None:
                continue

            frame_gaze = block_gaze[block_gaze["frame_idx"] == fidx]
            frame_heads = head_bboxes_df[head_bboxes_df["frame_idx"] == fidx]

            annotated = draw_gaze_overlay(
                img, frame_gaze, frame_heads,
                image_width, image_height, parent_id, child_id,
            )

            # Add timestamp label
            ts = frame_gaze["timestamp_s"].iloc[0] if not frame_gaze.empty else 0
            minutes = int(ts // 60)
            seconds = ts % 60
            label = f"{block.name} | {minutes:02d}:{seconds:04.1f} (f{fidx})"
            cv2.putText(
                annotated, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.putText(
                annotated, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
            )

            annotated_images.append(annotated)

        if not annotated_images:
            continue

        # Create contact sheet grid
        grid = _make_contact_sheet(annotated_images, n_cols=4)

        # Save individual frames
        for i, (fidx, img) in enumerate(zip(sampled_frames, annotated_images)):
            ts = block_gaze[block_gaze["frame_idx"] == fidx]["timestamp_s"].iloc[0]
            minutes = int(ts // 60)
            secs = int(ts % 60)
            fname = f"gaze_{block.name}_{i:02d}_{minutes}m{secs:02d}s_f{fidx}.jpg"
            path = output_dir / fname
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Save contact sheet
        grid_path = output_dir / f"gaze_grid_{block.name}.jpg"
        cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(grid_path)
        print(
            f"[gaze-snapshots] saved {n} frames + grid for block '{block.name}' "
            f"to {output_dir}"
        )

    return saved_paths


def _make_contact_sheet(
    images: List[np.ndarray],
    n_cols: int = 4,
    padding: int = 4,
) -> np.ndarray:
    """Arrange images in a grid contact sheet.

    Args:
        images: List of BGR images (all same size).
        n_cols: Number of columns.
        padding: Pixel padding between images.

    Returns:
        Contact sheet image.
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    h, w = images[0].shape[:2]
    n_rows = (len(images) + n_cols - 1) // n_cols

    sheet_h = n_rows * h + (n_rows - 1) * padding
    sheet_w = n_cols * w + (n_cols - 1) * padding
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // n_cols
        c = idx % n_cols
        y = r * (h + padding)
        x = c * (w + padding)
        # Resize if needed
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        sheet[y : y + h, x : x + w] = img

    return sheet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="video-gaze-snapshots",
        description="Generate gaze overlay snapshot grids per block.",
    )
    parser.add_argument("--camera-dir", required=True, type=str)
    parser.add_argument("--session-config", required=True, type=str)
    parser.add_argument("--camera-id", default="camera_a", type=str)
    parser.add_argument("--gaze-csv", default="gaze_heatmap.csv", type=str)
    parser.add_argument("--pose-input", default="pose_3d_filtered_5hz.csv", type=str)
    parser.add_argument("--tracks-input", default="tracks_2d_filtered_5hz.csv", type=str)
    parser.add_argument("--n-samples", default=8, type=int)
    parser.add_argument("--output-dir", default="block_checks", type=str)
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    camera_dir = Path(args.camera_dir)
    config = load_session_config(args.session_config)
    mapping = config.get_camera_mapping(args.camera_id)
    blocks = config.session_blocks

    gaze_df = pd.read_csv(camera_dir / args.gaze_csv)
    pose_df = pd.read_csv(camera_dir / args.pose_input)
    tracks_df = pd.read_csv(camera_dir / args.tracks_input)

    from gaze_analysis.head_bbox import extract_head_bboxes

    head_bboxes_df = extract_head_bboxes(
        pose_df, tracks_df, config.image_width, config.image_height
    )

    output_dir = camera_dir / args.output_dir
    generate_block_snapshots(
        camera_dir,
        gaze_df,
        head_bboxes_df,
        blocks,
        output_dir,
        n_samples=args.n_samples,
        image_width=config.image_width,
        image_height=config.image_height,
        parent_id=mapping.parent_track_id,
        child_id=mapping.child_track_id,
    )


if __name__ == "__main__":
    main()
