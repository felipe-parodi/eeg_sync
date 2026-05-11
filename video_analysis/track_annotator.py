"""Interactive track ID annotator using OpenCV.

Allows the user to view inference frames with bounding-box overlays,
click on a box to select it, and press 0-3 to reassign its track ID.
Corrections are saved to a JSON file and can later be applied with
``track_correction.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from video_analysis.visualize_pose_tracks import TRACK_COLORS

# Key constants.
_KEY_ESC = 27

# Arrow keys vary by platform.  cv2.waitKeyEx() returns full codes.
# Windows: LEFT=2424832, RIGHT=2555904, UP=2490368, DOWN=2621440
# Linux:   LEFT=65361,   RIGHT=65363,   UP=65362,   DOWN=65364
_ARROW_LEFT_CODES = {81, 2, 2424832, 65361}
_ARROW_RIGHT_CODES = {83, 3, 2555904, 65363}
_ARROW_UP_CODES = {82, 0, 2490368, 65362}
_ARROW_DOWN_CODES = {84, 1, 2621440, 65364}

MAX_DISPLAY_WIDTH = 1920


def _color_for_track(track_id: int) -> Tuple[int, int, int]:
    """Return BGR colour for *track_id*."""
    return TRACK_COLORS.get(track_id, (180, 180, 180))


@dataclass
class TrackAnnotatorConfig:
    """Configuration for the interactive track annotator."""

    session_dir: str
    camera: str = "camera_a"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    iou_threshold: float = 0.3
    keyframe_interval_s: float = 60.0
    display_fps: float = 5.0


@dataclass
class _AnnotatorState:
    """Mutable runtime state for the annotator loop."""

    tracks_df: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    frame_indices: List[int] = field(default_factory=list)
    display_indices: List[int] = field(default_factory=list)
    cursor: int = 0
    selected_bbox_idx: int = -1
    corrections: Dict[int, Dict[int, int]] = field(default_factory=dict)
    needs_redraw: bool = True
    fps: float = 30.0


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------


def _parse_time(time_str: str) -> float:
    """Parse ``MM:SS`` or ``HH:MM:SS`` to seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Cannot parse time: {time_str!r}")


def _format_time(seconds: float) -> str:
    """Format seconds as ``MM:SS.f``."""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def load_tracks(
    tracks_csv: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> pd.DataFrame:
    """Load and filter tracks CSV.

    Args:
        tracks_csv: Path to tracks_2d.csv.
        start_frame: First frame to include.
        end_frame: Last frame to include (exclusive).

    Returns:
        Filtered DataFrame.

    Raises:
        FileNotFoundError: If *tracks_csv* does not exist.
    """
    if not tracks_csv.exists():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_csv}")
    df = pd.read_csv(tracks_csv)
    if start_frame > 0:
        df = df[df["frame_idx"] >= start_frame]
    if end_frame is not None:
        df = df[df["frame_idx"] < end_frame]
    return df.reset_index(drop=True)


def _bboxes_for_frame(
    tracks_df: pd.DataFrame,
    frame_idx: int,
) -> List[Dict[str, Any]]:
    """Return list of bbox dicts for a single frame.

    Each dict has keys: track_id, x1, y1, x2, y2, confidence.
    """
    sub = tracks_df[tracks_df["frame_idx"] == frame_idx]
    boxes: List[Dict[str, Any]] = []
    for _, row in sub.iterrows():
        boxes.append(
            {
                "track_id": int(row["track_id"]),
                "x1": float(row["bbox_x1"]),
                "y1": float(row["bbox_y1"]),
                "x2": float(row["bbox_x2"]),
                "y2": float(row["bbox_y2"]),
                "confidence": float(row.get("track_confidence", 1.0)),
            }
        )
    return boxes


def _point_in_bbox(
    px: int,
    py: int,
    box: Dict[str, Any],
    scale: float,
) -> bool:
    """Check if pixel (px, py) in the *displayed* image falls inside *box*."""
    return (
        box["x1"] * scale <= px <= box["x2"] * scale
        and box["y1"] * scale <= py <= box["y2"] * scale
    )


def bbox_iou_dict(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Compute IoU between two bbox dicts (keys x1,y1,x2,y2)."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# ------------------------------------------------------------------
# Propagation
# ------------------------------------------------------------------


def propagate_corrections(
    tracks_df: pd.DataFrame,
    corrections: Dict[int, Dict[int, int]],
    source_frame: int,
    frame_indices: List[int],
    iou_threshold: float = 0.3,
) -> Dict[int, Dict[int, int]]:
    """Propagate corrections from *source_frame* forward via IoU.

    For each corrected track at *source_frame*, follow it forward through
    subsequent frames by matching the bbox with highest IoU. Stop when IoU
    drops below *iou_threshold* or we reach another user-annotated frame.

    Args:
        tracks_df: Full tracks DataFrame.
        corrections: Current corrections dict ``{frame_idx: {old_id: new_id}}``.
        source_frame: The frame whose corrections to propagate.
        frame_indices: Sorted list of all frame indices.
        iou_threshold: Minimum IoU to continue propagation.

    Returns:
        Updated corrections dict (mutated in-place and returned).
    """
    if source_frame not in corrections:
        return corrections

    src_mapping = corrections[source_frame]  # old_id -> new_id
    src_boxes = _bboxes_for_frame(tracks_df, source_frame)

    # Build bbox reference for each corrected track (keyed by new_id).
    ref_boxes: Dict[int, Dict[str, Any]] = {}
    for box in src_boxes:
        old_id = box["track_id"]
        if old_id in src_mapping:
            new_id = src_mapping[old_id]
            ref_boxes[new_id] = box

    if not ref_boxes:
        return corrections

    # Walk forward.
    src_pos = frame_indices.index(source_frame)
    propagated = 0
    for idx in frame_indices[src_pos + 1 :]:
        # Stop at the next user-annotated frame.
        if idx in corrections:
            break

        cur_boxes = _bboxes_for_frame(tracks_df, idx)
        if not cur_boxes:
            break

        frame_remap: Dict[int, int] = {}
        new_ref: Dict[int, Dict[str, Any]] = {}

        for new_id, ref_box in ref_boxes.items():
            best_iou = 0.0
            best_box = None
            for cb in cur_boxes:
                iou = bbox_iou_dict(ref_box, cb)
                if iou > best_iou:
                    best_iou = iou
                    best_box = cb
            if best_box is not None and best_iou >= iou_threshold:
                if best_box["track_id"] != new_id:
                    frame_remap[best_box["track_id"]] = new_id
                new_ref[new_id] = best_box

        if frame_remap:
            corrections[idx] = frame_remap
            propagated += 1

        if not new_ref:
            break
        ref_boxes = new_ref

    print(f"[annotator] propagated corrections to {propagated} frames")
    return corrections


# ------------------------------------------------------------------
# Drawing
# ------------------------------------------------------------------


def _draw_overlay(
    image: np.ndarray,
    boxes: List[Dict[str, Any]],
    frame_idx: int,
    state: _AnnotatorState,
    scale: float,
    corrections: Dict[int, Dict[int, int]],
) -> np.ndarray:
    """Draw bboxes, labels, and HUD onto *image* (already scaled)."""
    frame_corr = corrections.get(frame_idx, {})

    for i, box in enumerate(boxes):
        original_id = box["track_id"]
        display_id = frame_corr.get(original_id, original_id)
        color = _color_for_track(display_id)

        x1 = int(box["x1"] * scale)
        y1 = int(box["y1"] * scale)
        x2 = int(box["x2"] * scale)
        y2 = int(box["y2"] * scale)

        thickness = 4 if i == state.selected_bbox_idx else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label = f"id={display_id}"
        if original_id != display_id:
            label += f" (was {original_id})"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    # HUD background.
    cv2.rectangle(image, (8, 8), (460, 100), (0, 0, 0), -1)
    ts = frame_idx / state.fps
    cv2.putText(
        image,
        f"Frame {frame_idx}  |  {_format_time(ts)}",
        (16, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    n_corr = sum(len(v) for v in corrections.values())
    cv2.putText(
        image,
        f"Corrections: {n_corr}  |  Click bbox, press 0-3",
        (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    sel_msg = (
        f"Selected: box #{state.selected_bbox_idx}"
        if state.selected_bbox_idx >= 0
        else "No box selected"
    )
    cv2.putText(
        image,
        sel_msg,
        (16, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    return image


# ------------------------------------------------------------------
# Main interactive loop
# ------------------------------------------------------------------


def run_annotator(config: TrackAnnotatorConfig) -> Optional[Path]:
    """Launch the interactive track annotator.

    Args:
        config: Annotator configuration.

    Returns:
        Path to saved corrections JSON, or None if cancelled.
    """
    session_dir = Path(config.session_dir)
    camera_dir = session_dir / config.camera
    tracks_csv = camera_dir / "tracks_2d.csv"
    frames_dir = camera_dir / "frames"

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

    # Determine frame range.
    start_frame = 0
    end_frame = None
    if config.start_time:
        start_frame = int(_parse_time(config.start_time) * 30)
    if config.end_time:
        end_frame = int(_parse_time(config.end_time) * 30)

    tracks_df = load_tracks(tracks_csv, start_frame, end_frame)
    if tracks_df.empty:
        print("[annotator] No tracks in the specified range.")
        return None

    frame_indices = sorted(tracks_df["frame_idx"].unique())

    # Subsample for display; propagation still uses all frames.
    step = max(1, int(30.0 / config.display_fps))
    display_indices = frame_indices[::step]

    state = _AnnotatorState(
        tracks_df=tracks_df,
        frame_indices=frame_indices,
        display_indices=display_indices,
        fps=30.0,
    )

    # Load existing corrections if present.
    corrections_path = camera_dir / "track_corrections.json"
    if corrections_path.exists():
        with open(corrections_path) as f:
            raw = json.load(f)
        state.corrections = {int(k): v for k, v in raw.items()}
        print(f"[annotator] loaded {len(state.corrections)} existing corrections")

    # Display scaling.
    sample_frame_path = frames_dir / f"frame_{frame_indices[0] + 1:06d}.jpg"
    sample = cv2.imread(str(sample_frame_path))
    if sample is None:
        raise FileNotFoundError(f"Cannot read frame: {sample_frame_path}")
    img_h, img_w = sample.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / img_w)

    # Mouse callback state.
    click_xy: List[Optional[Tuple[int, int]]] = [None]

    def _on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            click_xy[0] = (x, y)

    window_name = "Track Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)

    print(f"\n{'=' * 60}")
    print("TRACK ANNOTATOR")
    print(f"{'=' * 60}")
    print(
        f"Frames: {frame_indices[0]}-{frame_indices[-1]} "
        f"({len(frame_indices)} total, {len(display_indices)} displayed "
        f"at {config.display_fps:.0f} Hz)"
    )
    print("\nControls:")
    print("  Click bbox      Select a bounding box")
    print("  0/1/2/3         Assign track ID to selected box")
    print("  P               Propagate corrections forward")
    print("  S               Save corrections")
    print("  N               Jump to next keyframe (+1 min)")
    print("  B               Jump to prev keyframe (-1 min)")
    print("  LEFT/RIGHT, A/D +/-1 frame")
    print("  UP/DOWN, W/S    +/-30 frames (~1 sec)")
    print("  [ / ]            +/-1 minute")
    print("  Q / ESC          Quit (auto-saves)")

    try:
        while True:
            frame_idx = state.display_indices[state.cursor]

            if state.needs_redraw:
                # Load image.
                img_path = frames_dir / f"frame_{frame_idx + 1:06d}.jpg"
                img = cv2.imread(str(img_path))
                if img is None:
                    state.cursor = min(
                        state.cursor + 1, len(state.display_indices) - 1
                    )
                    continue

                if scale < 1.0:
                    img = cv2.resize(
                        img,
                        (int(img_w * scale), int(img_h * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                boxes = _bboxes_for_frame(tracks_df, frame_idx)
                display = _draw_overlay(
                    img,
                    boxes,
                    frame_idx,
                    state,
                    scale,
                    state.corrections,
                )
                cv2.imshow(window_name, display)
                state.needs_redraw = False

            # Process click.
            if click_xy[0] is not None:
                px, py = click_xy[0]
                click_xy[0] = None
                boxes = _bboxes_for_frame(tracks_df, frame_idx)
                state.selected_bbox_idx = -1
                for i, box in enumerate(boxes):
                    if _point_in_bbox(px, py, box, scale):
                        state.selected_bbox_idx = i
                        break
                state.needs_redraw = True
                continue

            key = cv2.waitKeyEx(30)
            if key == -1:
                continue

            # Quit.
            if key == ord("q") or key == _KEY_ESC:
                break

            # Assign track ID (0-3).
            if key in (ord("0"), ord("1"), ord("2"), ord("3")):
                new_id = key - ord("0")
                boxes = _bboxes_for_frame(tracks_df, frame_idx)
                if 0 <= state.selected_bbox_idx < len(boxes):
                    old_id = boxes[state.selected_bbox_idx]["track_id"]
                    if old_id != new_id:
                        remap = state.corrections.setdefault(frame_idx, {})
                        remap[old_id] = new_id
                        print(
                            f"[annotator] frame {frame_idx}: "
                            f"track {old_id} -> {new_id}"
                        )
                    state.selected_bbox_idx = -1
                    state.needs_redraw = True
                continue

            # Propagate.
            if key == ord("p"):
                propagate_corrections(
                    tracks_df,
                    state.corrections,
                    frame_idx,
                    frame_indices,
                    config.iou_threshold,
                )
                state.needs_redraw = True
                continue

            # Navigation.
            new_cursor = _navigate(key, state, config)
            if new_cursor != state.cursor:
                state.cursor = new_cursor
                state.selected_bbox_idx = -1
                state.needs_redraw = True

    finally:
        _save_corrections(state.corrections, corrections_path)
        cv2.destroyAllWindows()

    return corrections_path


def _save_corrections(
    corrections: Dict[int, Dict[int, int]],
    path: Path,
) -> None:
    """Write corrections to JSON."""
    # JSON keys must be strings.
    out = {str(k): v for k, v in corrections.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    n = sum(len(v) for v in corrections.values())
    print(f"[annotator] saved {n} corrections to {path}")


def _navigate(
    key: int,
    state: _AnnotatorState,
    config: TrackAnnotatorConfig,
) -> int:
    """Return new cursor position based on key press.

    Cursor indexes into ``state.display_indices`` (subsampled frames).
    """
    last = len(state.display_indices) - 1

    # ±1 displayed frame
    if key in _ARROW_RIGHT_CODES or key == ord("d"):
        return min(state.cursor + 1, last)
    if key in _ARROW_LEFT_CODES or key == ord("a"):
        return max(state.cursor - 1, 0)

    # ±~1 second worth of displayed frames
    step_sec = max(1, int(config.display_fps))
    if key in _ARROW_UP_CODES or key == ord("w"):
        return min(state.cursor + step_sec, last)
    if key in _ARROW_DOWN_CODES or key == ord("s"):
        return max(state.cursor - step_sec, 0)

    # ±1 minute worth of displayed frames
    step_min = max(1, int(config.keyframe_interval_s * config.display_fps))
    if key == ord("]") or key == ord("}"):
        return min(state.cursor + step_min, last)
    if key == ord("[") or key == ord("{"):
        return max(state.cursor - step_min, 0)

    # N / B — next/prev keyframe (same as ±1 min)
    if key == ord("n"):
        return min(state.cursor + step_min, last)
    if key == ord("b"):
        return max(state.cursor - step_min, 0)

    return state.cursor


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="video-annotate-tracks",
        description="Interactive track ID annotator for correcting ByteTrack IDs.",
    )
    parser.add_argument("--session-dir", required=True, type=str)
    parser.add_argument("--camera", default="camera_a", type=str)
    parser.add_argument(
        "--start-time", default=None, type=str, help="Start time in MM:SS format"
    )
    parser.add_argument(
        "--end-time", default=None, type=str, help="End time in MM:SS format"
    )
    parser.add_argument("--iou-threshold", default=0.3, type=float)
    parser.add_argument(
        "--keyframe-interval",
        default=60.0,
        type=float,
        help="Seconds between keyframe jumps (N/B keys)",
    )
    parser.add_argument(
        "--display-fps",
        default=5.0,
        type=float,
        help="Display frame rate for navigation (default 5 Hz)",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = TrackAnnotatorConfig(
        session_dir=args.session_dir,
        camera=args.camera,
        start_time=args.start_time,
        end_time=args.end_time,
        iou_threshold=args.iou_threshold,
        keyframe_interval_s=args.keyframe_interval,
        display_fps=args.display_fps,
    )
    result = run_annotator(cfg)
    if result:
        print(f"\nCorrections saved to: {result}")


if __name__ == "__main__":
    main()
