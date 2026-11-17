"""Pick representative frames for human parent/child labeling and render them.

The portal workflow needs the colleague to confirm at sparse frames which
tracked person is the parent (and which is the child). Pure size-based
heuristics fail when the child sits closer to the camera. This module:

1. Reads a tracked-output ``tracks_2d.csv`` from a completed inference run.
2. Picks N anchor frames evenly time-spaced across the analysis window,
   filtering for clarity (exactly two detections with bbox overlap below
   ``max_iou``) and skipping the very start/end of the window.
3. Renders each anchor frame from the source video with bbox overlays
   labelling each track id and a distinct color per bbox.
4. Writes ``anchor_<NN>.jpg`` plus ``manifest.json`` so a portal UI can
   serve the images and collect user labels keyed by track id.

Standalone CLI: ``video-anchor-frames``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

# RGB colour wheel for bbox overlays. Same palette as the existing overlay
# video so the colleague's mental mapping stays consistent.
_BBOX_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0),  # green
    (0, 165, 255),  # orange
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
)


@dataclass
class AnchorCandidate:
    """A frame proposed for annotation, with the tracked detections in it."""

    frame_idx: int
    timestamp_s: float
    tracks: List[dict] = field(default_factory=list)


@dataclass
class AnchorMetadata:
    """Per-anchor metadata written to ``manifest.json``."""

    anchor_idx: int
    frame_idx: int
    timestamp_s: float
    image_filename: str
    tracks: List[dict]


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _frame_is_clear(frame_rows: pd.DataFrame, max_iou: float) -> bool:
    """A frame is "clear" when it has exactly two detections whose bboxes
    overlap by less than ``max_iou`` — ambiguous frames are excluded so the
    colleague can tell the two boxes apart at a glance."""
    if len(frame_rows) != 2:
        return False
    bboxes = frame_rows[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy()
    return _bbox_iou(bboxes[0], bboxes[1]) <= max_iou


MIN_ANCHORS_PER_BLOCK = 2


def _two_person_frames(
    tracks_df: pd.DataFrame, t_min: float, t_max: float
) -> List[Tuple[int, float, float, List[dict]]]:
    """Frames with exactly two detections: (frame_idx, time, iou, tracks)."""
    out: List[Tuple[int, float, float, List[dict]]] = []
    for frame_idx, frame_rows in tracks_df.groupby("frame_idx", sort=True):
        if len(frame_rows) != 2:
            continue
        ts = float(frame_rows["timestamp_s"].iloc[0])
        if ts < t_min or ts > t_max:
            continue
        boxes = frame_rows[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy()
        tracks = [
            {
                "track_id": int(row["track_id"]),
                "bbox_xyxy": [
                    float(row["bbox_x1"]),
                    float(row["bbox_y1"]),
                    float(row["bbox_x2"]),
                    float(row["bbox_y2"]),
                ],
            }
            for _, row in frame_rows.iterrows()
        ]
        out.append((int(frame_idx), ts, _bbox_iou(boxes[0], boxes[1]), tracks))
    return out


def _spread(items: list, k: int) -> list:
    """Pick k items evenly by rank (all of them when there are fewer)."""
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    picks = np.unique(np.round(np.linspace(0, len(items) - 1, k)).astype(int))
    return [items[int(i)] for i in picks]


def select_anchor_frames(
    tracks_df: pd.DataFrame,
    n_anchors: int = 10,
    max_iou: float = 0.3,
    edge_margin_s: float = 1.0,
    blocks: Sequence[Tuple[str, float, float]] | None = None,
    min_per_block: int = MIN_ANCHORS_PER_BLOCK,
) -> List[AnchorCandidate]:
    """Pick up to N evenly spread frames suitable for human annotation.

    Strategy:
        1. Drop the first/last ``edge_margin_s`` seconds (often startup or
           fade-out frames).
        2. Prefer *clear* frames (exactly two detections, bbox IoU <=
           ``max_iou``).
        3. When ``blocks`` are given, first give every block
           ``min_per_block`` anchors: its clear frames spread by rank, or,
           if it has too few (a child on the parent's lap), its two-person
           frames with the least overlap. Labels only steer roles near
           themselves, so a block without anchors can end up with swapped
           roles.
        4. Fill the remaining budget with clear frames spread by rank over
           the whole recording (time targets in gaps between blocks would
           all snap to the same frame).
        5. Sort the result by timestamp.

    Returns fewer than ``n_anchors`` candidates when there are not enough
    usable frames; with many blocks it may return more, so that each block
    keeps ``min_per_block``.
    """
    if tracks_df.empty or n_anchors < 1:
        return []
    t_min = float(tracks_df["timestamp_s"].min()) + edge_margin_s
    t_max = float(tracks_df["timestamp_s"].max()) - edge_margin_s
    if t_max <= t_min:
        return []
    frames = _two_person_frames(tracks_df, t_min, t_max)
    clear = [f for f in frames if f[2] <= max_iou]
    chosen: dict[int, Tuple[int, float, float, List[dict]]] = {}

    for _name, start, end in blocks or []:
        in_block = [f for f in frames if start <= f[1] <= end]
        block_clear = [f for f in in_block if f[2] <= max_iou]
        picks = _spread(block_clear, min_per_block)
        if len(picks) < min_per_block:
            taken = {f[0] for f in picks}
            fallback = sorted(
                (f for f in in_block if f[0] not in taken), key=lambda f: f[2]
            )
            picks += fallback[: min_per_block - len(picks)]
        for f in picks:
            chosen[f[0]] = f

    remaining = n_anchors - len(chosen)
    if remaining > 0:
        pool = [f for f in clear if f[0] not in chosen]
        for f in _spread(pool, remaining):
            chosen[f[0]] = f

    candidates = [
        AnchorCandidate(frame_idx=f[0], timestamp_s=f[1], tracks=f[3])
        for f in chosen.values()
    ]
    candidates.sort(key=lambda c: c.timestamp_s)
    return candidates


def _extract_frame_at_time(
    source_video: Path,
    timestamp_s: float,
    output_path: Path,
    ffmpeg_bin: str,
) -> None:
    """Use ffmpeg to extract a single JPEG frame at ``timestamp_s``."""
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp_s:.6f}",
        "-i",
        str(source_video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _draw_overlay(
    raw_image_path: Path,
    tracks: List[dict],
    output_path: Path,
) -> None:
    """Draw labelled bboxes onto a frame and write the result."""
    import cv2  # imported lazily; cv2 is heavy on import

    img = cv2.imread(str(raw_image_path))
    if img is None:
        raise RuntimeError(f"Could not read extracted frame: {raw_image_path}")
    for i, track in enumerate(tracks):
        x1, y1, x2, y2 = (int(round(float(v))) for v in track["bbox_xyxy"])
        color = _BBOX_COLORS_BGR[i % len(_BBOX_COLORS_BGR)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"track {track['track_id']}"
        text_y = max(0, y1 - 10)
        cv2.putText(
            img,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), img)


def extract_anchor_frames(
    tracks_csv: Path,
    source_video: Path,
    output_dir: Path,
    n_anchors: int = 10,
    max_iou: float = 0.3,
    edge_margin_s: float = 1.0,
    ffmpeg_bin: str = "ffmpeg",
    blocks: Sequence[Tuple[str, float, float]] | None = None,
) -> List[AnchorMetadata]:
    """End-to-end anchor frame extraction + overlay rendering.

    Layout written under ``output_dir``::

        anchor_01.jpg
        anchor_02.jpg
        ...
        manifest.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks_df = pd.read_csv(tracks_csv)
    candidates = select_anchor_frames(
        tracks_df,
        n_anchors=n_anchors,
        max_iou=max_iou,
        edge_margin_s=edge_margin_s,
        blocks=blocks,
    )

    metadata: List[AnchorMetadata] = []
    for anchor_idx, candidate in enumerate(candidates, start=1):
        filename = f"anchor_{anchor_idx:02d}.jpg"
        out_path = output_dir / filename
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_handle:
            tmp_path = Path(tmp_handle.name)
        try:
            _extract_frame_at_time(
                source_video=Path(source_video),
                timestamp_s=candidate.timestamp_s,
                output_path=tmp_path,
                ffmpeg_bin=ffmpeg_bin,
            )
            _draw_overlay(
                raw_image_path=tmp_path,
                tracks=candidate.tracks,
                output_path=out_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        tracks_with_color = [
            {
                **track,
                "color_bgr": list(_BBOX_COLORS_BGR[i % len(_BBOX_COLORS_BGR)]),
            }
            for i, track in enumerate(candidate.tracks)
        ]
        metadata.append(
            AnchorMetadata(
                anchor_idx=anchor_idx,
                frame_idx=candidate.frame_idx,
                timestamp_s=candidate.timestamp_s,
                image_filename=filename,
                tracks=tracks_with_color,
            )
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(m) for m in metadata], indent=2),
        encoding="utf-8",
    )
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-anchor-frames",
        description=(
            "Pick N representative frames from a tracked session and render "
            "them with bbox overlays for manual parent/child labelling."
        ),
    )
    parser.add_argument("--tracks-csv", required=True, type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-anchors", default=10, type=int)
    parser.add_argument(
        "--max-iou",
        default=0.3,
        type=float,
        help="Skip frames whose two detections overlap by more than this.",
    )
    parser.add_argument(
        "--edge-margin-s",
        default=1.0,
        type=float,
        help="Seconds to drop from both ends of the time window.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", type=str)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    metadata = extract_anchor_frames(
        tracks_csv=args.tracks_csv,
        source_video=args.source_video,
        output_dir=args.output_dir,
        n_anchors=args.n_anchors,
        max_iou=args.max_iou,
        edge_margin_s=args.edge_margin_s,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    print(json.dumps([asdict(m) for m in metadata], indent=2))


if __name__ == "__main__":
    main()
