"""Gaze ground-truth labels collected in the portal.

The colleague marks, on a sample of frames, what the parent and the child are
looking at (and optionally where). These labels let us score gaze models on
this lab's own videos instead of trusting public benchmarks.

Labels live in ``<session>/<camera>/gaze_labels.json`` next to the outputs, so
they survive the portal clearing its job list on restart.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gaze_analysis.config import SessionBlock

GAZE_LABELS_FILENAME = "gaze_labels.json"
GAZE_LABEL_FRAMES_PER_CAMERA = 30
PERSONS = ("parent", "child")

# Stored value -> button text shown to the colleague.
GAZE_TARGETS: dict[str, str] = {
    "partner_face": "The other person's face",
    "partner_body": "The other person's body or hands",
    "object": "An object (book, toy, food, ...)",
    "elsewhere": "Somewhere else",
    "cant_tell": "Can't tell",
}


def sample_gaze_label_frames(
    camera_dir: Path,
    blocks: Sequence[SessionBlock],
    frame_rate: float,
    n_frames: int = GAZE_LABEL_FRAMES_PER_CAMERA,
) -> list[dict[str, Any]]:
    """Pick frames to label, spread evenly over the analysis blocks.

    Only frames where both people were directly observed (not gap-filled) and
    whose image exists are eligible. Selection is deterministic.

    Args:
        camera_dir: Camera output folder (``tracks_2d_smooth.csv``, ``frames/``).
        blocks: This camera's analysis blocks.
        frame_rate: Time-grid rate used by the smoothed tables.
        n_frames: Number of frames to return.

    Returns:
        One dict per frame with ``frame_idx``, ``timestamp_s``, ``block``,
        ``image_name`` and pixel ``boxes`` for ``parent`` and ``child``.
    """
    camera_dir = Path(camera_dir)
    index = pd.read_csv(camera_dir / "frames" / "frame_index.csv")
    images = dict(
        zip(
            np.rint(index["timestamp_s"].to_numpy(float) * frame_rate).astype(int),
            index["image_name"],
        )
    )
    tracks = pd.read_csv(camera_dir / "tracks_2d_smooth.csv")
    if "interpolated" in tracks.columns:
        tracks = tracks[~tracks["interpolated"].astype(bool)]
    tracks = tracks[tracks["track_label"].isin(PERSONS)]

    candidates: list[tuple[int, float, str]] = []
    for frame_idx, group in tracks.groupby("frame_idx", sort=True):
        if set(group["track_label"]) != set(PERSONS) or frame_idx not in images:
            continue
        t = float(group["timestamp_s"].iloc[0])
        block = next((b.name for b in blocks if b.start_s <= t <= b.end_s), None)
        if block is not None:
            candidates.append((int(frame_idx), t, block))
    if not candidates:
        return []

    picks = np.unique(
        np.linspace(0, len(candidates) - 1, min(n_frames, len(candidates)))
        .round()
        .astype(int)
    )
    by_frame = {k: g for k, g in tracks.groupby("frame_idx")}
    items = []
    for i in picks:
        frame_idx, t, block = candidates[i]
        group = by_frame[frame_idx]
        boxes = {
            row.track_label: [
                float(row.bbox_x1),
                float(row.bbox_y1),
                float(row.bbox_x2),
                float(row.bbox_y2),
            ]
            for row in group.itertuples()
        }
        items.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": round(t, 3),
                "block": block,
                "image_name": str(images[frame_idx]),
                "boxes": boxes,
            }
        )
    return items


def _validate(entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("person") not in PERSONS:
        raise ValueError(f"person must be one of {PERSONS}")
    if entry.get("target") not in GAZE_TARGETS:
        raise ValueError(f"target must be one of {list(GAZE_TARGETS)}")
    point = entry.get("point")
    if point is not None:
        if len(point) != 2 or not all(0.0 <= float(v) <= 1.0 for v in point):
            raise ValueError("point must be [x, y] in 0..1 image coordinates")
        point = [float(point[0]), float(point[1])]
    return {
        "frame_idx": int(entry["frame_idx"]),
        "timestamp_s": float(entry["timestamp_s"]),
        "person": entry["person"],
        "target": entry["target"],
        "point": point,
    }


def save_gaze_labels(
    camera_dir: Path, labels: Sequence[dict[str, Any]], labeler: str = ""
) -> Path:
    """Validate and save gaze labels for one camera.

    Args:
        camera_dir: Camera output folder.
        labels: Entries with ``frame_idx``, ``timestamp_s``, ``person``
            (parent/child), ``target`` (a :data:`GAZE_TARGETS` key) and an
            optional ``point`` ([x, y] as fractions of image width/height).
        labeler: Optional name or initials of the person labeling.

    Returns:
        Path of the written file.

    Raises:
        ValueError: If an entry is malformed.
    """
    clean = [_validate(dict(entry)) for entry in labels]
    path = Path(camera_dir) / GAZE_LABELS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "labeler": labeler,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "targets": GAZE_TARGETS,
                "labels": clean,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_gaze_labels(camera_dir: Path) -> list[dict[str, Any]]:
    """Load saved gaze labels for one camera (empty list if none)."""
    path = Path(camera_dir) / GAZE_LABELS_FILENAME
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("labels", [])
