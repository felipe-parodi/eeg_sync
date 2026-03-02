"""Session configuration for gaze analysis and synchrony metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CameraPersonMapping:
    """Maps track IDs to parent/child roles for one camera."""

    camera_id: str
    parent_track_id: int
    child_track_id: int


@dataclass
class SessionBlock:
    """A labeled time window within the session."""

    name: str
    start_s: float
    end_s: float
    color: str = "gray"


@dataclass
class SessionConfig:
    """Top-level configuration for a gaze analysis session.

    Attributes:
        session_id: Unique identifier for the session (e.g. "P001c").
        output_dir: Path to the video inference output directory.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        frame_rate: Video sampling rate in FPS.
        camera_mappings: Per-camera parent/child track ID assignments.
        session_blocks: Labeled time windows for analysis.
    """

    session_id: str
    output_dir: str
    image_width: int
    image_height: int
    frame_rate: float = 5.0
    camera_mappings: List[CameraPersonMapping] = field(default_factory=list)
    session_blocks: List[SessionBlock] = field(default_factory=list)

    def get_session_block(self, name: str) -> SessionBlock:
        """Return the session block with the given name.

        Args:
            name: Block name (e.g. "grocery_free_play").

        Returns:
            The matching SessionBlock.

        Raises:
            KeyError: If no block exists with that name.
        """
        for block in self.session_blocks:
            if block.name == name:
                return block
        available = [b.name for b in self.session_blocks]
        raise KeyError(
            f"No session block named '{name}'. Available: {available}"
        )

    def get_camera_mapping(self, camera_id: str) -> CameraPersonMapping:
        """Return the mapping for a given camera_id.

        Args:
            camera_id: Camera identifier (e.g. "camera_a").

        Returns:
            The CameraPersonMapping for this camera.

        Raises:
            KeyError: If no mapping exists for the camera_id.
        """
        for mapping in self.camera_mappings:
            if mapping.camera_id == camera_id:
                return mapping
        raise KeyError(f"No camera mapping found for '{camera_id}'")


def _validate_raw(data: Dict[str, Any]) -> None:
    """Validate raw JSON dict before constructing dataclasses.

    Args:
        data: Parsed JSON dict.

    Raises:
        ValueError: On any validation failure.
    """
    if "session_id" not in data:
        raise ValueError("session_id is required")
    if "camera_mappings" not in data or not data["camera_mappings"]:
        raise ValueError("camera_mappings is required and must be non-empty")

    for mapping in data["camera_mappings"]:
        if mapping.get("parent_track_id") == mapping.get("child_track_id"):
            raise ValueError(
                f"parent_track_id and child_track_id must differ "
                f"(camera '{mapping.get('camera_id', '?')}')"
            )

    for block in data.get("session_blocks", []):
        if block.get("start_s", 0) > block.get("end_s", 0):
            raise ValueError(
                f"Block '{block.get('name', '?')}': "
                f"start_s ({block['start_s']}) must be <= end_s ({block['end_s']})"
            )


def load_session_config(path: Path | str) -> SessionConfig:
    """Load and validate a session config from a JSON file.

    Args:
        path: Path to the JSON config file.

    Returns:
        Validated SessionConfig instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On validation errors.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    _validate_raw(data)

    camera_mappings = [
        CameraPersonMapping(
            camera_id=m["camera_id"],
            parent_track_id=m["parent_track_id"],
            child_track_id=m["child_track_id"],
        )
        for m in data["camera_mappings"]
    ]

    session_blocks = [
        SessionBlock(
            name=b["name"],
            start_s=b["start_s"],
            end_s=b["end_s"],
            color=b.get("color", "gray"),
        )
        for b in data.get("session_blocks", [])
    ]

    return SessionConfig(
        session_id=data["session_id"],
        output_dir=data.get("output_dir", ""),
        image_width=data.get("image_width", 854),
        image_height=data.get("image_height", 480),
        frame_rate=data.get("frame_rate", 5.0),
        camera_mappings=camera_mappings,
        session_blocks=session_blocks,
    )


# ====================================================================
# Time-range filtering
# ====================================================================


def resolve_time_range(
    session_config: SessionConfig,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    time_block: Optional[str] = None,
) -> Tuple[float, float]:
    """Resolve CLI time-range arguments into a (start_s, end_s) pair.

    If ``time_block`` is provided, its start/end override ``start_s``/``end_s``.
    Otherwise falls back to the explicit values, using 0.0 / inf as defaults.

    Args:
        session_config: Session configuration (for block lookup).
        start_s: Explicit start time in seconds (None = session start).
        end_s: Explicit end time in seconds (None = session end).
        time_block: Named session block whose range to use.

    Returns:
        (start_s, end_s) tuple.

    Raises:
        KeyError: If time_block name is not found in session_config.
    """
    if time_block is not None:
        block = session_config.get_session_block(time_block)
        return (block.start_s, block.end_s)

    return (start_s if start_s is not None else 0.0,
            end_s if end_s is not None else float("inf"))


def filter_by_time_range(
    df: pd.DataFrame,
    start_s: float,
    end_s: float,
    time_column: str = "timestamp_s",
) -> pd.DataFrame:
    """Filter a DataFrame to rows within [start_s, end_s].

    Args:
        df: Input DataFrame.
        start_s: Start of time range (inclusive).
        end_s: End of time range (inclusive).
        time_column: Column containing timestamps.

    Returns:
        Filtered copy of the DataFrame.
    """
    if start_s <= 0.0 and end_s == float("inf"):
        return df
    mask = (df[time_column] >= start_s) & (df[time_column] <= end_s)
    return df.loc[mask].copy()


def filter_heatmaps_by_time_range(
    heatmaps: np.ndarray,
    keys: list[str],
    frame_to_timestamp: Dict[int, float],
    start_s: float,
    end_s: float,
) -> Tuple[np.ndarray, list[str]]:
    """Filter heatmaps array + keys by time range.

    Args:
        heatmaps: Array of shape [N, 64, 64].
        keys: List of keys like "f000001_t0".
        frame_to_timestamp: Mapping from frame_idx to timestamp_s.
        start_s: Start of time range (inclusive).
        end_s: End of time range (inclusive).

    Returns:
        (filtered_heatmaps, filtered_keys) tuple.
    """
    if start_s <= 0.0 and end_s == float("inf"):
        return heatmaps, keys

    keep_indices = []
    for i, key in enumerate(keys):
        frame_idx = int(key.split("_t")[0][1:])  # "f000001_t0" → 1
        ts = frame_to_timestamp.get(frame_idx, -1.0)
        if start_s <= ts <= end_s:
            keep_indices.append(i)

    if not keep_indices:
        return np.array([]).reshape(0, 64, 64), []

    return heatmaps[keep_indices], [keys[i] for i in keep_indices]


def add_time_range_args(parser: argparse.ArgumentParser) -> None:
    """Add shared --start-s, --end-s, --time-block arguments to a CLI parser.

    Args:
        parser: Argument parser to extend.
    """
    group = parser.add_argument_group("time range")
    group.add_argument(
        "--start-s",
        type=float,
        default=None,
        help="Start time in seconds (e.g. 600 for minute 10). Default: session start.",
    )
    group.add_argument(
        "--end-s",
        type=float,
        default=None,
        help="End time in seconds (e.g. 2226 for 37:06). Default: session end.",
    )
    group.add_argument(
        "--time-block",
        type=str,
        default=None,
        help="Use a named session block's time range instead of --start-s/--end-s. "
        "e.g. 'grocery_free_play', 'storybook_reading'.",
    )
