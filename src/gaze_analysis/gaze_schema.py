"""Output schema definitions for gaze analysis CSV files."""

from __future__ import annotations

GAZE_HEATMAP_COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "track_id",
    "gaze_peak_x",
    "gaze_peak_y",
    "gaze_peak_value",
    "inout_score",
    "head_source",
]
