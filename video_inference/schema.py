"""
Schema contract and validators for video inference outputs.

This module defines the minimal public-facing contract for Phase 2 outputs:
- session manifest metadata
- 2D tracking rows for exactly two people
- 3D pose rows for those two people
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

EXPECTED_TRACK_IDS = {0, 1}
EXPECTED_TRACK_LABELS = {"parent", "child"}

REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "session_id",
    "source_videos",
    "assumptions",
    "outputs",
}

REQUIRED_TRACK_COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "track_id",
    "track_label",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "track_confidence",
]

REQUIRED_POSE_COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "track_id",
    "track_label",
    "keypoint_name",
    "x_m",
    "y_m",
    "z_m",
    "keypoint_confidence",
]


@dataclass
class ValidationResult:
    """Container for schema validation results."""

    is_valid: bool
    errors: List[str]


def _add_missing_column_errors(
    dataframe: pd.DataFrame, required_columns: List[str], context: str
) -> List[str]:
    missing = [column for column in required_columns if column not in dataframe.columns]
    return [f"{context}: missing required column '{column}'" for column in missing]


def _to_numeric(series: pd.Series, column_name: str, context: str) -> Tuple[pd.Series, List[str]]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return numeric, [f"{context}: column '{column_name}' contains non-numeric values"]
    return numeric, []


def _validate_track_id_label_mapping(
    dataframe: pd.DataFrame, context: str
) -> List[str]:
    errors: List[str] = []
    label_map = dataframe.groupby("track_id")["track_label"].nunique(dropna=False)
    inconsistent = label_map[label_map != 1]
    if not inconsistent.empty:
        ids = ", ".join(str(track_id) for track_id in inconsistent.index.tolist())
        errors.append(
            f"{context}: inconsistent label mapping for track_id(s): {ids}"
        )
    return errors


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Validate required fields and top-level assumptions in manifest."""
    errors: List[str] = []

    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            errors.append(f"manifest: missing required key '{key}'")

    outputs = manifest.get("outputs", {})
    if "tracks_2d" not in outputs:
        errors.append("manifest: outputs.tracks_2d is required")
    if "pose_3d" not in outputs:
        errors.append("manifest: outputs.pose_3d is required")

    assumptions = manifest.get("assumptions", {})
    max_persons = assumptions.get("max_persons")
    if max_persons != 2:
        errors.append("manifest: assumptions.max_persons must be 2")

    return errors


def validate_tracks_2d(tracks_df: pd.DataFrame) -> List[str]:
    """Validate two-person 2D tracking table."""
    errors: List[str] = []
    context = "tracks_2d"

    errors.extend(_add_missing_column_errors(tracks_df, REQUIRED_TRACK_COLUMNS, context))
    if errors:
        return errors

    if tracks_df.empty:
        return [f"{context}: file is empty"]

    frame_idx, frame_errors = _to_numeric(tracks_df["frame_idx"], "frame_idx", context)
    timestamp_s, ts_errors = _to_numeric(
        tracks_df["timestamp_s"], "timestamp_s", context
    )
    track_id_raw, id_errors = _to_numeric(tracks_df["track_id"], "track_id", context)
    bbox_x1, bbox_x1_errors = _to_numeric(tracks_df["bbox_x1"], "bbox_x1", context)
    bbox_y1, bbox_y1_errors = _to_numeric(tracks_df["bbox_y1"], "bbox_y1", context)
    bbox_x2, bbox_x2_errors = _to_numeric(tracks_df["bbox_x2"], "bbox_x2", context)
    bbox_y2, bbox_y2_errors = _to_numeric(tracks_df["bbox_y2"], "bbox_y2", context)
    confidence, conf_errors = _to_numeric(
        tracks_df["track_confidence"], "track_confidence", context
    )

    errors.extend(
        frame_errors
        + ts_errors
        + id_errors
        + bbox_x1_errors
        + bbox_y1_errors
        + bbox_x2_errors
        + bbox_y2_errors
        + conf_errors
    )
    if errors:
        return errors

    track_id = track_id_raw.astype(int)
    labels = tracks_df["track_label"].astype(str)

    if (frame_idx < 0).any():
        errors.append(f"{context}: frame_idx must be non-negative")
    if (timestamp_s < 0).any():
        errors.append(f"{context}: timestamp_s must be non-negative")

    if ((confidence < 0) | (confidence > 1)).any():
        errors.append(f"{context}: track_confidence must be in [0, 1]")

    if ((bbox_x2 - bbox_x1) <= 0).any() or ((bbox_y2 - bbox_y1) <= 0).any():
        errors.append(f"{context}: bbox_x2/y2 must be greater than bbox_x1/y1")

    unique_ids = set(track_id.unique().tolist())
    if unique_ids != EXPECTED_TRACK_IDS:
        errors.append(
            f"{context}: expected track_id set {sorted(EXPECTED_TRACK_IDS)}, "
            f"found {sorted(unique_ids)}"
        )

    unique_labels = set(labels.unique().tolist())
    if unique_labels != EXPECTED_TRACK_LABELS:
        errors.append(
            f"{context}: expected track_label set {sorted(EXPECTED_TRACK_LABELS)}, "
            f"found {sorted(unique_labels)}"
        )

    duplicate_rows = tracks_df.duplicated(subset=["frame_idx", "track_id"])
    if duplicate_rows.any():
        errors.append(f"{context}: duplicate rows for (frame_idx, track_id)")

    per_frame_counts = track_id.groupby(frame_idx).count()
    if (per_frame_counts != 2).any():
        bad_frames = sorted(per_frame_counts[per_frame_counts != 2].index.astype(int).tolist())
        errors.append(
            f"{context}: each frame must contain exactly 2 tracks; bad frame_idx: {bad_frames}"
        )

    timestamp_by_frame = timestamp_s.groupby(frame_idx)
    timestamp_nunique = timestamp_by_frame.nunique(dropna=False)
    if (timestamp_nunique != 1).any():
        errors.append(f"{context}: timestamp_s must be consistent within each frame")

    frame_timestamp = timestamp_by_frame.first().sort_index()
    if not frame_timestamp.is_monotonic_increasing:
        errors.append(f"{context}: timestamp_s must increase with frame_idx")

    tracks_for_mapping = pd.DataFrame({"track_id": track_id, "track_label": labels})
    errors.extend(_validate_track_id_label_mapping(tracks_for_mapping, context))

    return errors


def validate_pose_3d(pose_df: pd.DataFrame) -> List[str]:
    """Validate two-person 3D pose table."""
    errors: List[str] = []
    context = "pose_3d"

    errors.extend(_add_missing_column_errors(pose_df, REQUIRED_POSE_COLUMNS, context))
    if errors:
        return errors

    if pose_df.empty:
        return [f"{context}: file is empty"]

    frame_idx, frame_errors = _to_numeric(pose_df["frame_idx"], "frame_idx", context)
    timestamp_s, ts_errors = _to_numeric(pose_df["timestamp_s"], "timestamp_s", context)
    track_id_raw, id_errors = _to_numeric(pose_df["track_id"], "track_id", context)
    x_m, x_errors = _to_numeric(pose_df["x_m"], "x_m", context)
    y_m, y_errors = _to_numeric(pose_df["y_m"], "y_m", context)
    z_m, z_errors = _to_numeric(pose_df["z_m"], "z_m", context)
    keypoint_confidence, conf_errors = _to_numeric(
        pose_df["keypoint_confidence"], "keypoint_confidence", context
    )

    errors.extend(
        frame_errors + ts_errors + id_errors + x_errors + y_errors + z_errors + conf_errors
    )
    if errors:
        return errors

    track_id = track_id_raw.astype(int)
    labels = pose_df["track_label"].astype(str)
    keypoint_name = pose_df["keypoint_name"].astype(str)

    if (frame_idx < 0).any():
        errors.append(f"{context}: frame_idx must be non-negative")
    if (timestamp_s < 0).any():
        errors.append(f"{context}: timestamp_s must be non-negative")
    if ((keypoint_confidence < 0) | (keypoint_confidence > 1)).any():
        errors.append(f"{context}: keypoint_confidence must be in [0, 1]")
    if keypoint_name.str.strip().eq("").any():
        errors.append(f"{context}: keypoint_name must be non-empty")

    finite_xyz = np.isfinite(x_m.to_numpy()) & np.isfinite(y_m.to_numpy()) & np.isfinite(
        z_m.to_numpy()
    )
    if not finite_xyz.all():
        errors.append(f"{context}: x_m/y_m/z_m must be finite")

    unique_ids = set(track_id.unique().tolist())
    if unique_ids != EXPECTED_TRACK_IDS:
        errors.append(
            f"{context}: expected track_id set {sorted(EXPECTED_TRACK_IDS)}, "
            f"found {sorted(unique_ids)}"
        )

    unique_labels = set(labels.unique().tolist())
    if unique_labels != EXPECTED_TRACK_LABELS:
        errors.append(
            f"{context}: expected track_label set {sorted(EXPECTED_TRACK_LABELS)}, "
            f"found {sorted(unique_labels)}"
        )

    duplicate_rows = pose_df.duplicated(subset=["frame_idx", "track_id", "keypoint_name"])
    if duplicate_rows.any():
        errors.append(f"{context}: duplicate rows for (frame_idx, track_id, keypoint_name)")

    rows_per_frame_track = track_id.groupby([frame_idx, track_id]).count()
    if (rows_per_frame_track < 1).any():
        errors.append(f"{context}: each (frame_idx, track_id) must have >=1 keypoint row")

    pose_for_mapping = pd.DataFrame({"track_id": track_id, "track_label": labels})
    errors.extend(_validate_track_id_label_mapping(pose_for_mapping, context))

    return errors


def validate_session_output(session_dir: Path | str) -> ValidationResult:
    """
    Validate a full inference output folder.

    Expected files:
    - manifest.json
    - tracks_2d CSV file as referenced by manifest.outputs.tracks_2d
    - pose_3d CSV file as referenced by manifest.outputs.pose_3d
    """
    errors: List[str] = []
    session_path = Path(session_dir)
    manifest_path = session_path / "manifest.json"

    if not manifest_path.exists():
        return ValidationResult(False, ["manifest: manifest.json not found"])

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return ValidationResult(False, [f"manifest: invalid JSON ({error})"])

    errors.extend(validate_manifest(manifest))

    outputs = manifest.get("outputs", {})
    tracks_path = session_path / str(outputs.get("tracks_2d", ""))
    pose_path = session_path / str(outputs.get("pose_3d", ""))

    if not tracks_path.exists():
        errors.append(f"tracks_2d: file not found '{tracks_path.name}'")
    if not pose_path.exists():
        errors.append(f"pose_3d: file not found '{pose_path.name}'")

    tracks_df = None
    pose_df = None
    if tracks_path.exists():
        tracks_df = pd.read_csv(tracks_path)
        errors.extend(validate_tracks_2d(tracks_df))
    if pose_path.exists():
        pose_df = pd.read_csv(pose_path)
        errors.extend(validate_pose_3d(pose_df))

    if tracks_df is not None and pose_df is not None:
        track_ids_tracks = set(pd.to_numeric(tracks_df["track_id"], errors="coerce").dropna().astype(int).unique().tolist())
        track_ids_pose = set(pd.to_numeric(pose_df["track_id"], errors="coerce").dropna().astype(int).unique().tolist())
        if track_ids_tracks != track_ids_pose:
            errors.append(
                "cross-check: track_id sets differ between tracks_2d and pose_3d"
            )

    return ValidationResult(is_valid=not errors, errors=errors)
