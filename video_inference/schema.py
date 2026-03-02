"""
Schema contract and validators for video inference outputs.

This module defines the minimal public-facing contract for Phase 2 outputs:
- session manifest metadata
- 2D tracking rows for up to N people (N from manifest assumptions.max_persons)
- 3D pose rows for those tracked identities
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

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


def _to_numeric(
    series: pd.Series, column_name: str, context: str
) -> Tuple[pd.Series, List[str]]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return numeric, [
            f"{context}: column '{column_name}' contains non-numeric values"
        ]
    return numeric, []


def _validate_track_id_label_mapping(
    dataframe: pd.DataFrame, context: str
) -> List[str]:
    errors: List[str] = []
    label_map = dataframe.groupby("track_id")["track_label"].nunique(dropna=False)
    inconsistent = label_map[label_map != 1]
    if not inconsistent.empty:
        ids = ", ".join(str(track_id) for track_id in inconsistent.index.tolist())
        errors.append(f"{context}: inconsistent label mapping for track_id(s): {ids}")
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
    if not isinstance(max_persons, int) or max_persons < 1:
        errors.append("manifest: assumptions.max_persons must be an integer >= 1")

    enforce_exact = assumptions.get("enforce_exact_person_count", False)
    if not isinstance(enforce_exact, bool):
        errors.append(
            "manifest: assumptions.enforce_exact_person_count must be boolean"
        )

    return errors


def validate_tracks_2d(
    tracks_df: pd.DataFrame,
    max_persons: int = 2,
    enforce_exact_person_count: bool = True,
    expected_track_labels: Optional[Set[str]] = None,
) -> List[str]:
    """Validate 2D tracking table."""
    errors: List[str] = []
    context = "tracks_2d"

    errors.extend(
        _add_missing_column_errors(tracks_df, REQUIRED_TRACK_COLUMNS, context)
    )
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

    expected_track_ids = set(range(int(max_persons)))
    unique_ids = set(track_id.unique().tolist())
    if not unique_ids:
        errors.append(f"{context}: track_id set cannot be empty")
    if not unique_ids.issubset(expected_track_ids):
        errors.append(
            f"{context}: track_id values must be within {sorted(expected_track_ids)}, "
            f"found {sorted(unique_ids)}"
        )
    if enforce_exact_person_count and unique_ids != expected_track_ids:
        errors.append(
            f"{context}: expected full track_id set {sorted(expected_track_ids)}, "
            f"found {sorted(unique_ids)}"
        )

    if expected_track_labels is not None:
        unique_labels = set(labels.unique().tolist())
        if unique_labels != expected_track_labels:
            errors.append(
                f"{context}: expected track_label set {sorted(expected_track_labels)}, "
                f"found {sorted(unique_labels)}"
            )

    duplicate_rows = tracks_df.duplicated(subset=["frame_idx", "track_id"])
    if duplicate_rows.any():
        errors.append(f"{context}: duplicate rows for (frame_idx, track_id)")

    per_frame_counts = track_id.groupby(frame_idx).count()
    if enforce_exact_person_count:
        if (per_frame_counts != max_persons).any():
            bad_frames = sorted(
                per_frame_counts[per_frame_counts != max_persons]
                .index.astype(int)
                .tolist()
            )
            errors.append(
                f"{context}: each frame must contain exactly {max_persons} tracks; "
                f"bad frame_idx: {bad_frames}"
            )
    else:
        out_of_range = (per_frame_counts < 1) | (per_frame_counts > max_persons)
        if out_of_range.any():
            bad_frames = sorted(
                per_frame_counts[out_of_range].index.astype(int).tolist()
            )
            errors.append(
                f"{context}: each frame must contain between 1 and {max_persons} tracks; "
                f"bad frame_idx: {bad_frames}"
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


def validate_pose_3d(
    pose_df: pd.DataFrame,
    max_persons: int = 2,
    enforce_exact_person_count: bool = True,
    expected_track_labels: Optional[Set[str]] = None,
) -> List[str]:
    """Validate 3D pose table."""
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
        frame_errors
        + ts_errors
        + id_errors
        + x_errors
        + y_errors
        + z_errors
        + conf_errors
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

    finite_xyz = (
        np.isfinite(x_m.to_numpy())
        & np.isfinite(y_m.to_numpy())
        & np.isfinite(z_m.to_numpy())
    )
    if not finite_xyz.all():
        errors.append(f"{context}: x_m/y_m/z_m must be finite")

    expected_track_ids = set(range(int(max_persons)))
    unique_ids = set(track_id.unique().tolist())
    if not unique_ids:
        errors.append(f"{context}: track_id set cannot be empty")
    if not unique_ids.issubset(expected_track_ids):
        errors.append(
            f"{context}: track_id values must be within {sorted(expected_track_ids)}, "
            f"found {sorted(unique_ids)}"
        )
    if enforce_exact_person_count and unique_ids != expected_track_ids:
        errors.append(
            f"{context}: expected full track_id set {sorted(expected_track_ids)}, "
            f"found {sorted(unique_ids)}"
        )

    if expected_track_labels is not None:
        unique_labels = set(labels.unique().tolist())
        if unique_labels != expected_track_labels:
            errors.append(
                f"{context}: expected track_label set {sorted(expected_track_labels)}, "
                f"found {sorted(unique_labels)}"
            )

    duplicate_rows = pose_df.duplicated(
        subset=["frame_idx", "track_id", "keypoint_name"]
    )
    if duplicate_rows.any():
        errors.append(
            f"{context}: duplicate rows for (frame_idx, track_id, keypoint_name)"
        )

    rows_per_frame_track = pose_df.groupby(["frame_idx", "track_id"]).size()
    if (rows_per_frame_track < 1).any():
        errors.append(
            f"{context}: each (frame_idx, track_id) must have >=1 keypoint row"
        )

    per_frame_track_count = pose_df.groupby("frame_idx")["track_id"].nunique()
    if enforce_exact_person_count:
        if (per_frame_track_count != max_persons).any():
            bad_frames = sorted(
                per_frame_track_count[per_frame_track_count != max_persons]
                .index.astype(int)
                .tolist()
            )
            errors.append(
                f"{context}: each frame must contain exactly {max_persons} distinct track_id; "
                f"bad frame_idx: {bad_frames}"
            )
    else:
        out_of_range = (per_frame_track_count < 1) | (
            per_frame_track_count > max_persons
        )
        if out_of_range.any():
            bad_frames = sorted(
                per_frame_track_count[out_of_range].index.astype(int).tolist()
            )
            errors.append(
                f"{context}: each frame must contain between 1 and {max_persons} distinct "
                f"track_id; bad frame_idx: {bad_frames}"
            )

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
    assumptions = manifest.get("assumptions", {})
    max_persons_raw = assumptions.get("max_persons", 2)
    max_persons = int(max_persons_raw) if isinstance(max_persons_raw, int) else 2
    enforce_exact = assumptions.get("enforce_exact_person_count", False)
    enforce_exact = bool(enforce_exact) if isinstance(enforce_exact, bool) else False
    id_policy = str(assumptions.get("id_policy", ""))
    expected_labels = (
        {"parent", "child"}
        if max_persons == 2
        and "parent_larger_child_smaller_with_temporal_consistency" in id_policy
        else None
    )

    if not tracks_path.exists():
        errors.append(f"tracks_2d: file not found '{tracks_path.name}'")
    if not pose_path.exists():
        errors.append(f"pose_3d: file not found '{pose_path.name}'")

    tracks_df = None
    pose_df = None
    if tracks_path.exists():
        tracks_df = pd.read_csv(tracks_path)
        errors.extend(
            validate_tracks_2d(
                tracks_df,
                max_persons=max_persons,
                enforce_exact_person_count=enforce_exact,
                expected_track_labels=expected_labels,
            )
        )
    if pose_path.exists():
        pose_df = pd.read_csv(pose_path)
        errors.extend(
            validate_pose_3d(
                pose_df,
                max_persons=max_persons,
                enforce_exact_person_count=enforce_exact,
                expected_track_labels=expected_labels,
            )
        )

    if tracks_df is not None and pose_df is not None:
        track_ids_tracks = set(
            pd.to_numeric(tracks_df["track_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        track_ids_pose = set(
            pd.to_numeric(pose_df["track_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if track_ids_tracks != track_ids_pose:
            errors.append(
                "cross-check: track_id sets differ between tracks_2d and pose_3d"
            )

    return ValidationResult(is_valid=not errors, errors=errors)
