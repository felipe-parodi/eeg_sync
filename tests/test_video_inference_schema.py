import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_inference.schema import (  # noqa: E402
    validate_manifest,
    validate_pose_3d,
    validate_session_output,
    validate_tracks_2d,
)


def _mock_session_dir() -> Path:
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "video_inference"
        / "mock_session"
    )


def test_mock_session_fixture_is_valid():
    result = validate_session_output(_mock_session_dir())
    assert result.is_valid, f"Expected valid fixture, got errors: {result.errors}"


def test_manifest_requires_positive_person_count():
    manifest_path = _mock_session_dir() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assumptions"]["max_persons"] = 0

    errors = validate_manifest(manifest)

    assert any("max_persons must be an integer >= 1" in error for error in errors)


def test_tracks_2d_rejects_frames_with_missing_track():
    tracks_df = pd.read_csv(_mock_session_dir() / "tracks_2d.csv")
    broken = tracks_df[~((tracks_df["frame_idx"] == 1) & (tracks_df["track_id"] == 1))]

    errors = validate_tracks_2d(broken, max_persons=2, enforce_exact_person_count=True)

    assert any("exactly 2 tracks" in error for error in errors)


def test_tracks_2d_rejects_inconsistent_track_label_mapping():
    tracks_df = pd.read_csv(_mock_session_dir() / "tracks_2d.csv")
    broken = tracks_df.copy()
    broken.loc[
        (broken["frame_idx"] == 2) & (broken["track_id"] == 0), "track_label"
    ] = "child"

    errors = validate_tracks_2d(broken)

    assert any("inconsistent label mapping" in error for error in errors)


def test_pose_3d_rejects_duplicate_keypoint_rows():
    pose_df = pd.read_csv(_mock_session_dir() / "pose_3d.csv")
    duplicate = pose_df.iloc[[0]].copy()
    broken = pd.concat([pose_df, duplicate], ignore_index=True)

    errors = validate_pose_3d(broken)

    assert any("duplicate rows" in error for error in errors)


def test_tracks_2d_accepts_four_people_when_configured():
    tracks_df = pd.DataFrame(
        [
            {
                "frame_idx": 0,
                "timestamp_s": 0.0,
                "track_id": track_id,
                "track_label": f"person_{track_id:02d}",
                "bbox_x1": 10.0 + track_id,
                "bbox_y1": 20.0 + track_id,
                "bbox_x2": 40.0 + track_id,
                "bbox_y2": 80.0 + track_id,
                "track_confidence": 0.9,
            }
            for track_id in range(4)
        ]
    )

    errors = validate_tracks_2d(
        tracks_df,
        max_persons=4,
        enforce_exact_person_count=True,
    )
    assert errors == []
