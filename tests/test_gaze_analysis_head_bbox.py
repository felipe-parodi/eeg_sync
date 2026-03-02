"""Tests for gaze_analysis.head_bbox — COCO keypoint to head bbox conversion."""

from __future__ import annotations

import pandas as pd

from gaze_analysis.head_bbox import (
    extract_head_bboxes,
)

# COCO keypoint indices for head: nose=0, left_eye=1, right_eye=2, left_ear=3, right_ear=4
# Pixel coords in 854x480 image space.

_IMAGE_W = 854
_IMAGE_H = 480


def _make_pose_df(
    keypoints: dict[str, tuple[float, float, float]],
    track_id: int = 0,
    frame_idx: int = 0,
    timestamp_s: float = 0.0,
) -> pd.DataFrame:
    """Build a pose_3d-shaped DataFrame from a dict of {kp_name: (x, y, conf)}.

    Returns one row per keypoint, all for the same frame and track.
    """
    rows = []
    for kp_name, (x, y, conf) in keypoints.items():
        rows.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "track_id": track_id,
                "track_label": f"person_{track_id:02d}",
                "keypoint_name": kp_name,
                "x_m": x,
                "y_m": y,
                "z_m": 0.0,
                "keypoint_confidence": conf,
            }
        )
    return pd.DataFrame(rows)


def _make_tracks_df(
    track_id: int = 0,
    bbox: tuple[float, float, float, float] = (100, 50, 300, 400),
    frame_idx: int = 0,
    timestamp_s: float = 0.0,
) -> pd.DataFrame:
    """Build a tracks_2d-shaped DataFrame for one person in one frame."""
    return pd.DataFrame(
        [
            {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "track_id": track_id,
                "track_label": f"person_{track_id:02d}",
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "track_confidence": 0.9,
            }
        ]
    )


# --- Primary path: head keypoints with sufficient confidence ---


def test_head_bbox_from_keypoints_with_padding() -> None:
    """Head bbox should contain all keypoints plus 20% padding."""
    keypoints = {
        "kp_000": (200.0, 100.0, 0.9),  # nose
        "kp_001": (190.0, 90.0, 0.85),  # left eye
        "kp_002": (210.0, 90.0, 0.88),  # right eye
        "kp_003": (175.0, 100.0, 0.7),  # left ear
        "kp_004": (225.0, 100.0, 0.75),  # right ear
        # Body keypoints (should be ignored for head bbox)
        "kp_005": (200.0, 200.0, 0.9),  # left shoulder
        "kp_006": (200.0, 200.0, 0.9),  # right shoulder
    }
    pose_df = _make_pose_df(keypoints, track_id=0)
    tracks_df = _make_tracks_df(track_id=0)

    result = extract_head_bboxes(
        pose_df, tracks_df, image_width=_IMAGE_W, image_height=_IMAGE_H
    )

    assert len(result) == 1
    row = result.iloc[0]

    # Raw keypoint extent: x=[175, 225], y=[90, 100]
    # Width=50, Height=10 → padding = 20% of max(50, 10) = 10px
    # Padded: x=[165, 235], y=[80, 110]
    # Normalized: [165/854, 80/480, 235/854, 110/480]
    assert row["head_x1"] < 175.0 / _IMAGE_W
    assert row["head_y1"] < 90.0 / _IMAGE_H
    assert row["head_x2"] > 225.0 / _IMAGE_W
    assert row["head_y2"] > 100.0 / _IMAGE_H

    # Must be in [0, 1]
    for col in ["head_x1", "head_y1", "head_x2", "head_y2"]:
        assert 0.0 <= row[col] <= 1.0, f"{col}={row[col]} out of [0,1]"


def test_head_bbox_output_columns() -> None:
    """Output should contain expected columns."""
    keypoints = {
        "kp_000": (200.0, 100.0, 0.9),
        "kp_001": (190.0, 90.0, 0.85),
        "kp_002": (210.0, 90.0, 0.88),
    }
    pose_df = _make_pose_df(keypoints)
    tracks_df = _make_tracks_df()

    result = extract_head_bboxes(
        pose_df, tracks_df, image_width=_IMAGE_W, image_height=_IMAGE_H
    )
    expected_cols = {
        "frame_idx",
        "timestamp_s",
        "track_id",
        "head_x1",
        "head_y1",
        "head_x2",
        "head_y2",
        "head_source",
    }
    assert expected_cols.issubset(set(result.columns))


# --- Fallback: low confidence keypoints → use body bbox top 30% ---


def test_head_bbox_fallback_to_track_bbox() -> None:
    """When head keypoints are low confidence, use top 30% of body bbox."""
    keypoints = {
        "kp_000": (200.0, 100.0, 0.1),  # too low
        "kp_001": (190.0, 90.0, 0.05),  # too low
        "kp_002": (210.0, 90.0, 0.08),  # too low
        "kp_005": (200.0, 200.0, 0.9),  # shoulders (not head)
    }
    body_bbox = (100.0, 50.0, 300.0, 400.0)
    pose_df = _make_pose_df(keypoints, track_id=0)
    tracks_df = _make_tracks_df(track_id=0, bbox=body_bbox)

    result = extract_head_bboxes(
        pose_df, tracks_df, image_width=_IMAGE_W, image_height=_IMAGE_H
    )

    row = result.iloc[0]
    assert row["head_source"] == "body_bbox_fallback"

    # Top 30% of body bbox: y range [50, 400], height=350
    # Top 30% → y1=50, y2=50 + 0.3*350 = 155
    # x stays same: [100, 300]
    # Normalized: [100/854, 50/480, 300/854, 155/480]
    expected_y2_norm = 155.0 / _IMAGE_H
    assert abs(row["head_y2"] - expected_y2_norm) < 0.01


# --- Multiple people, multiple frames ---


def test_head_bbox_multiple_people() -> None:
    """Extract head bboxes for two people in one frame."""
    kp_parent = {
        "kp_000": (200.0, 100.0, 0.9),
        "kp_001": (190.0, 90.0, 0.85),
        "kp_002": (210.0, 90.0, 0.88),
    }
    kp_child = {
        "kp_000": (500.0, 120.0, 0.9),
        "kp_001": (490.0, 110.0, 0.85),
        "kp_002": (510.0, 110.0, 0.88),
    }
    pose_df = pd.concat(
        [
            _make_pose_df(kp_parent, track_id=0),
            _make_pose_df(kp_child, track_id=1),
        ],
        ignore_index=True,
    )
    tracks_df = pd.concat(
        [
            _make_tracks_df(track_id=0, bbox=(100, 50, 300, 400)),
            _make_tracks_df(track_id=1, bbox=(400, 50, 600, 400)),
        ],
        ignore_index=True,
    )

    result = extract_head_bboxes(
        pose_df, tracks_df, image_width=_IMAGE_W, image_height=_IMAGE_H
    )
    assert len(result) == 2
    assert set(result["track_id"].values) == {0, 1}


def test_head_bbox_multiple_frames() -> None:
    """Extract head bboxes across two frames."""
    frames = []
    tracks = []
    for frame_idx in range(2):
        kp = {
            "kp_000": (200.0 + frame_idx * 10, 100.0, 0.9),
            "kp_001": (190.0, 90.0, 0.85),
            "kp_002": (210.0, 90.0, 0.88),
        }
        frames.append(
            _make_pose_df(
                kp, track_id=0, frame_idx=frame_idx, timestamp_s=frame_idx * 0.2
            )
        )
        tracks.append(
            _make_tracks_df(
                track_id=0, frame_idx=frame_idx, timestamp_s=frame_idx * 0.2
            )
        )

    pose_df = pd.concat(frames, ignore_index=True)
    tracks_df = pd.concat(tracks, ignore_index=True)

    result = extract_head_bboxes(
        pose_df, tracks_df, image_width=_IMAGE_W, image_height=_IMAGE_H
    )
    assert len(result) == 2
    assert list(result["frame_idx"].values) == [0, 1]


# --- Edge cases ---


def test_head_bbox_clamps_to_zero_one() -> None:
    """Head bbox coordinates should be clamped to [0, 1]."""
    # Keypoints near image edge — padding would push outside
    keypoints = {
        "kp_000": (5.0, 5.0, 0.9),
        "kp_001": (2.0, 2.0, 0.85),
        "kp_002": (8.0, 2.0, 0.88),
    }
    pose_df = _make_pose_df(keypoints)
    tracks_df = _make_tracks_df(bbox=(0, 0, 50, 50))

    result = extract_head_bboxes(
        pose_df, tracks_df, image_width=_IMAGE_W, image_height=_IMAGE_H
    )
    row = result.iloc[0]
    assert row["head_x1"] >= 0.0
    assert row["head_y1"] >= 0.0


def test_empty_pose_returns_empty() -> None:
    """Empty input DataFrames should return an empty result."""
    pose_df = pd.DataFrame(
        columns=[
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
    )
    tracks_df = pd.DataFrame(
        columns=[
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
    )
    result = extract_head_bboxes(
        pose_df, tracks_df, image_width=_IMAGE_W, image_height=_IMAGE_H
    )
    assert len(result) == 0
