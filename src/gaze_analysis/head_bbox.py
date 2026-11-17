"""Extract head bounding boxes from COCO keypoints for Gazelle input."""

from __future__ import annotations

from typing import List

import pandas as pd

# COCO keypoints 0-4 are head region: nose, left eye, right eye, left ear, right ear
COCO_HEAD_KEYPOINTS = ["kp_000", "kp_001", "kp_002", "kp_003", "kp_004"]

# Minimum number of high-confidence head keypoints to compute bbox from keypoints
_MIN_HEAD_KP = 2

# Padding as a fraction of the max(width, height) of the keypoint extent
_PADDING_FRAC = 0.20

# Fallback: use the top N% of the body bounding box as the head region
_BODY_BBOX_HEAD_FRAC = 0.30

# Minimum head keypoint confidence for inclusion
_DEFAULT_KP_CONF = 0.3


def _head_bbox_from_keypoints(
    head_kps: pd.DataFrame,
    conf_threshold: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Compute a normalized head bbox from high-confidence head keypoints.

    Args:
        head_kps: Rows from pose_3d filtered to head keypoints for one person
            in one frame.
        conf_threshold: Minimum keypoint confidence for inclusion.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.

    Returns:
        (x1, y1, x2, y2) normalized to [0, 1], or None if too few keypoints
        pass the confidence threshold.
    """
    good = head_kps[head_kps["keypoint_confidence"] >= conf_threshold]
    if len(good) < _MIN_HEAD_KP:
        return None

    xs = good["x_m"].values.astype(float)
    ys = good["y_m"].values.astype(float)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    width = x_max - x_min
    height = y_max - y_min
    pad = _PADDING_FRAC * max(width, height, 1.0)

    x1 = max(0.0, (x_min - pad) / image_width)
    y1 = max(0.0, (y_min - pad) / image_height)
    x2 = min(1.0, (x_max + pad) / image_width)
    y2 = min(1.0, (y_max + pad) / image_height)

    return (x1, y1, x2, y2)


def _head_bbox_from_body_bbox(
    track_row: pd.Series,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Derive a head bbox from the top fraction of the body bounding box.

    Args:
        track_row: A single row from tracks_2d with bbox_x1..bbox_y2.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.

    Returns:
        (x1, y1, x2, y2) normalized to [0, 1].
    """
    bx1 = float(track_row["bbox_x1"])
    by1 = float(track_row["bbox_y1"])
    bx2 = float(track_row["bbox_x2"])
    by2 = float(track_row["bbox_y2"])

    body_height = by2 - by1
    head_y2 = by1 + _BODY_BBOX_HEAD_FRAC * body_height

    x1 = max(0.0, bx1 / image_width)
    y1 = max(0.0, by1 / image_height)
    x2 = min(1.0, bx2 / image_width)
    y2 = min(1.0, head_y2 / image_height)

    return (x1, y1, x2, y2)


def extract_head_bboxes(
    pose_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    image_width: int = 854,
    image_height: int = 480,
    kp_conf_threshold: float = _DEFAULT_KP_CONF,
) -> pd.DataFrame:
    """Extract normalized head bounding boxes for every person in every frame.

    Primary path: uses COCO head keypoints (kp_000-kp_004) when at least 2
    pass the confidence threshold. Adds 20% padding around the keypoint extent.

    Fallback: if too few head keypoints are confident, uses the top 30% of the
    person's body bounding box from tracks_2d.

    All coordinates are normalized to [0, 1] for Gazelle input.

    Args:
        pose_df: DataFrame with pose_3d schema (frame_idx, track_id,
            keypoint_name, x_m, y_m, keypoint_confidence, ...).
        tracks_df: DataFrame with tracks_2d schema (frame_idx, track_id,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, ...).
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        kp_conf_threshold: Minimum keypoint confidence for head keypoints.

    Returns:
        DataFrame with columns: frame_idx, timestamp_s, track_id,
        head_x1, head_y1, head_x2, head_y2, head_source.
    """
    if pose_df.empty:
        return pd.DataFrame(
            columns=[
                "frame_idx",
                "timestamp_s",
                "track_id",
                "head_x1",
                "head_y1",
                "head_x2",
                "head_y2",
                "head_source",
            ]
        )

    rows: List[dict] = []

    for (frame_idx, track_id), group in pose_df.groupby(
        ["frame_idx", "track_id"], sort=False
    ):
        timestamp_s = float(group["timestamp_s"].iloc[0])
        head_kps = group[group["keypoint_name"].isin(COCO_HEAD_KEYPOINTS)]

        bbox = _head_bbox_from_keypoints(
            head_kps, kp_conf_threshold, image_width, image_height
        )
        source = "keypoints"

        if bbox is None:
            # Fallback to body bbox
            track_match = tracks_df[
                (tracks_df["frame_idx"] == frame_idx)
                & (tracks_df["track_id"] == track_id)
            ]
            if track_match.empty:
                continue
            bbox = _head_bbox_from_body_bbox(
                track_match.iloc[0], image_width, image_height
            )
            source = "body_bbox_fallback"

        rows.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "track_id": track_id,
                "head_x1": bbox[0],
                "head_y1": bbox[1],
                "head_x2": bbox[2],
                "head_y2": bbox[3],
                "head_source": source,
            }
        )

    return pd.DataFrame(rows)
