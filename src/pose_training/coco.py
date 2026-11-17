"""COCO-17 keypoint constants shared across the pose-training pipeline.

Single source of truth for the keypoint names, the horizontal-flip index, the
skeleton edges, and the OKS sigmas. Keeping these in one place avoids drift
between the dataset export (which writes ``flip_idx`` into ``data.yaml``) and
the evaluation code (which needs the OKS sigmas).
"""

from __future__ import annotations

from typing import List, Tuple

NUM_KEYPOINTS = 17

# Canonical COCO-17 ordering. gaze_analysis/head_bbox.py relies on indices 0-4
# being the head group (nose, eyes, ears).
KEYPOINT_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# When an image is mirrored horizontally, each keypoint takes the position of
# its left/right counterpart. Central keypoints (the nose) map to themselves.
# Ultralytics reads this from data.yaml as ``flip_idx`` and applies it during
# fliplr augmentation; a wrong value corrupts every left/right joint silently.
FLIP_IDX: List[int] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# COCO-17 skeleton edges (0-indexed), matching video_analysis.visualize_pose_tracks.
SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# Standard COCO OKS per-keypoint sigmas, used for object-keypoint-similarity
# during evaluation (Phase 5).
OKS_SIGMAS: List[float] = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
]


def validate_flip_idx() -> None:
    """Raise if FLIP_IDX is not a valid involution permutation of the keypoints.

    Raises:
        ValueError: If FLIP_IDX is not a permutation of range(NUM_KEYPOINTS)
            or is not its own inverse.
    """
    if sorted(FLIP_IDX) != list(range(NUM_KEYPOINTS)):
        raise ValueError("FLIP_IDX must be a permutation of range(NUM_KEYPOINTS)")
    for index, mapped in enumerate(FLIP_IDX):
        if FLIP_IDX[mapped] != index:
            raise ValueError("FLIP_IDX must be its own inverse (an involution)")
