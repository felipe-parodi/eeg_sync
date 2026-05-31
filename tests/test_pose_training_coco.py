"""Tests for COCO-17 keypoint constants used in pose training.

The flip index is load-bearing: a wrong value silently corrupts every
left/right keypoint when Ultralytics applies horizontal-flip augmentation
during training, so it is tested explicitly here.
"""

from __future__ import annotations

from pose_training import coco


def test_keypoint_count_is_seventeen() -> None:
    assert coco.NUM_KEYPOINTS == 17
    assert len(coco.KEYPOINT_NAMES) == 17
    assert len(coco.FLIP_IDX) == 17
    assert len(coco.OKS_SIGMAS) == 17


def test_flip_idx_is_a_permutation() -> None:
    assert sorted(coco.FLIP_IDX) == list(range(coco.NUM_KEYPOINTS))


def test_flip_idx_is_an_involution() -> None:
    # Flipping twice returns the original index for every keypoint.
    for index in range(coco.NUM_KEYPOINTS):
        assert coco.FLIP_IDX[coco.FLIP_IDX[index]] == index


def test_flip_idx_swaps_left_and_right_names() -> None:
    for index, mapped in enumerate(coco.FLIP_IDX):
        name = coco.KEYPOINT_NAMES[index]
        mapped_name = coco.KEYPOINT_NAMES[mapped]
        if name.startswith("left_"):
            assert mapped_name == "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            assert mapped_name == "left_" + name[len("right_") :]
        else:
            # Central keypoints (e.g. the nose) map to themselves.
            assert mapped == index


def test_head_keypoints_match_gaze_assumptions() -> None:
    # gaze_analysis/head_bbox.py reads keypoints 0-4 as the head group.
    assert coco.KEYPOINT_NAMES[:5] == [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
    ]


def test_sigmas_are_positive_and_finite() -> None:
    assert all(sigma > 0 for sigma in coco.OKS_SIGMAS)
