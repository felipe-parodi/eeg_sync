"""Tests for 3-way teacher fusion (box matching + per-keypoint majority vote)."""

from __future__ import annotations

from pose_training import fuse_multi as fm


def _p(box, kx, conf=0.9):
    return {
        "bbox_xyxy": list(box),
        "box_conf": conf,
        "keypoints": [[float(kx), float(kx), conf] for _ in range(17)],
    }


def test_cluster_three_teachers_same_person() -> None:
    a = [_p([0, 0, 100, 200], 50)]
    b = [_p([2, 2, 102, 202], 51)]
    c = [_p([1, 1, 101, 201], 49)]
    clusters = fm.cluster_persons([a, b, c], match_iou=0.5)
    assert len(clusters) == 1
    assert sum(1 for m in clusters[0] if m is not None) == 3


def test_cluster_distinct_people() -> None:
    a = [_p([0, 0, 50, 100], 25)]
    b = [_p([200, 0, 250, 100], 225)]
    clusters = fm.cluster_persons([a, b], match_iou=0.5)
    assert len(clusters) == 2


def test_cluster_one_box_per_teacher() -> None:
    # A teacher with two overlapping boxes must not both land in one cluster.
    a = [_p([0, 0, 100, 200], 50), _p([1, 1, 101, 201], 50)]
    clusters = fm.cluster_persons([a], match_iou=0.5)
    assert len(clusters) == 2  # same teacher -> separate clusters


def test_vote_keypoint_majority_drops_outlier() -> None:
    cfg = fm.FuseMultiConfig()
    out = fm.vote_keypoint(
        [[10.0, 10.0, 0.8], [11.0, 10.0, 0.8], [90.0, 90.0, 0.7]], 5.0, cfg
    )
    assert abs(out[0] - 10.5) < 1.5  # consensus of the two that agree
    assert out[2] > 0.0


def test_vote_keypoint_single_is_penalized() -> None:
    cfg = fm.FuseMultiConfig(single_vote_conf=0.6)
    out = fm.vote_keypoint([[10.0, 10.0, 0.8]], 5.0, cfg)
    assert abs(out[2] - 0.8 * 0.6) < 1e-6


def test_vote_keypoint_empty_is_zero() -> None:
    cfg = fm.FuseMultiConfig()
    assert fm.vote_keypoint([], 5.0, cfg) == [0.0, 0.0, 0.0]


def test_fuse_cluster_outputs_17_keypoints() -> None:
    cfg = fm.FuseMultiConfig()
    cluster = [_p([0, 0, 100, 200], 50), _p([0, 0, 100, 200], 51), None]
    fused = fm.fuse_cluster(cluster, cfg)
    assert len(fused["keypoints"]) == 17
    assert len(fused["bbox_xyxy"]) == 4
