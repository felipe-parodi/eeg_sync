"""Tests for fusing yolo11x-pose and ViTPose pseudo-labels."""

from __future__ import annotations

from pose_training import fuse_labels


def _cfg(**kw) -> fuse_labels.FuseConfig:
    return fuse_labels.FuseConfig(**kw)


def test_fuse_keypoint_agreement_averages_and_boosts() -> None:
    cfg = _cfg(agree_frac=0.05, agree_boost=1.1, disagree_penalty=0.7)
    out = fuse_labels.fuse_keypoint([10.0, 10.0, 0.8], [12.0, 10.0, 0.8], 100.0, cfg)
    assert abs(out[0] - 11.0) < 1e-6  # averaged x
    assert out[2] > 0.8  # confidence boosted on agreement


def test_fuse_keypoint_disagreement_keeps_vitpose_and_penalizes() -> None:
    cfg = _cfg(agree_frac=0.05, disagree_penalty=0.5)
    out = fuse_labels.fuse_keypoint([10.0, 10.0, 0.9], [80.0, 80.0, 0.85], 100.0, cfg)
    assert out[0] == 10.0 and out[1] == 10.0  # ViTPose position kept
    assert abs(out[2] - 0.45) < 1e-6  # 0.9 * 0.5 penalty


def test_fuse_keypoint_falls_back_when_one_unconfident() -> None:
    cfg = _cfg(conf_floor=0.3)
    # ViTPose unconfident, x-pose confident -> use x-pose
    out = fuse_labels.fuse_keypoint([10.0, 10.0, 0.1], [50.0, 50.0, 0.9], 100.0, cfg)
    assert out[0] == 50.0 and out[2] == 0.9


def _person(box, conf, kx):
    return {
        "bbox_xyxy": list(box),
        "box_conf": conf,
        "keypoints": [[kx, kx, 0.9] for _ in range(17)],
    }


def test_fuse_frame_matches_and_caps_to_four() -> None:
    cfg = _cfg(match_iou=0.5, max_persons=4)
    xpose = [_person([0, 0, 10, 20], 0.9, 5.0)]
    vit = [_person([0, 0, 10, 20], 0.95, 5.0)]  # same box -> matched
    fused = fuse_labels.fuse_frame(xpose, vit, cfg)
    assert len(fused) == 1
    assert fused[0]["box_conf"] == 0.95  # max of the two


def test_fuse_frame_keeps_unmatched_from_both() -> None:
    cfg = _cfg(match_iou=0.5, max_persons=4)
    xpose = [_person([0, 0, 10, 20], 0.9, 5.0)]
    vit = [_person([100, 100, 120, 140], 0.8, 110.0)]  # far -> unmatched
    fused = fuse_labels.fuse_frame(xpose, vit, cfg)
    assert len(fused) == 2


def test_fuse_frame_caps_persons() -> None:
    cfg = _cfg(match_iou=0.5, max_persons=4)
    xpose = [
        _person([i * 50, 0, i * 50 + 10, 20], 0.5 + i * 0.05, 5.0) for i in range(6)
    ]
    vit = []
    fused = fuse_labels.fuse_frame(xpose, vit, cfg)
    assert len(fused) == 4  # capped
    assert all(fused[i]["box_conf"] >= fused[i + 1]["box_conf"] for i in range(3))
