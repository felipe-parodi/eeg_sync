"""Tests for geometric keypoint sanity cleaning (anatomy + cross-instance)."""

from __future__ import annotations

from pose_training import pose_sanity


def _kpts(overrides=None):
    # Default: a plausible upright person. Shoulders at (90,100)/(110,100),
    # hips at (92,160)/(108,160), nose just above shoulders.
    pts = [[100.0, 80.0, 0.9]] + [[0.0, 0.0, 0.0]] * 16
    pts[5] = [90.0, 100.0, 0.9]  # left_shoulder
    pts[6] = [110.0, 100.0, 0.9]  # right_shoulder
    pts[11] = [92.0, 160.0, 0.9]  # left_hip
    pts[12] = [108.0, 160.0, 0.9]  # right_hip
    if overrides:
        for k, v in overrides.items():
            pts[k] = v
    return pts


def _person(kpts, box=(80.0, 70.0, 120.0, 170.0), conf=0.9):
    return {"bbox_xyxy": list(box), "box_conf": conf, "keypoints": kpts}


def test_body_scale_and_center_from_torso() -> None:
    scale, center = pose_sanity.body_scale_and_center(_kpts(), 0.3)
    assert scale is not None and scale > 0
    assert abs(center[0] - 100.0) < 5.0  # roughly torso-centered


def test_demote_offbody_keypoint() -> None:
    cfg = pose_sanity.CleanConfig()
    # Nose teleported far from the body (stolen from another person). Box is
    # small so the box-floored scale still flags the nose as off-body.
    kpts = _kpts({0: [400.0, 90.0, 0.9]})
    person = pose_sanity.demote_offbody_keypoints(_person(kpts), cfg)
    assert person["keypoints"][0][2] == 0.0  # nose demoted
    assert person["keypoints"][5][2] == 0.9  # shoulder kept


def test_keeps_plausible_keypoints() -> None:
    cfg = pose_sanity.CleanConfig()
    person = pose_sanity.demote_offbody_keypoints(_person(_kpts()), cfg)
    assert person["keypoints"][0][2] == 0.9  # plausible nose kept


def test_offbody_keeps_far_limbs() -> None:
    cfg = pose_sanity.CleanConfig()
    # A legitimately far ankle on a tall thin box must NOT be demoted, even
    # when only a couple of torso keypoints anchor the scale.
    kpts = _kpts({15: [95.0, 300.0, 0.9]})  # left ankle far below torso
    person = _person(kpts, box=(80.0, 70.0, 120.0, 320.0))
    cleaned = pose_sanity.demote_offbody_keypoints(person, cfg)
    assert cleaned["keypoints"][15][2] == 0.9  # far limb kept


def test_cross_instance_conflict_demotes_far_owner() -> None:
    cfg = pose_sanity.CleanConfig()
    a = _person(_kpts())  # nose near (100,80), body around x=100
    # Person B sits far away (x ~ 400) but its nose was stolen onto A's nose.
    b_kpts = _kpts(
        {
            0: [101.0, 80.0, 0.9],  # stolen: at A's nose
            5: [390.0, 100.0, 0.9],
            6: [410.0, 100.0, 0.9],
            11: [392.0, 160.0, 0.9],
            12: [408.0, 160.0, 0.9],
        }
    )
    b = _person(b_kpts, box=(380.0, 70.0, 420.0, 170.0))
    cleaned = pose_sanity.resolve_cross_instance_conflicts([a, b], cfg)
    # A owns the nose (its body is nearest); B's stolen nose is demoted.
    assert cleaned[0]["keypoints"][0][2] == 0.9
    assert cleaned[1]["keypoints"][0][2] == 0.0


def test_tighten_box_shrinks_to_keypoints() -> None:
    cfg = pose_sanity.CleanConfig()
    person = _person(_kpts(), box=(0.0, 0.0, 500.0, 500.0))
    tight = pose_sanity.tighten_box(person, cfg, image_w=500, image_h=500)
    x1, y1, x2, y2 = tight["bbox_xyxy"]
    assert x1 > 0 and y1 > 0 and x2 < 500 and y2 < 500  # shrunk inward
    # keypoints all inside the new box
    for x, y, c in tight["keypoints"]:
        if c > 0:
            assert x1 - 1 <= x <= x2 + 1 and y1 - 1 <= y <= y2 + 1


def test_demote_offimage_keypoint() -> None:
    cfg = pose_sanity.CleanConfig()
    kpts = _kpts({9: [900.0, 90.0, 0.9]})  # left_wrist way off a 200px image
    person = pose_sanity.demote_offimage_keypoints(_person(kpts), 200, 200, cfg)
    assert person["keypoints"][9][2] == 0.0


def test_demote_asymmetric_limb() -> None:
    cfg = pose_sanity.CleanConfig(max_asym_ratio=3.0)
    # Left forearm normal (7->9 short), right forearm absurdly long (8->10).
    kpts = _kpts(
        {
            7: [85.0, 120.0, 0.9],
            9: [85.0, 140.0, 0.9],  # left forearm length 20
            8: [115.0, 120.0, 0.9],
            10: [115.0, 320.0, 0.9],  # right forearm length 200 -> ratio 10
        }
    )
    person = pose_sanity.demote_asymmetric_limbs(_person(kpts), cfg)
    assert person["keypoints"][10][2] == 0.0  # long distal joint demoted
    assert person["keypoints"][9][2] == 0.9  # normal side kept


def test_pose_nms_drops_duplicate_instance() -> None:
    cfg = pose_sanity.CleanConfig()
    a = _person(_kpts(), conf=0.95)
    b = _person(_kpts(), conf=0.6)  # identical keypoints -> duplicate
    kept = pose_sanity.pose_nms([a, b], cfg)
    assert len(kept) == 1
    assert kept[0]["box_conf"] == 0.95


def test_pose_nms_keeps_distinct_instances() -> None:
    cfg = pose_sanity.CleanConfig()
    a = _person(_kpts())
    far = _kpts(
        {k: [v[0] + 400.0, v[1], v[2]] for k, v in enumerate(_kpts()) if v[2] > 0}
    )
    b = _person(far, box=(480.0, 70.0, 520.0, 170.0))
    kept = pose_sanity.pose_nms([a, b], cfg)
    assert len(kept) == 2


def test_clean_frame_runs_end_to_end() -> None:
    cfg = pose_sanity.CleanConfig()
    a = _person(_kpts())
    # A genuinely distinct second person (shifted body) whose nose was stolen
    # onto A's nose. They should stay two instances, with B's nose demoted.
    b = _person(
        _kpts(
            {
                0: [101.0, 80.0, 0.9],
                5: [390.0, 100.0, 0.9],
                6: [410.0, 100.0, 0.9],
                11: [392.0, 160.0, 0.9],
                12: [408.0, 160.0, 0.9],
            }
        ),
        box=(380.0, 70.0, 420.0, 170.0),
    )
    cleaned = pose_sanity.clean_frame([a, b], image_w=500, image_h=500, cfg=cfg)
    assert len(cleaned) == 2
