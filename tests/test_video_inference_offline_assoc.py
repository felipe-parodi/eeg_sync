"""Tests for offline two-identity association (video_inference.offline_assoc)."""

from __future__ import annotations

import numpy as np

from video_inference.offline_assoc import (
    AnchorBoxes,
    AssocConfig,
    FrameDetections,
    associate_two_people,
    build_tracklets,
    dedup_frame,
    keypoint_similarity,
)


def _person_kpts(x: float, y: float, h: float) -> np.ndarray:
    """A crude upright 17-keypoint skeleton inside box (x, y, x+h/2, y+h)."""
    w = h / 2
    rel = np.array(
        [
            [0.5, 0.1],
            [0.45, 0.08],
            [0.55, 0.08],
            [0.4, 0.1],
            [0.6, 0.1],
            [0.3, 0.25],
            [0.7, 0.25],
            [0.25, 0.4],
            [0.75, 0.4],
            [0.2, 0.55],
            [0.8, 0.55],
            [0.35, 0.55],
            [0.65, 0.55],
            [0.35, 0.75],
            [0.65, 0.75],
            [0.35, 0.95],
            [0.65, 0.95],
        ]
    )
    out = np.ones((17, 3), np.float32)
    out[:, 0] = x + rel[:, 0] * w
    out[:, 1] = y + rel[:, 1] * h
    return out


def _det(x: float, y: float, h: float, conf: float, emb: np.ndarray) -> dict:
    return {
        "box": np.array([x, y, x + h / 2, y + h], np.float32),
        "conf": conf,
        "kpts": _person_kpts(x, y, h),
        "emb": emb / np.linalg.norm(emb),
    }


def _frames(
    dets_per_frame: list[list[dict]], fps: float = 6.0
) -> list[FrameDetections]:
    frames = []
    for i, dets in enumerate(dets_per_frame):
        frames.append(
            FrameDetections(
                frame_idx=i,
                timestamp_s=i / fps,
                boxes=np.array([d["box"] for d in dets], np.float32).reshape(-1, 4),
                conf=np.array([d["conf"] for d in dets], np.float32),
                keypoints=np.array([d["kpts"] for d in dets], np.float32).reshape(
                    -1, 17, 3
                ),
                embeddings=np.array([d["emb"] for d in dets], np.float32).reshape(
                    -1, 4
                ),
            )
        )
    return frames


PARENT_EMB = np.array([1.0, 0.0, 0.0, 0.0])
CHILD_EMB = np.array([0.0, 1.0, 0.0, 0.0])


def test_keypoint_similarity_identical_vs_shifted() -> None:
    a = _person_kpts(0, 0, 200)
    assert keypoint_similarity(a, a, scale=200) > 0.99
    b = _person_kpts(150, 0, 200)
    assert keypoint_similarity(a, b, scale=200) < 0.1


def test_dedup_keeps_overlapping_lap_pair_but_drops_duplicate() -> None:
    parent = _det(100, 100, 300, 0.9, PARENT_EMB)
    # Child on lap: box largely inside the parent's box, different skeleton.
    child = _det(160, 250, 140, 0.4, CHILD_EMB)
    # Duplicate of the parent: same skeleton, slightly shifted box.
    dup = _det(104, 102, 296, 0.3, PARENT_EMB)
    frame = _frames([[parent, child, dup]])[0]
    keep = dedup_frame(frame, AssocConfig())
    assert sorted(keep.tolist()) == [0, 1]


def test_build_tracklets_bridges_short_gap_and_splits_far_jump() -> None:
    seq = []
    for i in range(12):
        if i == 4:
            seq.append([])  # one missing frame
        elif i < 8:
            seq.append([_det(100 + i, 100, 300, 0.9, PARENT_EMB)])
        else:
            seq.append([_det(600, 100, 300, 0.9, PARENT_EMB)])  # teleport
    tracklets = build_tracklets(_frames(seq), AssocConfig())
    lengths = sorted(len(t.frames) for t in tracklets)
    assert lengths == [4, 7]


def test_associate_uses_anchor_and_appearance_after_occlusion() -> None:
    seq = []
    for i in range(30):
        dets = [_det(100, 100, 300, 0.9, PARENT_EMB)]
        if not 10 <= i < 16:  # child hidden for 1 s, then reappears elsewhere
            cx = 500 if i < 10 else 650
            dets.append(_det(cx, 250, 140, 0.35, CHILD_EMB))
        # Shuffle detection order every other frame.
        seq.append(dets[::-1] if i % 2 else dets)
    frames = _frames(seq)
    anchors = [
        AnchorBoxes(
            timestamp_s=frames[2].timestamp_s,
            parent_box=tuple(frames[2].boxes[0 if 2 % 2 == 0 else 1]),
            child_box=tuple(frames[2].boxes[1 if 2 % 2 == 0 else 0]),
        )
    ]
    out = associate_two_people(frames, anchors, AssocConfig())
    child = out[out["role"] == "child"]
    parent = out[out["role"] == "parent"]
    assert len(parent) == 30
    # Child recovered after re-appearing at a new location, through appearance.
    assert set(child["frame_idx"]) == set(range(10)) | set(range(16, 30))
    assert (child["x1"] >= 400).all()
    assert (parent["x1"] < 200).all()


def test_associate_never_gives_one_role_two_boxes_in_a_frame() -> None:
    rng = np.random.default_rng(0)
    seq = []
    for i in range(40):
        dets = [
            _det(100, 100, 300, 0.9, PARENT_EMB),
            _det(500, 250, 140, 0.6, CHILD_EMB),
        ]
        if i % 5 == 0:  # spurious low-confidence blob
            dets.append(_det(800, 50, 100, 0.15, rng.normal(size=4)))
        seq.append(dets)
    frames = _frames(seq)
    anchors = [
        AnchorBoxes(
            frames[0].timestamp_s, tuple(frames[0].boxes[0]), tuple(frames[0].boxes[1])
        )
    ]
    out = associate_two_people(frames, anchors, AssocConfig())
    assert out.groupby(["frame_idx", "role"]).size().max() == 1
    assert (out["x1"] < 790).all()


def test_associate_without_anchors_assigns_larger_person_as_parent() -> None:
    seq = [
        [_det(100, 100, 300, 0.9, PARENT_EMB), _det(500, 250, 140, 0.8, CHILD_EMB)]
        for _ in range(20)
    ]
    out = associate_two_people(_frames(seq), [], AssocConfig())
    parent = out[out["role"] == "parent"]
    assert len(parent) == 20 and (parent["x1"] < 200).all()


def test_conflicting_anchors_split_where_appearance_changes() -> None:
    # One detection stream whose identity silently changes at frame 7
    # (e.g. a tracker hop): the appearance embedding flips there.
    seq = []
    for i in range(24):
        emb = PARENT_EMB if i < 7 else CHILD_EMB
        seq.append([_det(100 + i, 100, 300, 0.9, emb)])
    frames = _frames(seq)
    anchors = [
        AnchorBoxes(
            frames[2].timestamp_s, tuple(frames[2].boxes[0]), (900, 0, 950, 50)
        ),
        AnchorBoxes(
            frames[20].timestamp_s, (900, 0, 950, 50), tuple(frames[20].boxes[0])
        ),
    ]
    out = associate_two_people(frames, anchors, AssocConfig())
    roles = dict(zip(out["frame_idx"], out["role"]))
    assert all(roles[i] == "parent" for i in range(7))
    assert all(roles[i] == "child" for i in range(7, 24))


def test_gap_fill_uses_low_confidence_detection_on_interpolated_path() -> None:
    seq = []
    for i in range(20):
        dets = [_det(100, 100, 300, 0.9, PARENT_EMB)]
        conf = 0.12 if 8 <= i < 12 else 0.7  # child fades behind a shelf
        dets.append(_det(500 + 2 * i, 250, 140, conf, CHILD_EMB))
        seq.append(dets)
    frames = _frames(seq)
    anchors = [
        AnchorBoxes(
            frames[0].timestamp_s, tuple(frames[0].boxes[0]), tuple(frames[0].boxes[1])
        )
    ]
    out = associate_two_people(frames, anchors, AssocConfig(conf_min=0.05))
    child = out[out["role"] == "child"]
    assert set(child["frame_idx"]) == set(range(20))


def test_identity_swap_inside_tracklets_is_split_and_relabeled() -> None:
    # Two people whose tracks silently swap identity at frame 30 (a crossing):
    # the stream at x=100 shrinks and changes appearance, the one at x=500 grows.
    seq = []
    for i in range(60):
        if i < 30:
            a = _det(100, 100, 300, 0.9, PARENT_EMB)
            b = _det(500, 180, 220, 0.9, CHILD_EMB)
        else:
            a = _det(100, 180, 220, 0.9, CHILD_EMB)
            b = _det(500, 100, 300, 0.9, PARENT_EMB)
        seq.append([a, b])
    frames = _frames(seq)
    anchors = [
        AnchorBoxes(
            frames[5].timestamp_s, tuple(frames[5].boxes[0]), tuple(frames[5].boxes[1])
        )
    ]
    out = associate_two_people(
        frames,
        anchors,
        AssocConfig(change_window=6, change_threshold=7.0, anchor_force_s=None),
    )
    parent = out[out["role"] == "parent"].set_index("frame_idx")
    assert (parent.loc[:24, "x1"] < 200).all()
    assert (parent.loc[36:, "x1"] > 400).all()
