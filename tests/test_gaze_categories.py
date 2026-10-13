"""Tests for frame-level gaze categories (gaze_analysis.gaze_categories)."""

from __future__ import annotations

import pandas as pd
import pytest

from gaze_analysis.gaze_categories import (
    CATEGORIES,
    absorb_blips,
    build_frame_table,
    category_shares,
    chance_shares,
    classify_frames,
)

# Pixel layout on a 960x540 frame: parent on the left, child on the right.
P_HEAD = (100, 100, 160, 160)
C_HEAD = (700, 200, 740, 240)
P_BODY = (80, 100, 200, 500)
C_BODY = (680, 200, 760, 450)
P_FACE_PT = (130, 130)
C_FACE_PT = (720, 220)
C_BODY_PT = (720, 380)
TOY = (450, 470)


def _row(p_pt, c_pt, p_in=0.9, c_in=0.9, **extra):
    row = {
        "frame_idx": 0,
        "timestamp_s": 0.0,
        "p_gx": p_pt[0], "p_gy": p_pt[1], "p_in": p_in,
        "c_gx": c_pt[0], "c_gy": c_pt[1], "c_in": c_in,
        "p_head": P_HEAD, "c_head": C_HEAD,
        "p_body": P_BODY, "c_body": C_BODY,
    }
    row.update(extra)
    return row


@pytest.mark.parametrize(
    "p_pt, c_pt, p_in, c_in, expected",
    [
        (C_FACE_PT, P_FACE_PT, 0.9, 0.9, "mutual_gaze"),
        (C_FACE_PT, TOY, 0.9, 0.9, "parent_watching"),
        (C_BODY_PT, TOY, 0.9, 0.9, "parent_watching"),
        (TOY, P_FACE_PT, 0.9, 0.9, "child_watching"),
        (C_BODY_PT, (140, 300), 0.9, 0.9, "both_watching"),
        (TOY, (TOY[0] + 30, TOY[1]), 0.9, 0.9, "joint_attention"),
        (TOY, (450, 60), 0.9, 0.9, "independent"),
        (TOY, TOY, 0.2, 0.9, "unknown"),  # parent looks out of frame
        # Child watching still counts when the parent's gaze is unknown.
        (TOY, P_FACE_PT, 0.2, 0.9, "child_watching"),
    ],
)
def test_classify_frames(p_pt, c_pt, p_in, c_in, expected) -> None:
    frames = pd.DataFrame([_row(p_pt, c_pt, p_in, c_in)])
    assert classify_frames(frames).tolist() == [expected]


def test_joint_attention_scales_with_head_size() -> None:
    # 70 px apart: within 1.5 head-heights when heads are 50 px tall
    # (mean of 60 and 40), not when heads are small.
    near = pd.DataFrame([_row(TOY, (TOY[0] + 70, TOY[1]))])
    assert classify_frames(near).tolist() == ["joint_attention"]
    small = _row(TOY, (TOY[0] + 70, TOY[1]))
    small["p_head"] = (100, 100, 115, 115)
    small["c_head"] = (700, 200, 715, 215)
    assert classify_frames(pd.DataFrame([small])).tolist() == ["independent"]


def test_missing_partner_body_is_unknown() -> None:
    row = _row(TOY, TOY)
    row["c_body"] = None
    row["c_head"] = None
    assert classify_frames(pd.DataFrame([row])).tolist() == ["unknown"]


def test_absorb_blips_only_fills_single_consecutive_frames() -> None:
    cats = pd.Series(["a", "a", "b", "a", "a", "c", "a"])
    frames = pd.Series([0, 1, 2, 3, 4, 5, 9])  # gap after frame 5
    assert absorb_blips(cats, frames).tolist() == ["a", "a", "a", "a", "a", "c", "a"]


def test_category_shares_sum_to_100() -> None:
    shares = category_shares(pd.Series(["mutual_gaze", "unknown", "unknown", "independent"]))
    assert set(shares) == {f"{c}_pct" for c in CATEGORIES}
    assert shares["unknown_pct"] == 50.0
    assert sum(shares.values()) == pytest.approx(100.0)


def test_chance_shares_break_the_pairing() -> None:
    # Child looks at the parent's face only while the parent looks back
    # (first half); shifting the child's gaze in time destroys the pairing.
    rows = []
    for i in range(20):
        mutual = i < 10
        rows.append(
            dict(
                _row(C_FACE_PT if mutual else TOY, P_FACE_PT if mutual else (450, 60)),
                frame_idx=i,
                timestamp_s=i / 6,
            )
        )
    frames = pd.DataFrame(rows)
    real = category_shares(classify_frames(frames))
    chance = chance_shares(frames, shift_rows=10)
    assert real["mutual_gaze_pct"] == 50.0
    assert chance["mutual_gaze_pct"] == 0.0


def test_build_frame_table_joins_gaze_heads_and_bodies() -> None:
    gaze = pd.DataFrame(
        [
            {"frame_idx": 5, "timestamp_s": 1.0, "track_id": 0, "gaze_peak_x": 0.75,
             "gaze_peak_y": 0.4, "inout_score": 0.9},
            {"frame_idx": 5, "timestamp_s": 1.0, "track_id": 1, "gaze_peak_x": 0.135,
             "gaze_peak_y": 0.24, "inout_score": 0.8},
        ]
    )
    heads = pd.DataFrame(
        [
            {"frame_idx": 5, "track_id": 0, "head_x1": 100 / 960, "head_y1": 100 / 540,
             "head_x2": 160 / 960, "head_y2": 160 / 540},
            {"frame_idx": 5, "track_id": 1, "head_x1": 700 / 960, "head_y1": 200 / 540,
             "head_x2": 740 / 960, "head_y2": 240 / 540},
        ]
    )
    tracks = pd.DataFrame(
        [
            {"frame_idx": 5, "track_id": 0, "bbox_x1": 80, "bbox_y1": 100, "bbox_x2": 200, "bbox_y2": 500},
            {"frame_idx": 5, "track_id": 1, "bbox_x1": 680, "bbox_y1": 200, "bbox_x2": 760, "bbox_y2": 450},
        ]
    )
    frames = build_frame_table(gaze, heads, tracks, image_size=(960, 540))

    assert len(frames) == 1
    row = frames.iloc[0]
    assert row["p_gx"] == pytest.approx(720.0)
    assert row["c_in"] == pytest.approx(0.8)
    assert tuple(row["c_head"]) == pytest.approx((700, 200, 740, 240))
    assert classify_frames(frames).tolist() == ["mutual_gaze"]
