"""Tests for scoring gaze predictions against the colleague's labels."""

from __future__ import annotations

import pandas as pd
import pytest

from gaze_analysis.label_eval import predicted_target, score_gaze_predictions

# Image 1000 x 500. Parent box on the left, child box on the right.
BOXES = {"parent": (100, 100, 300, 500), "child": (600, 200, 800, 500)}
W, H = 1000, 500


@pytest.mark.parametrize(
    "point, expected",
    [
        ((0.70, 0.45), "partner_face"),  # top of child box, as seen by parent
        ((0.70, 0.90), "partner_body"),
        ((0.45, 0.20), "not_partner"),
    ],
)
def test_predicted_target_from_gaze_point(point, expected) -> None:
    got = predicted_target(point, BOXES["child"], W, H, inout=0.9)
    assert got == expected


def test_predicted_target_out_of_frame_is_not_partner() -> None:
    assert predicted_target((0.7, 0.45), BOXES["child"], W, H, inout=0.1) == "not_partner"


def test_score_gaze_predictions() -> None:
    labels = pd.DataFrame(
        [
            # parent looks at child's face; model agrees
            {"frame_idx": 1, "person": "parent", "target": "partner_face", "point": [0.7, 0.45]},
            # child looks at an object; model says parent's face -> wrong
            {"frame_idx": 1, "person": "child", "target": "object", "point": None},
            # unusable label is ignored
            {"frame_idx": 2, "person": "child", "target": "cant_tell", "point": None},
        ]
    )
    gaze = pd.DataFrame(
        [
            {"frame_idx": 1, "track_id": 0, "gaze_peak_x": 0.72, "gaze_peak_y": 0.45, "inout_score": 0.9},
            {"frame_idx": 1, "track_id": 1, "gaze_peak_x": 0.20, "gaze_peak_y": 0.30, "inout_score": 0.9},
        ]
    )
    tracks = pd.DataFrame(
        [
            {"frame_idx": 1, "track_id": 0, "bbox_x1": 100, "bbox_y1": 100, "bbox_x2": 300, "bbox_y2": 500},
            {"frame_idx": 1, "track_id": 1, "bbox_x1": 600, "bbox_y1": 200, "bbox_x2": 800, "bbox_y2": 500},
        ]
    )

    scores = score_gaze_predictions(labels, gaze, tracks, image_size=(W, H))

    assert scores["n_labels"] == 2
    assert scores["target_accuracy"] == 0.5
    assert scores["looking_at_partner_accuracy"] == 0.5
    assert scores["n_points"] == 1
    assert scores["point_error_median"] == pytest.approx(0.02, abs=1e-6)
