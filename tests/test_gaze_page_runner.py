"""Pure-logic tests for the PaGE gaze runner (no model download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaze_analysis.gaze_schema import GAZE_HEATMAP_COLUMNS
from gaze_analysis.page_runner import (
    HEAD_SIZE,
    SCENE_SIZE,
    build_arg_parser,
    heatmap_peak,
    pixel_head_box,
    prepare_frame,
)


def test_pixel_head_box_rounds_clips_and_renormalizes() -> None:
    pixel, norm = pixel_head_box((0.25, 0.5, 0.5, 1.2), width=960, height=540)

    assert pixel == (240, 270, 480, 540)
    assert norm == pytest.approx((0.25, 0.5, 0.5, 1.0))


def test_pixel_head_box_rejects_empty_boxes() -> None:
    assert pixel_head_box((1.2, 0.1, 1.5, 0.2), width=960, height=540) is None
    assert pixel_head_box((0.3, 0.3, 0.3, 0.4), width=960, height=540) is None


def test_heatmap_peak_matches_gazelle_convention() -> None:
    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[10, 63] = 0.9

    x, y, value = heatmap_peak(heatmap)

    assert x == pytest.approx(1.0)
    assert y == pytest.approx(10 / 63)
    assert value == pytest.approx(0.9)


def test_prepare_frame_builds_scene_and_head_tensors(tmp_path: Path) -> None:
    from PIL import Image

    rgb = np.zeros((540, 960, 3), dtype=np.uint8)
    rgb[:, 480:] = 255
    path = tmp_path / "frame_000001.jpg"
    Image.fromarray(rgb).save(path)

    scene, heads, boxes = prepare_frame(
        path, [(0.6, 0.1, 0.7, 0.3), (2.0, 0.1, 2.5, 0.3)]
    )

    assert scene.shape == (3, *SCENE_SIZE)
    assert [h.shape for h in heads] == [(3, *HEAD_SIZE)] * 2
    assert boxes[0] == pytest.approx((576 / 960, 54 / 540, 672 / 960, 162 / 540))
    assert boxes[1] is None
    # Head crop is taken from the white half: normalized values well above 0.
    assert heads[0].mean() > 1.5


def test_cli_defaults_match_portal_gaze_step() -> None:
    args = build_arg_parser().parse_args(
        ["--camera-dir", "x", "--session-config", "y.json"]
    )

    assert args.model_name == "page-vitb"
    assert args.output_csv == "gaze_heatmap.csv"
    assert args.output_npz == "gaze_heatmaps.npz"
    assert args.output_dir is None


def test_output_columns_are_the_shared_gaze_schema() -> None:
    assert GAZE_HEATMAP_COLUMNS[:3] == ["frame_idx", "timestamp_s", "track_id"]
