"""Tests for gaze_analysis.config — session configuration loading/validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaze_analysis.config import (
    filter_by_time_range,
    filter_heatmaps_by_time_range,
    load_session_config,
    resolve_time_range,
)


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Write a valid session config JSON and return its path."""
    cfg = {
        "session_id": "P001c",
        "output_dir": "video_inference/output/enhanced_5fps",
        "image_width": 854,
        "image_height": 480,
        "frame_rate": 5.0,
        "camera_mappings": [
            {
                "camera_id": "camera_a",
                "parent_track_id": 0,
                "child_track_id": 1,
            },
            {
                "camera_id": "camera_b",
                "parent_track_id": 0,
                "child_track_id": 1,
            },
        ],
        "session_blocks": [
            {
                "name": "grocery_free_play",
                "start_s": 806,
                "end_s": 1420,
                "color": "green",
            },
            {
                "name": "synchrony_intervention",
                "start_s": 1676,
                "end_s": 1725,
                "color": "orange",
            },
            {
                "name": "storybook_reading",
                "start_s": 1762,
                "end_s": 2226,
                "color": "blue",
            },
        ],
    }
    if overrides:
        cfg.update(overrides)
    path = tmp_path / "session_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


# --- Load / round-trip ---


def test_load_session_config_from_json(tmp_path: Path) -> None:
    """Load a well-formed JSON and verify all fields parse correctly."""
    path = _write_config(tmp_path)
    config = load_session_config(path)

    assert config.session_id == "P001c"
    assert config.image_width == 854
    assert config.image_height == 480
    assert config.frame_rate == 5.0
    assert len(config.camera_mappings) == 2
    assert len(config.session_blocks) == 3


def test_camera_mapping_fields(tmp_path: Path) -> None:
    """Verify CameraPersonMapping fields parse correctly."""
    config = load_session_config(_write_config(tmp_path))
    cam_a = config.camera_mappings[0]
    assert cam_a.camera_id == "camera_a"
    assert cam_a.parent_track_id == 0
    assert cam_a.child_track_id == 1


def test_session_block_fields(tmp_path: Path) -> None:
    """Verify SessionBlock fields parse correctly."""
    config = load_session_config(_write_config(tmp_path))
    block = config.session_blocks[0]
    assert block.name == "grocery_free_play"
    assert block.start_s == 806
    assert block.end_s == 1420
    assert block.color == "green"


# --- Defaults ---


def test_default_frame_rate(tmp_path: Path) -> None:
    """frame_rate defaults to 5.0 when omitted."""
    path = _write_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["frame_rate"]
    path.write_text(json.dumps(data), encoding="utf-8")

    config = load_session_config(path)
    assert config.frame_rate == 5.0


# --- Validation errors ---


def test_missing_session_id_raises(tmp_path: Path) -> None:
    """Missing session_id should raise ValueError."""
    path = _write_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["session_id"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="session_id"):
        load_session_config(path)


def test_missing_camera_mappings_raises(tmp_path: Path) -> None:
    """Missing camera_mappings should raise ValueError."""
    path = _write_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["camera_mappings"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="camera_mappings"):
        load_session_config(path)


def test_empty_camera_mappings_raises(tmp_path: Path) -> None:
    """Empty camera_mappings list should raise ValueError."""
    path = _write_config(tmp_path, overrides={"camera_mappings": []})

    with pytest.raises(ValueError, match="camera_mappings"):
        load_session_config(path)


def test_block_start_after_end_raises(tmp_path: Path) -> None:
    """Session block with start_s > end_s should raise ValueError."""
    path = _write_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["session_blocks"] = [
        {"name": "bad_block", "start_s": 100, "end_s": 50, "color": "red"}
    ]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="start_s.*end_s"):
        load_session_config(path)


def test_duplicate_parent_child_ids_raises(tmp_path: Path) -> None:
    """parent_track_id == child_track_id should raise ValueError."""
    path = _write_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["camera_mappings"] = [
        {"camera_id": "camera_a", "parent_track_id": 0, "child_track_id": 0}
    ]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="parent_track_id.*child_track_id"):
        load_session_config(path)


def test_nonexistent_file_raises(tmp_path: Path) -> None:
    """Loading from a nonexistent path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_session_config(tmp_path / "nope.json")


# --- Helpers ---


def test_get_camera_mapping(tmp_path: Path) -> None:
    """get_camera_mapping returns the right mapping for a camera_id."""
    config = load_session_config(_write_config(tmp_path))
    mapping = config.get_camera_mapping("camera_b")
    assert mapping.camera_id == "camera_b"


def test_get_camera_mapping_not_found(tmp_path: Path) -> None:
    """get_camera_mapping raises KeyError for unknown camera_id."""
    config = load_session_config(_write_config(tmp_path))
    with pytest.raises(KeyError):
        config.get_camera_mapping("camera_z")


# --- Session block lookup ---


def test_get_session_block(tmp_path: Path) -> None:
    """get_session_block returns the right block by name."""
    config = load_session_config(_write_config(tmp_path))
    block = config.get_session_block("storybook_reading")
    assert block.start_s == 1762
    assert block.end_s == 2226


def test_get_session_block_not_found(tmp_path: Path) -> None:
    """get_session_block raises KeyError for unknown name."""
    config = load_session_config(_write_config(tmp_path))
    with pytest.raises(KeyError, match="nope"):
        config.get_session_block("nope")


# --- Time range resolution ---


def test_resolve_time_range_explicit(tmp_path: Path) -> None:
    """Explicit start_s/end_s are returned directly."""
    config = load_session_config(_write_config(tmp_path))
    start, end = resolve_time_range(config, start_s=600.0, end_s=1200.0)
    assert start == 600.0
    assert end == 1200.0


def test_resolve_time_range_defaults(tmp_path: Path) -> None:
    """No args → (0.0, inf)."""
    config = load_session_config(_write_config(tmp_path))
    start, end = resolve_time_range(config)
    assert start == 0.0
    assert end == float("inf")


def test_resolve_time_range_start_only(tmp_path: Path) -> None:
    """Only start_s → (start_s, inf)."""
    config = load_session_config(_write_config(tmp_path))
    start, end = resolve_time_range(config, start_s=600.0)
    assert start == 600.0
    assert end == float("inf")


def test_resolve_time_range_block_overrides(tmp_path: Path) -> None:
    """time_block overrides explicit start_s/end_s."""
    config = load_session_config(_write_config(tmp_path))
    start, end = resolve_time_range(
        config, start_s=0.0, end_s=9999.0, time_block="grocery_free_play"
    )
    assert start == 806
    assert end == 1420


def test_resolve_time_range_unknown_block_raises(tmp_path: Path) -> None:
    """Unknown time_block name raises KeyError."""
    config = load_session_config(_write_config(tmp_path))
    with pytest.raises(KeyError):
        resolve_time_range(config, time_block="nope")


# --- DataFrame filtering ---


def test_filter_by_time_range_basic() -> None:
    """Rows outside [start, end] are removed."""
    df = pd.DataFrame({"timestamp_s": [1.0, 5.0, 10.0, 15.0, 20.0], "v": range(5)})
    result = filter_by_time_range(df, 5.0, 15.0)
    assert list(result["timestamp_s"]) == [5.0, 10.0, 15.0]


def test_filter_by_time_range_noop() -> None:
    """(0, inf) returns the original DataFrame unchanged."""
    df = pd.DataFrame({"timestamp_s": [1.0, 5.0, 10.0], "v": range(3)})
    result = filter_by_time_range(df, 0.0, float("inf"))
    assert len(result) == 3


def test_filter_by_time_range_empty_result() -> None:
    """No rows match the range."""
    df = pd.DataFrame({"timestamp_s": [1.0, 2.0, 3.0], "v": range(3)})
    result = filter_by_time_range(df, 100.0, 200.0)
    assert len(result) == 0


# --- Heatmap filtering ---


def test_filter_heatmaps_by_time_range() -> None:
    """Only heatmaps within the time range are kept."""
    heatmaps = np.random.rand(4, 64, 64).astype(np.float32)
    keys = ["f000000_t0", "f000000_t1", "f000005_t0", "f000005_t1"]
    frame_ts = {0: 0.0, 5: 1.0}

    filtered_hm, filtered_keys = filter_heatmaps_by_time_range(
        heatmaps, keys, frame_ts, 0.5, 2.0
    )
    assert len(filtered_keys) == 2
    assert all("f000005" in k for k in filtered_keys)


def test_filter_heatmaps_noop() -> None:
    """(0, inf) returns all heatmaps."""
    heatmaps = np.random.rand(4, 64, 64).astype(np.float32)
    keys = ["f000000_t0", "f000000_t1", "f000005_t0", "f000005_t1"]
    frame_ts = {0: 0.0, 5: 1.0}

    filtered_hm, filtered_keys = filter_heatmaps_by_time_range(
        heatmaps, keys, frame_ts, 0.0, float("inf")
    )
    assert len(filtered_keys) == 4
