"""Gaze steps must find each row's image by source-video time."""

from __future__ import annotations

from pathlib import Path

from gaze_analysis.gazelle_runner import _find_frame_paths


def test_find_frame_paths_maps_time_grid_frame_idx_to_images(tmp_path: Path) -> None:
    # Extracted frames are numbered from 1 within the analysis windows, while
    # smoothed tables number frames by time (frame_idx = round(t * fps)).
    frames = tmp_path / "frames"
    frames.mkdir()
    rows = ["frame_idx,timestamp_s,image_name,window_name,window_start_s,window_end_s"]
    for i, t in enumerate([880.0, 880.0 + 1 / 6, 900.0]):
        name = f"frame_{i + 1:06d}.jpg"
        (frames / name).write_bytes(b"")
        rows.append(f"{i},{t:.6f},{name},grocery,880,1523")
    (frames / "frame_index.csv").write_text("\n".join(rows), encoding="utf-8")

    paths = _find_frame_paths(tmp_path, frame_rate=6.0)

    assert paths[5280].name == "frame_000001.jpg"
    assert paths[5281].name == "frame_000002.jpg"
    assert paths[5400].name == "frame_000003.jpg"
    assert 1 not in paths
