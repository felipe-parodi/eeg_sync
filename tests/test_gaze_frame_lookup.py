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


def test_subsample_frames_keeps_every_nth_grid_frame() -> None:
    from gaze_analysis.gazelle_runner import subsample_frame_indices

    frames = list(range(5280, 5300))  # 6 fps grid
    kept = subsample_frame_indices(frames, frame_rate=6.0, target_fps=3.0)
    assert kept == list(range(5280, 5300, 2))
    assert subsample_frame_indices(frames, 6.0, 0.0) == frames
    assert subsample_frame_indices(frames, 6.0, 12.0) == frames


def test_frame_image_size_reads_actual_frames(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from gaze_analysis.config import frame_image_size

    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "frame_000001.jpg"), np.zeros((540, 960, 3), np.uint8))
    (frames / "frame_index.csv").write_text(
        "frame_idx,timestamp_s,image_name\n0,1.0,frame_000001.jpg\n", encoding="utf-8"
    )

    assert frame_image_size(tmp_path, fallback=(854, 480)) == (960, 540)
    assert frame_image_size(tmp_path / "missing", fallback=(854, 480)) == (854, 480)
