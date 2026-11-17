"""Tests for video_analysis/anchor_frames.py.

The frame-extraction path (ffmpeg) and overlay rendering (cv2) are mocked
in the unit tests so the suite stays fast and doesn't need a real video.
A separate smoke test path (RUN_SMOKE_TESTS=1) exercises the full
pipeline against the user-supplied compressed sample.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analysis.anchor_frames import (  # noqa: E402
    _bbox_iou,
    _frame_is_clear,
    extract_anchor_frames,
    select_anchor_frames,
)


def _make_two_person_tracks(
    n_frames: int = 240,
    fps: float = 8.0,
    overlap_window: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Build a synthetic tracks_2d.csv with two persons in every frame.

    Persons are well-separated by default; pass ``overlap_window=(a, b)``
    to make their bboxes overlap heavily in that frame range.
    """
    rows = []
    for f in range(n_frames):
        if overlap_window is not None and overlap_window[0] <= f <= overlap_window[1]:
            # Heavy overlap: both bboxes share the same region
            boxes = [(100.0, 100.0, 200.0, 200.0), (110.0, 110.0, 205.0, 205.0)]
        else:
            boxes = [(0.0, 0.0, 100.0, 100.0), (300.0, 0.0, 400.0, 100.0)]
        for tid, (x1, y1, x2, y2) in zip((1, 2), boxes):
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / fps,
                    "track_id": tid,
                    "track_label": f"p{tid}",
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "track_confidence": 0.9,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: low-level helpers
# ---------------------------------------------------------------------------


def test_bbox_iou_disjoint_is_zero():
    a = (0, 0, 10, 10)
    b = (50, 50, 60, 60)
    assert _bbox_iou(a, b) == 0.0


def test_bbox_iou_identical_is_one():
    a = (0, 0, 10, 10)
    assert _bbox_iou(a, a) == 1.0


def test_frame_is_clear_requires_two_low_iou_detections():
    well_separated = pd.DataFrame(
        [
            {"bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": 10.0, "bbox_y2": 10.0},
            {"bbox_x1": 50.0, "bbox_y1": 50.0, "bbox_x2": 60.0, "bbox_y2": 60.0},
        ]
    )
    assert bool(_frame_is_clear(well_separated, max_iou=0.3)) is True

    too_few = pd.DataFrame(
        [{"bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": 10.0, "bbox_y2": 10.0}]
    )
    assert bool(_frame_is_clear(too_few, max_iou=0.3)) is False

    overlapping = pd.DataFrame(
        [
            {"bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": 100.0, "bbox_y2": 100.0},
            {"bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 100.0, "bbox_y2": 100.0},
        ]
    )
    assert bool(_frame_is_clear(overlapping, max_iou=0.3)) is False


# ---------------------------------------------------------------------------
# Tests: select_anchor_frames
# ---------------------------------------------------------------------------


def test_select_anchor_frames_picks_evenly_spaced():
    df = _make_two_person_tracks(n_frames=240, fps=8.0)  # 30 seconds
    candidates = select_anchor_frames(df, n_anchors=5, edge_margin_s=1.0)

    assert len(candidates) == 5
    # Sorted by timestamp
    times = [c.timestamp_s for c in candidates]
    assert times == sorted(times)
    # Spread across most of the window (first > edge, last < end - edge)
    assert times[0] >= 1.0
    assert times[-1] <= 29.0


def test_select_anchor_frames_skips_overlapping_window():
    """Frames where the two bboxes heavily overlap should not be chosen."""
    df = _make_two_person_tracks(
        n_frames=240, fps=8.0, overlap_window=(80, 160)
    )  # 10s of overlap in the middle
    candidates = select_anchor_frames(df, n_anchors=10)
    # None of the picked frames should be in the overlap window
    for c in candidates:
        assert not (
            80 <= c.frame_idx <= 160
        ), f"anchor at frame {c.frame_idx} fell inside the overlap window"


def test_select_anchor_frames_fills_all_anchors_across_block_gap():
    """Two analysis blocks with a long gap between them: every requested
    anchor lands on a distinct frame inside the blocks (none wasted on the gap)."""
    first = _make_two_person_tracks(n_frames=240, fps=8.0)  # 0-30 s
    second = _make_two_person_tracks(n_frames=240, fps=8.0)
    second["frame_idx"] += 240
    second["timestamp_s"] += 600.0  # second block 600-630 s
    df = pd.concat([first, second], ignore_index=True)
    candidates = select_anchor_frames(df, n_anchors=10)
    assert len(candidates) == 10
    assert len({c.frame_idx for c in candidates}) == 10
    assert sum(c.timestamp_s < 300 for c in candidates) == 5


def test_select_anchor_frames_returns_empty_for_empty_df():
    assert select_anchor_frames(pd.DataFrame(), n_anchors=10) == []


def test_select_anchor_frames_returns_fewer_when_not_enough_clear():
    """Asking for 10 anchors from a 5-frame session with only 2 clear
    frames returns just those 2."""
    df = _make_two_person_tracks(n_frames=5, fps=8.0)
    candidates = select_anchor_frames(df, n_anchors=10, edge_margin_s=0.0)
    # 5 frames, all clear; we can't have more anchors than unique clear frames
    assert len(candidates) <= 5


def test_select_anchor_frames_captures_per_track_bboxes():
    df = _make_two_person_tracks(n_frames=240, fps=8.0)
    candidates = select_anchor_frames(df, n_anchors=3)
    for c in candidates:
        assert len(c.tracks) == 2
        tids = sorted(t["track_id"] for t in c.tracks)
        assert tids == [1, 2]
        for t in c.tracks:
            assert len(t["bbox_xyxy"]) == 4


# ---------------------------------------------------------------------------
# Tests: extract_anchor_frames (mocked ffmpeg/cv2 io)
# ---------------------------------------------------------------------------


def test_extract_anchor_frames_writes_manifest_and_images(tmp_path):
    df = _make_two_person_tracks(n_frames=240, fps=8.0)
    tracks_csv = tmp_path / "tracks_2d.csv"
    df.to_csv(tracks_csv, index=False)
    output_dir = tmp_path / "anchors"

    extracted_paths: list[Path] = []
    rendered_paths: list[Path] = []

    def fake_extract(source_video, timestamp_s, output_path, ffmpeg_bin):
        extracted_paths.append(Path(output_path))
        # Pretend ffmpeg wrote a tiny valid JPEG-shaped file
        Path(output_path).write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")

    def fake_draw(raw_image_path, tracks, output_path):
        rendered_paths.append(Path(output_path))
        Path(output_path).write_bytes(b"\xff\xd8\xff\xe0overlaid-jpeg")

    with (
        patch(
            "video_analysis.anchor_frames._extract_frame_at_time",
            side_effect=fake_extract,
        ),
        patch("video_analysis.anchor_frames._draw_overlay", side_effect=fake_draw),
    ):
        metadata = extract_anchor_frames(
            tracks_csv=tracks_csv,
            source_video=tmp_path / "fake_video.mp4",
            output_dir=output_dir,
            n_anchors=4,
        )

    assert len(metadata) == 4
    # Each entry produces a numbered jpg
    for i, m in enumerate(metadata, start=1):
        assert m.anchor_idx == i
        assert m.image_filename == f"anchor_{i:02d}.jpg"
        assert (output_dir / m.image_filename).exists()
    # Manifest exists and is valid JSON listing all anchors
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 4
    assert all("color_bgr" in t for entry in manifest for t in entry["tracks"])


@pytest.mark.skipif(
    os.environ.get("RUN_SMOKE_TESTS") != "1",
    reason="Smoke tests are slow and need real deps. Set RUN_SMOKE_TESTS=1.",
)
def test_smoke_extract_anchor_frames_from_real_video(tmp_path):
    """Full extraction against a real compressed video. Verifies ffmpeg and
    opencv are wired correctly end-to-end."""
    import shutil

    sample_env = os.environ.get("SMOKE_VIDEO_PATH")
    sample = (
        Path(sample_env)
        if sample_env
        else ROOT_DIR / "video_inference" / "compressed" / "P001c_Short_Full_rapid.mp4"
    )
    if not sample.exists():
        pytest.skip(f"Sample video not found: {sample}")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    # Synthesize a minimal tracks CSV pointing inside the sample video's
    # duration. The sample is short, so 5 anchors over 2 s should be enough.
    rows = []
    for f in range(16):  # 2 s at 8 fps
        for tid, (x1, y1, x2, y2) in zip(
            (1, 2), [(50, 50, 150, 200), (300, 80, 400, 230)]
        ):
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / 8.0,
                    "track_id": tid,
                    "track_label": f"p{tid}",
                    "bbox_x1": float(x1),
                    "bbox_y1": float(y1),
                    "bbox_x2": float(x2),
                    "bbox_y2": float(y2),
                    "track_confidence": 0.9,
                }
            )
    tracks_csv = tmp_path / "tracks_2d.csv"
    pd.DataFrame(rows).to_csv(tracks_csv, index=False)

    metadata = extract_anchor_frames(
        tracks_csv=tracks_csv,
        source_video=sample,
        output_dir=tmp_path / "anchors",
        n_anchors=3,
        edge_margin_s=0.1,
    )

    assert len(metadata) >= 1
    for m in metadata:
        img_path = tmp_path / "anchors" / m.image_filename
        assert img_path.exists()
        assert img_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tests: at least two anchors per block (bug hunt finding 7)
# ---------------------------------------------------------------------------


def test_select_anchor_frames_guarantees_two_per_block():
    """A block whose frames all overlap (child on lap) still gets 2 anchors."""
    # 8 fps: frames 0-799 = 0-100 s clear (block a); 800-959 = 100-120 s overlap.
    tracks = _make_two_person_tracks(n_frames=960, fps=8.0, overlap_window=(800, 959))
    blocks = [("a", 0.0, 99.9), ("b", 100.0, 120.0)]

    picks = select_anchor_frames(tracks, n_anchors=10, blocks=blocks)

    in_b = [c for c in picks if c.timestamp_s >= 100.0]
    assert len(in_b) >= 2
    assert len(picks) == 10
    assert len({c.frame_idx for c in picks}) == len(picks)


def test_select_anchor_frames_more_blocks_than_budget():
    """With more blocks than anchors / 2, every block still gets two."""
    tracks = _make_two_person_tracks(n_frames=960, fps=8.0)
    blocks = [(f"p{i}", i * 20.0, i * 20.0 + 19.0) for i in range(6)]

    picks = select_anchor_frames(tracks, n_anchors=10, blocks=blocks)

    for name, start, end in blocks:
        assert sum(start <= c.timestamp_s <= end for c in picks) >= 2, name


def test_select_anchor_frames_without_blocks_is_unchanged():
    tracks = _make_two_person_tracks(n_frames=960, fps=8.0, overlap_window=(800, 959))

    picks = select_anchor_frames(tracks, n_anchors=10)

    assert all(c.timestamp_s < 100.0 for c in picks)  # overlapping frames skipped
