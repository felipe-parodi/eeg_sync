"""Tests for block-based track filtering."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analysis.track_filter import (
    BlockDef,
    TrackFilterConfig,
    identify_top_tracks,
    parse_blocks_string,
)


def _make_tracks_df(n_frames: int = 100, tracks=None):
    """Create synthetic tracks DataFrame.

    Args:
        n_frames: Number of frames.
        tracks: List of (track_id, start_frame, end_frame, bbox_size) tuples.
    """
    if tracks is None:
        tracks = [
            (0, 0, n_frames, 200),  # parent: large bbox, always present
            (1, 0, n_frames, 100),  # child: small bbox, always present
            (2, 30, 50, 150),  # researcher: medium, brief appearance
        ]

    rows = []
    for tid, sf, ef, size in tracks:
        for f in range(sf, ef):
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / 30.0,
                    "track_id": tid,
                    "track_label": f"person_{tid:02d}",
                    "bbox_x1": 100.0,
                    "bbox_y1": 100.0,
                    "bbox_x2": 100.0 + size,
                    "bbox_y2": 100.0 + size,
                    "track_confidence": 0.9,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: identify_top_tracks
# ---------------------------------------------------------------------------


def test_identify_top_tracks_basic():
    df = _make_tracks_df(100)
    result = identify_top_tracks(df, 0, 99, n_keep=2)
    assert len(result) == 2
    # Track 0 (area=200*200=40000) should be parent (first).
    assert result[0][0] == 0
    assert result[0][1] == "parent"
    # Track 1 (area=100*100=10000) should be child.
    assert result[1][0] == 1
    assert result[1][1] == "child"


def test_identify_top_tracks_ignores_brief():
    """Brief researcher track should not be in top 2."""
    df = _make_tracks_df(100)
    result = identify_top_tracks(df, 0, 99, n_keep=2)
    track_ids = [r[0] for r in result]
    assert 2 not in track_ids


def test_identify_top_tracks_empty():
    df = _make_tracks_df(100)
    result = identify_top_tracks(df, 200, 300, n_keep=2)
    assert result == []


def test_identify_top_tracks_area_ordering():
    """Larger bbox area should be parent (role 0)."""
    tracks = [
        (5, 0, 50, 300),  # big bbox
        (7, 0, 50, 80),  # small bbox
    ]
    df = _make_tracks_df(50, tracks)
    result = identify_top_tracks(df, 0, 49, n_keep=2)
    assert result[0][0] == 5  # bigger = parent
    assert result[0][1] == "parent"
    assert result[1][0] == 7  # smaller = child
    assert result[1][1] == "child"


# ---------------------------------------------------------------------------
# Tests: parse_blocks_string
# ---------------------------------------------------------------------------


def test_parse_blocks_string():
    blocks = parse_blocks_string(
        "grocery,13:26,23:40;synchrony,27:56,28:45;storybook,29:22,37:06"
    )
    assert len(blocks) == 3
    assert blocks[0].name == "grocery"
    assert blocks[0].start_time == "13:26"
    assert blocks[2].name == "storybook"
    assert blocks[2].end_time == "37:06"


def test_parse_blocks_string_invalid():
    with pytest.raises(ValueError):
        parse_blocks_string("bad_format")


# ---------------------------------------------------------------------------
# Tests: filter_tracks (integration)
# ---------------------------------------------------------------------------


def test_filter_tracks_integration(tmp_path):
    """End-to-end test: filter tracks by block, remap IDs."""
    from video_analysis.track_filter import filter_tracks

    # Create session structure.
    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)

    # Track 0 = big bbox (parent), track 1 = small (child), track 2 = brief.
    df = _make_tracks_df(100)
    df.to_csv(cam_dir / "tracks_2d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[BlockDef("test_block", "0:00", "0:03")],  # frames 0-90
        source_fps=30.0,
        n_keep=2,
    )
    summary = filter_tracks(cfg)

    assert summary["total_rows"] > 0
    # Read filtered output.
    filtered = pd.read_csv(cam_dir / "tracks_2d_filtered.csv")
    assert set(filtered["track_id"].unique()) == {0, 1}
    assert set(filtered["track_label"].unique()) == {"parent", "child"}


def test_filter_tracks_uses_timestamp_windows_for_segmented_outputs(tmp_path):
    """Filtering should use original timestamps, not sparse frame indices."""
    from video_analysis.track_filter import filter_tracks

    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)
    rows = []
    for frame_idx, timestamp_s in [(0, 100.0), (1, 100.5), (2, 200.0)]:
        rows.extend(
            [
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "track_id": 7,
                    "track_label": "person_07",
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": 100.0,
                    "bbox_y2": 200.0,
                    "track_confidence": 0.9,
                },
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "track_id": 9,
                    "track_label": "person_09",
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": 80.0,
                    "bbox_y2": 120.0,
                    "track_confidence": 0.9,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(cam_dir / "tracks_2d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[BlockDef("free_play", "1:40", "1:41")],
        source_fps=30.0,
        n_keep=2,
    )
    summary = filter_tracks(cfg)

    filtered = pd.read_csv(cam_dir / "tracks_2d_filtered.csv")

    assert summary["total_rows"] == 4
    assert set(filtered["frame_idx"].unique()) == {0, 1}
    assert set(filtered["timestamp_s"].unique()) == {100.0, 100.5}
    assert set(filtered["track_id"].unique()) == {0, 1}


def test_filter_tracks_treats_adjacent_blocks_as_half_open(tmp_path):
    """The end timestamp belongs to the next adjacent block, not both."""
    from video_analysis.track_filter import filter_tracks

    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)
    rows = []
    for frame_idx, timestamp_s in [(0, 10.0), (1, 10.5), (2, 11.0), (3, 11.5)]:
        for track_id, bbox_x2 in [(7, 100.0), (9, 80.0)]:
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "track_id": track_id,
                    "track_label": f"person_{track_id}",
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": bbox_x2,
                    "bbox_y2": 100.0,
                    "track_confidence": 0.9,
                }
            )
    pd.DataFrame(rows).to_csv(cam_dir / "tracks_2d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[
            BlockDef("first", "0:10", "0:11"),
            BlockDef("second", "0:11", "0:12"),
        ],
        n_keep=2,
    )
    filter_tracks(cfg)

    filtered = pd.read_csv(cam_dir / "tracks_2d_filtered.csv")

    assert filtered["timestamp_s"].tolist().count(11.0) == 2
    assert len(filtered) == 8


# ---------------------------------------------------------------------------
# Tests: bbox-area continuity re-link
# ---------------------------------------------------------------------------


def _swap_tracks_df_with_crossover(
    n_frames: int = 60,
    crossover_frame: int = 30,
    parent_id: int = 10,
    child_id: int = 20,
    parent_size: float = 220.0,
    child_size: float = 90.0,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Synthetic tracks where IDs swap at ``crossover_frame``.

    Before ``crossover_frame``: parent_id has parent_size bbox, child_id has
    child_size bbox. After ``crossover_frame``: the IDs swap.
    """
    rows = []
    for f in range(n_frames):
        if f < crossover_frame:
            big_id, small_id = parent_id, child_id
        else:
            big_id, small_id = child_id, parent_id
        for tid, size in ((big_id, parent_size), (small_id, child_size)):
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / fps,
                    "track_id": tid,
                    "track_label": f"person_{tid:02d}",
                    "bbox_x1": 100.0,
                    "bbox_y1": 100.0,
                    "bbox_x2": 100.0 + size,
                    "bbox_y2": 100.0 + size,
                    "track_confidence": 0.9,
                }
            )
    return pd.DataFrame(rows)


def test_filter_tracks_relinks_id_swap(tmp_path):
    """When ByteTrack swaps parent/child IDs mid-block, the filter corrects it.

    Before the fix, post-swap frames where original_id == parent_id (now
    actually the child) would be labeled "parent". After re-link, role
    follows bbox size at every frame.
    """
    from video_analysis.track_filter import filter_tracks

    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)

    # 10s block, crossover at 5s — anchor window (default 4s) sees only the
    # pre-crossover assignment, so the size signal stays clean.
    df = _swap_tracks_df_with_crossover(
        n_frames=300, crossover_frame=150, parent_size=220.0, child_size=90.0
    )
    df.to_csv(cam_dir / "tracks_2d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[BlockDef("test_block", "0:00", "0:10")],
        source_fps=30.0,
        n_keep=2,
    )
    summary = filter_tracks(cfg)

    assert summary["blocks"][0]["relink_swaps"], "expected at least one swap range"

    filtered = pd.read_csv(cam_dir / "tracks_2d_filtered.csv")
    parent_rows = filtered[filtered["track_id"] == 0]
    child_rows = filtered[filtered["track_id"] == 1]

    parent_areas = (parent_rows["bbox_x2"] - parent_rows["bbox_x1"]) * (
        parent_rows["bbox_y2"] - parent_rows["bbox_y1"]
    )
    child_areas = (child_rows["bbox_x2"] - child_rows["bbox_x1"]) * (
        child_rows["bbox_y2"] - child_rows["bbox_y1"]
    )
    # After re-link, every parent row should carry the larger bbox area.
    assert parent_areas.min() > child_areas.max()


def test_filter_tracks_relink_propagates_to_pose(tmp_path):
    """The same swap correction must be applied to pose_3d_filtered.csv."""
    from video_analysis.track_filter import filter_tracks

    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)

    df = _swap_tracks_df_with_crossover(n_frames=300, crossover_frame=150)
    df.to_csv(cam_dir / "tracks_2d.csv", index=False)
    # Pose CSV: one keypoint per (frame, track) carrying a track-identifying
    # coordinate so we can check the swap got applied.
    pose_rows = []
    for _, r in df.iterrows():
        pose_rows.append(
            {
                "frame_idx": int(r["frame_idx"]),
                "timestamp_s": float(r["timestamp_s"]),
                "track_id": int(r["track_id"]),
                "kp_idx": 0,
                "x": float(r["bbox_x2"]) - float(r["bbox_x1"]),  # carries size
                "y": 0.0,
                "z": 0.0,
                "confidence": 0.9,
            }
        )
    pd.DataFrame(pose_rows).to_csv(cam_dir / "pose_3d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[BlockDef("test_block", "0:00", "0:10")],
        source_fps=30.0,
        n_keep=2,
    )
    filter_tracks(cfg)

    filtered_pose = pd.read_csv(cam_dir / "pose_3d_filtered.csv")
    parent_x = filtered_pose[filtered_pose["track_id"] == 0]["x"]
    child_x = filtered_pose[filtered_pose["track_id"] == 1]["x"]
    # x carries bbox size; after re-link parent_x should always be larger.
    assert parent_x.min() > child_x.max()


def test_filter_tracks_no_relink_when_size_signal_too_weak(tmp_path):
    """If parent and child have similar bbox sizes, relink should not act."""
    from video_analysis.track_filter import filter_tracks

    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)

    # Two tracks with very similar areas (110 vs 100) — below the 1.3 ratio
    # gate. Even with a synthetic crossover, the filter shouldn't intervene.
    df = _swap_tracks_df_with_crossover(
        n_frames=300, crossover_frame=150, parent_size=110.0, child_size=100.0
    )
    df.to_csv(cam_dir / "tracks_2d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[BlockDef("test_block", "0:00", "0:10")],
        source_fps=30.0,
        n_keep=2,
    )
    summary = filter_tracks(cfg)

    assert summary["blocks"][0]["relink_swaps"] == []


# ---------------------------------------------------------------------------
# Tests: session-wide parent/child role anchors
# ---------------------------------------------------------------------------


def _two_block_session_with_local_size_flip() -> pd.DataFrame:
    """Two blocks where block-local size ranking would flip parent/child.

    Block 1 (0-5 s): track A bigger than track B (200 vs 100).
    Block 2 (5-10 s): track A SMALLER than track B locally (60 vs 130),
        e.g. because the parent crouched and the child stood close to the
        camera. But across the whole session, track A's median area is
        still larger than track B's, so the global classifier should
        still pick A as parent.
    """
    rows = []
    for f in range(150):  # 5s at 30 fps
        # Block 1: A is large, B is small
        for tid, size in ((1, 200), (2, 100)):
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / 30.0,
                    "track_id": tid,
                    "track_label": f"person_{tid:02d}",
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": float(size),
                    "bbox_y2": float(size),
                    "track_confidence": 0.9,
                }
            )
    for f in range(150, 300):  # next 5s
        # Block 2: A briefly smaller than B in this block
        for tid, size in ((1, 60), (2, 130)):
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / 30.0,
                    "track_id": tid,
                    "track_label": f"person_{tid:02d}",
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": float(size),
                    "bbox_y2": float(size),
                    "track_confidence": 0.9,
                }
            )
    return pd.DataFrame(rows)


def test_global_anchors_keep_role_consistent_across_blocks(tmp_path):
    """Block-local size ranking would flip parent/child mid-session; the
    global anchor should keep track A as parent in both blocks because its
    session-wide median area is larger."""
    from video_analysis.track_filter import filter_tracks

    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)

    df = _two_block_session_with_local_size_flip()
    df.to_csv(cam_dir / "tracks_2d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[
            BlockDef("first", "0:00", "0:05"),
            BlockDef("second", "0:05", "0:10"),
        ],
        source_fps=30.0,
        n_keep=2,
    )
    summary = filter_tracks(cfg)

    # Track 1 (originally "A") should be parent (new_id=0) in BOTH blocks.
    for blk in summary["blocks"]:
        parent_track = next(t for t in blk["tracks"] if t["new_id"] == 0)
        assert parent_track["original_id"] == 1, (
            f"block {blk['name']}: parent should be track 1 (session-wide "
            f"larger), got {parent_track['original_id']}"
        )


def test_global_anchors_opt_out_falls_back_to_block_local(tmp_path):
    """When use_global_role_anchors=False, the old block-local ranking
    is used and the parent/child roles flip on the second block."""
    from video_analysis.track_filter import filter_tracks

    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)

    df = _two_block_session_with_local_size_flip()
    df.to_csv(cam_dir / "tracks_2d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[
            BlockDef("first", "0:00", "0:05"),
            BlockDef("second", "0:05", "0:10"),
        ],
        source_fps=30.0,
        n_keep=2,
        use_global_role_anchors=False,
    )
    summary = filter_tracks(cfg)

    # Without global anchors, block 2's parent should be track 2 (bigger
    # within that block) — opposite of block 1.
    parents = [
        next(t["original_id"] for t in blk["tracks"] if t["new_id"] == 0)
        for blk in summary["blocks"]
    ]
    assert parents == [1, 2], f"expected role flip without anchors, got {parents}"


def test_global_anchors_fall_back_when_block_has_only_one_class(tmp_path):
    """Block where only one global class is present must still assign both
    roles, falling back to block-local mean area for the missing one."""
    from video_analysis.track_filter import filter_tracks

    session_dir = tmp_path / "session"
    cam_dir = session_dir / "camera_a"
    cam_dir.mkdir(parents=True)

    # Track 1 is consistently large (parent_like globally).
    # Track 2 is consistently small (child_like globally).
    # Track 3 has medium size but only appears in block 2 (no global class
    # because it ties to one side — but specifically, in block 2 the parent
    # (track 1) is absent and we must still assign role=0.
    rows = []
    for f in range(150):
        for tid, size in ((1, 200), (2, 100)):
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / 30.0,
                    "track_id": tid,
                    "track_label": f"person_{tid:02d}",
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": float(size),
                    "bbox_y2": float(size),
                    "track_confidence": 0.9,
                }
            )
    # Block 2: only track 2 (child_like) and a new track 3 (also child-sized).
    for f in range(150, 300):
        for tid, size in ((2, 100), (3, 80)):
            rows.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / 30.0,
                    "track_id": tid,
                    "track_label": f"person_{tid:02d}",
                    "bbox_x1": 0.0,
                    "bbox_y1": 0.0,
                    "bbox_x2": float(size),
                    "bbox_y2": float(size),
                    "track_confidence": 0.9,
                }
            )
    pd.DataFrame(rows).to_csv(cam_dir / "tracks_2d.csv", index=False)

    cfg = TrackFilterConfig(
        session_dir=str(session_dir),
        camera="camera_a",
        blocks=[
            BlockDef("first", "0:00", "0:05"),
            BlockDef("second", "0:05", "0:10"),
        ],
        source_fps=30.0,
        n_keep=2,
    )
    summary = filter_tracks(cfg)

    # Block 2 should still pick a parent (role 0) and a child (role 1) — the
    # parent role falls back to the block-local larger track.
    block2 = summary["blocks"][1]
    roles = sorted(t["new_id"] for t in block2["tracks"])
    assert roles == [0, 1], f"expected both roles assigned, got {roles}"
