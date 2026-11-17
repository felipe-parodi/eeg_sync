"""Tests for frame sampling, deduplication, and participant-level splitting."""

from __future__ import annotations

import numpy as np
import pytest

from pose_training import sampling


def test_parse_video_name_numeric_participant() -> None:
    meta = sampling.parse_video_name("1010_FULL_SHORT_compressed.mp4")
    assert meta.participant == "1010"
    assert meta.angle == "SHORT"


def test_parse_video_name_pcode_participant() -> None:
    meta = sampling.parse_video_name("P001_FULL_TALL_compressed.mp4")
    assert meta.participant == "P001"
    assert meta.angle == "TALL"


def test_parse_video_name_rejects_unexpected() -> None:
    with pytest.raises(ValueError):
        sampling.parse_video_name("random_clip.mp4")


def test_evenly_spaced_timestamps_count_and_margin() -> None:
    stamps = sampling.evenly_spaced_timestamps(
        duration_s=100.0, count=5, margin_frac=0.05
    )
    assert len(stamps) == 5
    assert stamps[0] >= 5.0
    assert stamps[-1] <= 95.0
    assert all(b > a for a, b in zip(stamps, stamps[1:]))


def test_evenly_spaced_timestamps_short_video_is_clamped() -> None:
    stamps = sampling.evenly_spaced_timestamps(
        duration_s=3.0, count=10, margin_frac=0.05
    )
    assert len(stamps) <= 10
    assert all(0.0 <= t <= 3.0 for t in stamps)


def test_dhash_identical_images_match() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    assert sampling.dhash(image) == sampling.dhash(image.copy())


def test_dhash_distinct_images_differ() -> None:
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    half = np.zeros((64, 64, 3), dtype=np.uint8)
    half[:, 32:] = 255
    assert sampling.hamming(sampling.dhash(black), sampling.dhash(half)) > 5


def test_dedup_drops_near_duplicates() -> None:
    # Three hashes: two identical, one far away.
    hashes = [0b0000, 0b0000, 0b1111]
    kept = sampling.dedup_indices(hashes, threshold=1)
    assert kept == [0, 2]


def test_assign_splits_is_participant_disjoint() -> None:
    participants = [f"P{i:03d}" for i in range(10)]
    splits = sampling.assign_splits(participants, n_test=3, n_val=2, seed=42)
    test = {p for p, s in splits.items() if s == "test"}
    val = {p for p, s in splits.items() if s == "val"}
    train = {p for p, s in splits.items() if s == "train"}
    assert len(test) == 3
    assert len(val) == 2
    assert len(train) == 5
    assert test.isdisjoint(val)
    assert test.isdisjoint(train)
    assert val.isdisjoint(train)


def test_assign_splits_is_deterministic() -> None:
    participants = [f"P{i:03d}" for i in range(10)]
    first = sampling.assign_splits(participants, n_test=3, n_val=2, seed=7)
    second = sampling.assign_splits(participants, n_test=3, n_val=2, seed=7)
    assert first == second


def test_timestamps_in_spans_stay_inside_spans() -> None:
    spans = [(100.0, 160.0), (300.0, 360.0)]
    stamps = sampling.timestamps_in_spans(spans, sample_fps=0.5, max_frames=100)
    assert stamps == sorted(stamps)
    assert all(
        any(a <= t <= b for a, b in spans) for t in stamps
    )  # never lands in the excluded gap (160-300)


def test_timestamps_in_spans_respects_cap_and_proportions() -> None:
    # 60s + 30s of footage; cap below the fps-implied count is honored.
    spans = [(0.0, 60.0), (100.0, 130.0)]
    stamps = sampling.timestamps_in_spans(
        spans, sample_fps=1.0, max_frames=9, margin_frac=0.0
    )
    assert len(stamps) == 9  # capped at max_frames
    # Roughly 2:1 split between the 60s and 30s spans.
    first = [t for t in stamps if t <= 60.0]
    assert len(first) == 6


def test_timestamps_in_spans_empty_when_no_footage() -> None:
    assert sampling.timestamps_in_spans([], sample_fps=1.0, max_frames=10) == []


def test_select_indices_modes() -> None:
    # Near-static run then a jump: hashes 0,0,0,15 (last is far).
    hashes = [0b0000, 0b0000, 0b0000, 0b1111]
    assert sampling.select_indices(hashes, "none", threshold=6) == [0, 1, 2, 3]
    # Consecutive: keep first, skip the two identical, keep the far one.
    assert sampling.select_indices(hashes, "consecutive", threshold=2) == [0, 3]
    # Global matches the legacy dedup.
    assert sampling.select_indices(
        hashes, "global", threshold=2
    ) == sampling.dedup_indices(hashes, 2)


def test_select_indices_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        sampling.select_indices([1, 2], "bogus", threshold=1)
