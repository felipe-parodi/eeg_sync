"""Tests for the human-verify A/B harness (sampling + tally)."""

from __future__ import annotations

from pose_training import verify


def _rows(n_per):
    rows = []
    for participant in ("P001", "P002", "1010"):
        for angle in ("SHORT", "TALL"):
            for i in range(n_per):
                rows.append(
                    {
                        "frame_path": f"{participant}_{angle}_{i}.jpg",
                        "participant": participant,
                        "angle": angle,
                    }
                )
    return rows


def test_stratified_sample_spreads_across_sessions() -> None:
    rows = _rows(20)
    picked = verify.stratified_sample(rows, n=12, seed=0)
    assert len(picked) == 12
    sessions = {(r["participant"], r["angle"]) for r in picked}
    # All 6 session/angle groups represented in a 12-frame draw.
    assert len(sessions) == 6


def test_stratified_sample_is_deterministic() -> None:
    rows = _rows(20)
    a = verify.stratified_sample(rows, n=10, seed=3)
    b = verify.stratified_sample(rows, n=10, seed=3)
    assert [r["frame_path"] for r in a] == [r["frame_path"] for r in b]


def test_tally_maps_blind_sides_to_models() -> None:
    # left_is_a=True -> a verdict of "L" credits model A.
    records = [
        {"left_is_a": True, "verdict": "L"},  # A wins
        {"left_is_a": False, "verdict": "L"},  # B wins (A was on the right)
        {"left_is_a": True, "verdict": "R"},  # B wins
        {"left_is_a": True, "verdict": "T"},  # tie
        {"left_is_a": True, "verdict": ""},  # unjudged -> ignored
    ]
    tally = verify.tally(records, "fused3", "allsessions")
    assert tally["fused3"] == 1
    assert tally["allsessions"] == 2
    assert tally["tie"] == 1
    assert tally["unjudged"] == 1
