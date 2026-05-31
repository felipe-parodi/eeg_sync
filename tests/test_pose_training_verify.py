"""Tests for the human-verify A/B harness (sampling + tally + review writes)."""

from __future__ import annotations

import csv
from pathlib import Path

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


def test_set_verdict_updates_matching_record_only() -> None:
    records = [
        {"id": "0", "verdict": ""},
        {"id": "1", "verdict": ""},
    ]
    assert verify.set_verdict(records, 1, "r") is True  # lowercase + int id
    assert records[1]["verdict"] == "R"
    assert records[0]["verdict"] == ""  # untouched


def test_set_verdict_rejects_bad_verdict_and_missing_id() -> None:
    records = [{"id": "0", "verdict": "L"}]
    assert verify.set_verdict(records, 0, "X") is False  # invalid letter
    assert records[0]["verdict"] == "L"  # unchanged
    assert verify.set_verdict(records, 99, "R") is False  # no such id


def test_write_records_round_trips_through_summarize(tmp_path: Path) -> None:
    csv_path = tmp_path / "verdicts.csv"
    records = [
        {
            "id": "0",
            "image": "a.png",
            "frame_path": "x",
            "left_is_a": "True",
            "verdict": "",
        },
        {
            "id": "1",
            "image": "b.png",
            "frame_path": "y",
            "left_is_a": "False",
            "verdict": "",
        },
    ]
    verify.set_verdict(records, "0", "L")  # left_is_a True -> A wins
    verify.set_verdict(records, "1", "L")  # left_is_a False -> B wins
    verify._write_records(csv_path, records)
    reloaded = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    result = verify.tally(reloaded, "fused3", "allsessions")
    assert result["fused3"] == 1
    assert result["allsessions"] == 1
    assert result["unjudged"] == 0


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
