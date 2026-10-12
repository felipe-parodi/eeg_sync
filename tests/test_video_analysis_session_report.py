"""Tests for the per-session metrics table and HTML report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from video_analysis.session_report import (
    build_all_sessions_table,
    collect_session_rows,
    render_report_html,
    write_session_outputs,
)


def _write_camera(session_dir: Path, camera_id: str, coverage: float) -> None:
    cam = session_dir / camera_id
    cam.mkdir(parents=True)
    (cam / "pose_metrics_summary.json").write_text(
        json.dumps(
            [
                {
                    "block": "grocery",
                    "block_start_s": 100.0,
                    "block_end_s": 200.0,
                    "block_frames": 601,
                    "frames_with_both": int(601 * coverage),
                    "coverage_both": coverage,
                    "low_coverage": coverage < 0.5,
                    "proximity_mean": 120.0,
                    "xcorr_mean": 0.5,
                }
            ]
        ),
        encoding="utf-8",
    )
    (cam / "gaze_metrics_summary.json").write_text(
        json.dumps([{"block": "grocery", "n_frames": 300, "mutual_gaze_pct": 2.0}]),
        encoding="utf-8",
    )
    (cam / "two_person_summary.json").write_text(
        json.dumps({"anchors_used": 10, "share_with_both": 0.7}), encoding="utf-8"
    )
    (cam / "overlay_grocery.mp4").write_bytes(b"")


def test_collect_session_rows_one_row_per_camera_and_block(tmp_path: Path) -> None:
    session = tmp_path / "1010"
    _write_camera(session, "camera_a", 0.9)
    _write_camera(session, "camera_b", 0.2)

    rows = collect_session_rows(session)

    assert list(rows["camera_id"]) == ["camera_a", "camera_b"]
    assert set(rows["session_id"]) == {"1010"}
    assert list(rows["block"]) == ["grocery", "grocery"]
    assert list(rows["low_coverage"]) == [False, True]
    assert list(rows["gaze_mutual_gaze_pct"]) == [2.0, 2.0]
    assert list(rows["tracking_anchors_used"]) == [10, 10]


def test_collect_session_rows_without_gaze(tmp_path: Path) -> None:
    session = tmp_path / "P003"
    _write_camera(session, "camera_a", 0.9)
    (session / "camera_a" / "gaze_metrics_summary.json").unlink()

    rows = collect_session_rows(session)

    assert len(rows) == 1
    assert not any(c.startswith("gaze_") for c in rows.columns)


def test_write_session_outputs_and_all_sessions_table(tmp_path: Path) -> None:
    for sid in ("1010", "1011"):
        _write_camera(tmp_path / sid, "camera_a", 0.8)
        write_session_outputs(tmp_path / sid)
    (tmp_path / "portal_jobs").mkdir()  # not a session

    assert (tmp_path / "1010" / "session_metrics.csv").exists()
    assert (tmp_path / "1010" / "report.html").exists()
    table = build_all_sessions_table(tmp_path)
    assert sorted(table["session_id"]) == ["1010", "1011"]


def test_report_flags_low_coverage_and_links_overlays(tmp_path: Path) -> None:
    session = tmp_path / "1012"
    _write_camera(session, "camera_a", 0.2)
    html = render_report_html(session, collect_session_rows(session))

    assert "1012" in html
    assert "Low coverage" in html
    assert 'href="camera_a/overlay_grocery.mp4"' in html


def test_all_sessions_table_empty_when_no_sessions(tmp_path: Path) -> None:
    table = build_all_sessions_table(tmp_path)
    assert isinstance(table, pd.DataFrame)
    assert table.empty
