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


def test_report_shows_body_synchrony_with_chance(tmp_path: Path) -> None:
    """The synchrony column is the whole-body measure, shown with its chance level."""
    session = tmp_path / "1012"
    _write_camera(session, "camera_a", 0.9)
    summary_path = session / "camera_a" / "pose_metrics_summary.json"
    blocks = json.loads(summary_path.read_text(encoding="utf-8"))
    blocks[0].update(
        {
            "sync_mean": 0.21,
            "sync_chance_mean": 0.03,
            "sync_minus_chance": 0.18,
            "sync_n_windows": 10,
        }
    )
    summary_path.write_text(json.dumps(blocks), encoding="utf-8")

    html = render_report_html(session, collect_session_rows(session))

    assert "0.21" in html and "(chance 0.03)" in html
    assert "20-second" in html
    assert "0.50" not in html  # the old torso cross-correlation is not shown


# ---- Task-level table: parts rejoined (metrics-validity plan, phase 5) ----


def _write_split_session(session: Path) -> None:
    """camera_a: grocery split into two parts, storybook whole."""
    cam = session / "camera_a"
    cam.mkdir(parents=True)
    blocks = [
        ("grocery_part1", 100.0, 150.0, 300, 150, 0.3, 0.05, 4),
        ("grocery_part2", 200.0, 260.0, 360, 300, 0.1, 0.01, 6),
        ("storybook", 400.0, 500.0, 600, 540, 0.5, 0.00, 10),
    ]
    (cam / "pose_metrics_summary.json").write_text(
        json.dumps(
            [
                {
                    "block": name,
                    "block_start_s": start,
                    "block_end_s": end,
                    "block_frames": frames,
                    "frames_with_both": both,
                    "coverage_both": both / frames,
                    "low_coverage": both / frames < 0.5,
                    "sync_mean": sync,
                    "sync_chance_mean": chance,
                    "sync_minus_chance": sync - chance,
                    "sync_n_windows": n_win,
                }
                for name, start, end, frames, both, sync, chance, n_win in blocks
            ]
        ),
        encoding="utf-8",
    )
    # Per-frame distances: part1 all 10 px (150 frames), part2 all 40 px (300).
    prox = pd.DataFrame(
        {
            "frame_idx": range(450 + 540),
            "timestamp_s": range(450 + 540),
            "torso_distance_px": [10.0] * 150 + [40.0] * 300 + [25.0] * 540,
            "block": ["grocery_part1"] * 150
            + ["grocery_part2"] * 300
            + ["storybook"] * 540,
        }
    )
    prox.to_csv(cam / "pose_metrics_proximity.csv", index=False)
    sync_rows = []
    for name, *_rest, sync, _chance, n_win in blocks:
        sync_rows += [{"block": name, "sync_r": sync, "segment_id": 0}] * n_win
    pd.DataFrame(sync_rows).assign(window_start_s=0.0, window_end_s=20.0).to_csv(
        cam / "pose_metrics_sync.csv", index=False
    )
    (cam / "gaze_metrics_summary.json").write_text(
        json.dumps(
            [
                {
                    "block": "grocery_part1",
                    "n_frames": 100,
                    "mutual_gaze_pct": 10.0,
                    "mutual_gaze_chance_pct": 2.0,
                },
                {
                    "block": "grocery_part2",
                    "n_frames": 300,
                    "mutual_gaze_pct": 2.0,
                    "mutual_gaze_chance_pct": 2.0,
                },
                {
                    "block": "storybook",
                    "n_frames": 500,
                    "mutual_gaze_pct": 5.0,
                    "mutual_gaze_chance_pct": 1.0,
                },
            ]
        ),
        encoding="utf-8",
    )


def test_task_rows_rejoin_parts_by_pooling(tmp_path: Path) -> None:
    from video_analysis.session_report import collect_task_rows

    session = tmp_path / "1019"
    _write_split_session(session)

    tasks = collect_task_rows(session).set_index("task")

    g = tasks.loc["grocery"]
    assert g["n_parts"] == 2
    assert g["parts"] == "grocery_part1;grocery_part2"
    assert g["duration_s"] == 110.0
    assert g["frames_with_both"] == 450 and g["block_frames"] == 660
    assert abs(g["coverage_both"] - 450 / 660) < 1e-9
    # Pooled frames: (150*10 + 300*40) / 450 = 30; median of pooled frames = 40
    # (averaging the two part medians would wrongly give 25).
    assert abs(g["proximity_mean"] - 30.0) < 1e-9
    assert g["proximity_median"] == 40.0
    assert g["proximity_n_frames"] == 450
    # Pooled windows: (4*0.3 + 6*0.1) / 10 = 0.18; chance weighted the same way.
    assert abs(g["sync_mean"] - 0.18) < 1e-9
    assert g["sync_n_windows"] == 10
    assert abs(g["sync_chance_mean"] - (4 * 0.05 + 6 * 0.01) / 10) < 1e-9
    assert abs(g["sync_minus_chance"] - (g["sync_mean"] - g["sync_chance_mean"])) < 1e-9
    # Gaze shares weighted by gaze frames: (100*10 + 300*2) / 400 = 4.
    assert abs(g["gaze_mutual_gaze_pct"] - 4.0) < 1e-9
    assert abs(g["gaze_mutual_gaze_chance_pct"] - 2.0) < 1e-9
    assert g["gaze_n_frames"] == 400

    s = tasks.loc["storybook"]
    assert s["n_parts"] == 1 and s["proximity_median"] == 25.0
    assert abs(s["sync_mean"] - 0.5) < 1e-9


def test_write_session_outputs_writes_task_table_and_study_table(
    tmp_path: Path,
) -> None:
    from video_analysis.session_report import (
        ALL_SESSIONS_TASK_CSV,
        TASK_METRICS_CSV,
        build_all_sessions_task_table,
    )

    _write_split_session(tmp_path / "1019")
    _write_split_session(tmp_path / "1020")
    for sid in ("1019", "1020"):
        write_session_outputs(tmp_path / sid)
        assert (tmp_path / sid / TASK_METRICS_CSV).exists()
    (tmp_path / "_archive").mkdir()  # never a session

    study = build_all_sessions_task_table(tmp_path)

    assert sorted(study["session_id"].unique()) == ["1019", "1020"]
    assert len(study) == 4  # 2 sessions x (grocery, storybook)
    assert ALL_SESSIONS_TASK_CSV == "all_sessions_task_metrics.csv"


def test_cli_writes_study_task_table(tmp_path: Path) -> None:
    from video_analysis.session_report import ALL_SESSIONS_TASK_CSV, main

    _write_split_session(tmp_path / "1019")
    main(
        ["--session-dir", str(tmp_path / "1019"), "--all-sessions-root", str(tmp_path)]
    )

    study = pd.read_csv(tmp_path / ALL_SESSIONS_TASK_CSV)
    assert sorted(study["task"]) == ["grocery", "storybook"]


def test_task_rows_carry_pooled_verdict(tmp_path: Path) -> None:
    from video_analysis.session_report import collect_task_rows

    session = tmp_path / "1019"
    _write_split_session(session)
    tasks = collect_task_rows(session).set_index("task")

    assert "quality_verdict" in tasks.columns
    # grocery pools 4 + 6 = 10 synchrony windows and 450 both-tracked frames.
    assert tasks.loc["grocery", "sync_n_windows"] == 10


def test_report_hides_synchrony_with_too_few_windows(tmp_path: Path) -> None:
    session = tmp_path / "1019"
    _write_camera(session, "camera_a", 0.9)
    path = session / "camera_a" / "pose_metrics_summary.json"
    blocks = json.loads(path.read_text(encoding="utf-8"))
    blocks[0].update({"sync_mean": 0.77, "sync_chance_mean": 0.01, "sync_n_windows": 1})
    path.write_text(json.dumps(blocks), encoding="utf-8")

    html = render_report_html(session, collect_session_rows(session))

    assert "0.77" not in html
    assert "too few" in html


def test_study_tables_mark_metrics_version(tmp_path: Path) -> None:
    """New rows carry the current metrics version; older tables get a marker."""
    from video_analysis.session_report import METRICS_VERSION

    _write_split_session(tmp_path / "1019v4")
    write_session_outputs(tmp_path / "1019v4")
    old = tmp_path / "1012v3"
    old.mkdir()
    pd.DataFrame(
        [{"session_id": "1012v3", "camera_id": "camera_a", "block": "grocery"}]
    ).to_csv(old / "session_metrics.csv", index=False)

    study = build_all_sessions_table(tmp_path).set_index("session_id")

    assert study.loc["1019v4", "metrics_version"].iloc[0] == METRICS_VERSION
    assert study.loc["1012v3", "metrics_version"] == "before-2026-11-16"


# ---- report.html as the results page (portal-revamp plan, phase 6) ----


def _split_session_with_valid_time(session: Path) -> None:
    """The split session, with valid time per block and overlay clips."""
    _write_split_session(session)
    cam = session / "camera_a"
    path = cam / "pose_metrics_summary.json"
    blocks = json.loads(path.read_text(encoding="utf-8"))
    for block in blocks:
        block["valid_seconds_both"] = {
            "grocery_part1": 50.0,
            "grocery_part2": 100.0,
            "storybook": 90.0,
        }[block["block"]]
    path.write_text(json.dumps(blocks), encoding="utf-8")
    for name in ("grocery_part1", "grocery_part1_end", "storybook", "storybook_end"):
        (cam / f"overlay_{name}.mp4").write_bytes(b"")


def test_report_task_table_has_verdict_chips_and_reasons(tmp_path: Path) -> None:
    session = tmp_path / "1019"
    _split_session_with_valid_time(session)

    html = render_report_html(session, collect_session_rows(session))

    assert 'class="task-table' in html
    # grocery: 150 s valid, 450/660 = 68% coverage, 10 windows -> usable.
    assert "chip--ok" in html and "Usable" in html
    assert "2 min 30 s" in html  # pooled usable time of the two pieces
    assert "1 min 30 s" in html  # storybook
    # Verdict text comes with a symbol, not color alone.
    assert 'class="chip__icon" aria-hidden="true">✓<' in html
    # Pieces are joined and the recommended inclusion rule is stated.
    assert "joined" in html and "Recommended rule" in html
    # Per-piece table kept, collapsed.
    assert "<details" in html and "Per piece" in html
    assert "grocery_part2" in html


def test_report_task_table_shows_check_reasons_in_plain_words(tmp_path: Path) -> None:
    session = tmp_path / "1019"
    _split_session_with_valid_time(session)
    path = session / "camera_a" / "pose_metrics_summary.json"
    blocks = json.loads(path.read_text(encoding="utf-8"))
    for block in blocks:
        if block["block"] == "storybook":
            block["valid_seconds_both"] = 30.0
    path.write_text(json.dumps(blocks), encoding="utf-8")

    html = render_report_html(session, collect_session_rows(session))

    assert "chip--insufficient" in html and "Not enough data" in html
    assert "Both people tracked for only 30 s" in html


def test_report_task_table_hides_synchrony_below_minimum_windows(
    tmp_path: Path,
) -> None:
    session = tmp_path / "1019"
    _split_session_with_valid_time(session)
    sync = pd.read_csv(session / "camera_a" / "pose_metrics_sync.csv")
    # storybook keeps only 2 of its 10 windows.
    keep = sync[sync["block"] != "storybook"]
    story = sync[sync["block"] == "storybook"].head(2)
    pd.concat([keep, story]).to_csv(
        session / "camera_a" / "pose_metrics_sync.csv", index=False
    )

    html = render_report_html(session, collect_session_rows(session))
    task_table = html.split('class="task-table')[1].split("</table>")[0]

    assert "0.18 (chance 0.03)" in task_table.replace(
        ' <span class="muted">', " "
    ).replace("</span>", "")
    assert "too few unbroken 20-second stretches" in task_table
    assert "0.50" not in task_table


def test_report_embeds_clips_with_relative_sources(tmp_path: Path) -> None:
    session = tmp_path / "1019"
    _split_session_with_valid_time(session)

    html = render_report_html(session, collect_session_rows(session))

    assert '<video controls preload="none" src="camera_a/overlay_storybook.mp4"' in html
    assert 'src="camera_a/overlay_storybook_end.mp4"' in html
    assert 'href="camera_a/overlay_storybook.mp4" download' in html
    # Grouped by block: storybook start and end clips under one heading.
    block_html = html.split('data-block="storybook"')[1].split("data-block=")[0]
    assert "overlay_storybook_end.mp4" in block_html


def test_report_is_self_contained(tmp_path: Path) -> None:
    import re

    session = tmp_path / "1019"
    _split_session_with_valid_time(session)

    html = render_report_html(session, collect_session_rows(session))

    assert not re.search(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", html)
    assert "@import" not in html and "<link" not in html
    assert "--color-accent" in html  # the portal's tokens, inlined


def test_task_rows_pool_body_scaled_proximity(tmp_path: Path) -> None:
    from video_analysis.session_report import collect_task_rows

    session = tmp_path / "1019"
    _write_split_session(session)
    prox_path = session / "camera_a" / "pose_metrics_proximity.csv"
    prox = pd.read_csv(prox_path)
    # grocery part1 torsos 20/10 px, part2 60/30 px; storybook 50/30 px.
    prox["parent_torso_px"] = [20.0] * 150 + [60.0] * 300 + [50.0] * 540
    prox["child_torso_px"] = [10.0] * 150 + [30.0] * 300 + [30.0] * 540
    prox.to_csv(prox_path, index=False)

    tasks = collect_task_rows(session).set_index("task")

    # One ruler per camera, from all of its frames (not per task: sitting in
    # storybook shortens the torso on screen and would leak posture into the
    # task contrast). Median parent torso 50, child 30 -> 40 px for both tasks.
    g = tasks.loc["grocery"]
    assert g["body_scale_px"] == 40.0
    assert tasks.loc["storybook", "body_scale_px"] == 40.0
    assert abs(g["proximity_scaled_median"] - 40.0 / 40.0) < 1e-9
    assert abs(g["proximity_scaled_mean"] - 30.0 / 40.0) < 1e-9
    # Task contrast in body sizes equals the contrast in pixels.
    s = tasks.loc["storybook"]
    ratio_px = g["proximity_median"] / s["proximity_median"]
    ratio_scaled = g["proximity_scaled_median"] / s["proximity_scaled_median"]
    assert abs(ratio_px - ratio_scaled) < 1e-9


def test_task_rows_without_torso_columns_have_no_scaled_proximity(
    tmp_path: Path,
) -> None:
    from video_analysis.session_report import collect_task_rows

    session = tmp_path / "1019"
    _write_split_session(session)  # older runs: no torso columns

    tasks = collect_task_rows(session)

    assert "proximity_scaled_median" not in tasks.columns
    assert tasks["proximity_median"].notna().all()


def test_task_chance_is_missing_when_a_part_with_windows_has_none(
    tmp_path: Path,
) -> None:
    """Never subtract a chance level that covers only some of the windows."""
    from video_analysis.session_report import collect_task_rows

    session = tmp_path / "1019"
    _write_split_session(session)
    summary_path = session / "camera_a" / "pose_metrics_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary:
        if row["block"] == "grocery_part1":
            row.pop("sync_chance_mean")
            row.pop("sync_minus_chance")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    g = collect_task_rows(session).set_index("task").loc["grocery"]

    assert abs(g["sync_mean"] - 0.18) < 1e-9  # raw mean keeps all windows
    assert pd.isna(g["sync_chance_mean"])
    assert pd.isna(g["sync_minus_chance"])
