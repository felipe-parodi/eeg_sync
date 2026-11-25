"""Recalculating finished sessions' numbers when metric definitions change."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from portal import refresh
from video_analysis.pose_metrics import METRICS_VERSION


def _finished_job(
    job_root: Path,
    output_root: Path,
    job_id: str = "20261121_180741_1018v4_abc",
    session_id: str = "1018v4",
    state: str = "complete",
    gaze: bool = True,
) -> Path:
    job_dir = job_root / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "status.json").write_text(json.dumps({"state": state}), "utf-8")
    # Stored paths are from an older folder layout on purpose.
    (job_dir / "job_config.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "job_dir": "C:/old/layout/portal_jobs/x",
                "session_config_path": "C:/old/layout/session_config.json",
            }
        ),
        "utf-8",
    )
    (job_dir / "session_config.json").write_text("{}", "utf-8")
    (job_dir / "job.log").write_text("", "utf-8")
    for cam in ("camera_a", "camera_b"):
        cam_dir = output_root / session_id / cam
        cam_dir.mkdir(parents=True)
        (cam_dir / "pose_3d_smooth.csv").write_text("x", "utf-8")
        if gaze:
            (cam_dir / "synchrony_metrics.csv").write_text("x", "utf-8")
    return job_dir


def test_finished_jobs_without_current_marker_need_refresh(tmp_path: Path) -> None:
    job = _finished_job(tmp_path / "jobs", tmp_path / "out")

    assert refresh.needs_refresh(job, tmp_path / "out")
    refresh.write_marker(job)
    assert not refresh.needs_refresh(job, tmp_path / "out")


def test_unfinished_or_orphaned_jobs_are_left_alone(tmp_path: Path) -> None:
    running = _finished_job(tmp_path / "jobs", tmp_path / "out", state="running")
    orphan = _finished_job(
        tmp_path / "jobs", tmp_path / "out", job_id="j2", session_id="gone"
    )
    for cam in ("camera_a", "camera_b"):
        (tmp_path / "out" / "gone" / cam / "pose_3d_smooth.csv").unlink()

    assert not refresh.needs_refresh(running, tmp_path / "out")
    assert not refresh.needs_refresh(orphan, tmp_path / "out")


def test_refresh_steps_use_current_locations_not_stored_paths(tmp_path: Path) -> None:
    job = _finished_job(tmp_path / "jobs", tmp_path / "out")

    steps = refresh.refresh_steps(job, tmp_path / "out", python="py")

    names = [s.name for s in steps]
    assert names == [
        "pose metrics camera_a",
        "gaze dashboard camera_a",
        "pose metrics camera_b",
        "gaze dashboard camera_b",
        "session report",
    ]
    joined = " ".join(" ".join(s.argv) for s in steps)
    assert "C:/old/layout" not in joined
    assert str(job / "session_config.json") in joined
    assert str(tmp_path / "out" / "1018v4") in joined
    # Nothing that touches the video: no inference, tracking or overlays.
    assert not any("infer" in a or "visualize" in a for s in steps for a in s.argv)


def test_refresh_steps_skip_dashboard_without_gaze(tmp_path: Path) -> None:
    job = _finished_job(tmp_path / "jobs", tmp_path / "out", gaze=False)

    names = [s.name for s in refresh.refresh_steps(job, tmp_path / "out", "py")]

    assert "gaze dashboard camera_a" not in names


def test_refresh_job_rebuilds_zip_marks_and_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _finished_job(tmp_path / "jobs", tmp_path / "out")
    ran: list[str] = []
    monkeypatch.setattr(
        refresh, "run_command_step", lambda step, cwd, log_path: ran.append(step.name)
    )

    refresh.refresh_job(job, tmp_path / "out", repo_root=tmp_path)

    assert ran[-1] == "session report"
    assert (job / "1018v4_results.zip").exists()
    assert not refresh.needs_refresh(job, tmp_path / "out")
    notes = json.loads((job / "status.json").read_text("utf-8"))
    assert notes["state"] == "complete"
    from portal.jobs import read_job_notes

    assert any("recalculated" in n for n in read_job_notes(job))


def test_failed_refresh_leaves_job_marked_outdated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _finished_job(tmp_path / "jobs", tmp_path / "out")

    def boom(step, cwd, log_path):
        raise RuntimeError("step failed")

    monkeypatch.setattr(refresh, "run_command_step", boom)

    summary = refresh.refresh_outdated_jobs(
        tmp_path / "jobs", tmp_path / "out", repo_root=tmp_path
    )

    assert summary == {"refreshed": 0, "failed": 1}
    assert refresh.needs_refresh(job, tmp_path / "out")
    assert json.loads((job / "status.json").read_text("utf-8"))["state"] == "complete"


def test_refresh_reports_busy_while_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _finished_job(tmp_path / "jobs", tmp_path / "out")
    seen: list[bool] = []
    monkeypatch.setattr(
        refresh,
        "run_command_step",
        lambda step, cwd, log_path: seen.append(refresh.is_refreshing()),
    )

    refresh.refresh_outdated_jobs(tmp_path / "jobs", tmp_path / "out", tmp_path)

    assert seen and all(seen)
    assert not refresh.is_refreshing()


def test_marker_holds_current_version(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()

    refresh.write_marker(job)

    assert (job / refresh.MARKER_NAME).read_text("utf-8").strip() == METRICS_VERSION
