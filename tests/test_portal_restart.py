"""Tests for portal restarts: jobs survive, and 'restart when idle'."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import portal.app as portal_app
from portal.jobs import read_status, write_status


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    job_root = tmp_path / "jobs"
    upload_root = tmp_path / "uploads"
    job_root.mkdir()
    upload_root.mkdir()
    monkeypatch.setattr(portal_app, "JOB_ROOT", job_root)
    monkeypatch.setattr(portal_app, "UPLOAD_ROOT", upload_root)
    return job_root, upload_root


def _job(job_root: Path, upload_root: Path, job_id: str, state: str) -> Path:
    job_dir = job_root / job_id
    job_dir.mkdir()
    write_status(job_dir, {"state": state, "message": "x"})
    (upload_root / job_id).mkdir()
    (upload_root / job_id / "camera_a.mp4").write_bytes(b"video")
    return job_dir


def test_startup_keeps_jobs_and_marks_interrupted_ones(roots) -> None:
    job_root, upload_root = roots
    done = _job(job_root, upload_root, "j_done", "complete")
    failed = _job(job_root, upload_root, "j_failed", "failed")
    waiting = _job(job_root, upload_root, "j_wait", "awaiting_annotation")
    running = _job(job_root, upload_root, "j_run", "running")
    queued = _job(job_root, upload_root, "j_queue", "queued")
    uploading = _job(job_root, upload_root, "j_up", "uploading")
    (upload_root / "orphan").mkdir()  # upload with no job folder
    loose_log = job_root / "server.out.log"
    loose_log.write_text("log", encoding="utf-8")

    summary = portal_app.tidy_jobs_on_startup()

    # Every job folder survives.
    for job_dir in (done, failed, waiting, running, queued, uploading):
        assert job_dir.exists()
    assert loose_log.exists()
    # Jobs that were in flight are marked interrupted, in plain words.
    for job_dir in (running, queued, uploading):
        status = read_status(job_dir)
        assert status["state"] == "failed"
        assert "restarted" in status["message"]
    assert read_status(done)["state"] == "complete"
    assert read_status(waiting)["state"] == "awaiting_annotation"
    # Uploaded videos are kept only where the job can still continue.
    assert (upload_root / "j_wait").exists()
    for job_id in ("j_done", "j_failed", "j_run", "j_queue", "j_up", "orphan"):
        assert not (upload_root / job_id).exists()
    assert summary == {"kept": 6, "interrupted": 3, "uploads_removed": 6}


def test_startup_handles_missing_job_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path / "missing")
    monkeypatch.setattr(portal_app, "UPLOAD_ROOT", tmp_path / "uploads")

    assert portal_app.tidy_jobs_on_startup() == {
        "kept": 0,
        "interrupted": 0,
        "uploads_removed": 0,
    }


@pytest.mark.parametrize(
    ("state", "idle"),
    [
        ("running", False),
        ("queued", False),
        ("uploading", False),
        ("awaiting_annotation", True),
        ("complete", True),
        ("failed", True),
    ],
)
def test_portal_is_idle(roots, state: str, idle: bool) -> None:
    job_root, upload_root = roots
    _job(job_root, upload_root, "j", state)

    assert portal_app.portal_is_idle() is idle


def test_abandoned_upload_does_not_block_restart(roots) -> None:
    job_root, upload_root = roots
    job_dir = _job(job_root, upload_root, "j", "uploading")
    old = time.time() - 2 * portal_app.STALE_UPLOAD_S
    os.utime(job_dir / "status.json", (old, old))

    assert portal_app.portal_is_idle() is True


def test_restart_request_waits_for_running_job(roots) -> None:
    job_root, upload_root = roots
    job_dir = _job(job_root, upload_root, "j", "running")
    portal_app.request_restart()

    assert portal_app.restart_due() is False
    assert portal_app.restart_flag_path().exists()  # still pending

    write_status(job_dir, {"state": "complete", "message": "done"})
    assert portal_app.restart_due() is True
    assert not portal_app.restart_flag_path().exists()  # consumed


def test_restart_cli_writes_and_cancels_request(roots, capsys) -> None:
    from portal import restart

    restart.main([])
    assert portal_app.restart_flag_path().exists()
    assert "restart" in capsys.readouterr().out.lower()

    restart.main(["--cancel"])
    assert not portal_app.restart_flag_path().exists()


def test_portal_is_not_idle_while_recalculating_numbers(roots) -> None:
    from portal import refresh

    refresh._refreshing.set()
    try:
        assert portal_app.portal_is_idle() is False
    finally:
        refresh._refreshing.clear()
    assert portal_app.portal_is_idle() is True


def test_startup_starts_recalculation_in_the_background(
    roots, monkeypatch: pytest.MonkeyPatch
) -> None:
    from portal import refresh

    calls: list[tuple] = []
    monkeypatch.setattr(
        refresh,
        "refresh_outdated_jobs",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or {"refreshed": 0, "failed": 0},
    )

    thread = portal_app.start_metrics_refresh()
    thread.join(timeout=5)

    assert calls and calls[0][0][0] == portal_app.JOB_ROOT
    assert calls[0][1]["lock"] is portal_app.JOB_RUN_LOCK
