"""Tests for portal/run_log.py: per-job run records for troubleshooting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from gaze_analysis.config import SessionBlock
from portal import run_log
from portal.jobs import CommandStep, PortalJobConfig, run_command_step


def _config(tmp_path: Path) -> PortalJobConfig:
    job_dir = tmp_path / "jobs" / "job-1"
    job_dir.mkdir(parents=True)
    return PortalJobConfig(
        job_id="job-1",
        session_id="1021",
        video_a=tmp_path / "a.mp4",
        video_b=None,
        job_dir=job_dir,
        output_root=tmp_path / "output",
        session_config_path=job_dir / "session_config.json",
        blocks=[SessionBlock("grocery", 0.0, 600.0, "gray")],
    )


def _write_timings(
    job_dir: Path, rows: list[tuple[str, str | None, float, float, bool]]
):
    for name, camera, start, end, ok in rows:
        run_log.record_step(job_dir, name, camera, start, end, ok)


def test_run_command_step_records_timing(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    step = CommandStep(
        name="pose metrics camera_a",
        argv=[sys.executable, "-c", "print('hi')"],
        camera_id="camera_a",
    )

    run_command_step(step, cwd=tmp_path, log_path=job_dir / "job.log")

    lines = (job_dir / run_log.STEP_TIMINGS).read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1])
    assert rec["step"] == "pose metrics camera_a"
    assert rec["camera"] == "camera_a"
    assert rec["ok"] is True
    assert rec["seconds"] >= 0


def test_run_command_step_records_failure(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    step = CommandStep(name="boom", argv=[sys.executable, "-c", "raise SystemExit(3)"])

    with pytest.raises(Exception):
        run_command_step(step, cwd=tmp_path, log_path=job_dir / "job.log")

    rec = json.loads(
        (job_dir / run_log.STEP_TIMINGS).read_text(encoding="utf-8").splitlines()[-1]
    )
    assert rec["step"] == "boom" and rec["ok"] is False


def test_write_run_record_complete(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(run_log, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(run_log, "_gpu_names", lambda: ["NVIDIA RTX A6000"])
    (config.job_dir / "submission_metadata.json").write_text(
        json.dumps(
            {
                "created_at": "2026-11-17T10:00:00",
                "upload_finished_at": "2026-11-17T10:05:00",
                "videos": {
                    "camera_a": {
                        "original_filename": "1021_A.mp4",
                        "size_bytes": 30_000_000,
                        "duration_s": 2500.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    frames = config.session_output_dir / "camera_a" / "frames"
    frames.mkdir(parents=True)
    (frames / "extraction_info.json").write_text(
        json.dumps({"source_fps": 10.0, "achieved_frame_rate": 5.0}), encoding="utf-8"
    )
    (frames / "frame_index.csv").write_text(
        "frame_idx,timestamp_s,image_name\n0,0.0,a.jpg\n1,0.2,b.jpg\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "camera_id": ["camera_a"],
            "task": ["grocery"],
            "valid_seconds_both": [400.0],
            "quality_verdict": ["usable"],
        }
    ).to_csv(config.session_output_dir / "task_metrics.csv", index=False)
    t0 = 1_800_000_000.0
    _write_timings(
        config.job_dir,
        [
            ("pose inference", None, t0, t0 + 120, True),
            # 10 minutes waiting for the parent labels
            ("filter tracks camera_a", "camera_a", t0 + 720, t0 + 760, True),
            ("session report", None, t0 + 800, t0 + 810, True),
        ],
    )

    record = run_log.write_run_record(config, "complete")

    assert record["state"] == "complete"
    assert record["code"]["git_commit"] == "abc123"
    assert record["machine"]["gpus"] == ["NVIDIA RTX A6000"]
    assert record["videos"]["camera_a"]["size_mb"] == pytest.approx(30.0)
    assert record["videos"]["camera_a"]["achieved_frame_rate"] == 5.0
    assert record["videos"]["camera_a"]["frames_extracted"] == 2
    assert record["durations_s"]["upload"] == pytest.approx(300.0)
    assert record["durations_s"]["label_wait"] == pytest.approx(600.0)
    assert record["durations_s"]["phase2"] == pytest.approx(90.0)
    assert record["durations_s"]["processing"] == pytest.approx(210.0)
    assert record["results"] == [
        {
            "camera_id": "camera_a",
            "task": "grocery",
            "valid_seconds_both": 400.0,
            "quality_verdict": "usable",
        }
    ]
    assert (
        json.loads((config.job_dir / "run_info.json").read_text())["job_id"] == "job-1"
    )
    assert (config.session_output_dir / "run_info.json").exists()
    line = (config.output_root / run_log.RUNS_JSONL).read_text().splitlines()[-1]
    assert json.loads(line)["session_id"] == "1021"


def test_write_run_record_failed_keeps_error_and_log_tail(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (config.job_dir / "job.log").write_text(
        "\n".join(f"line {i}" for i in range(100)), encoding="utf-8"
    )
    error = RuntimeError("ffmpeg exploded")
    error.portal_step_name = "pose inference"  # type: ignore[attr-defined]

    record = run_log.write_run_record(config, "failed", error=error)

    assert record["state"] == "failed"
    assert record["failed_step"] == "pose inference"
    assert "ffmpeg exploded" in record["error"]
    assert record["log_tail"][-1] == "line 99" and len(record["log_tail"]) == 30


def test_write_run_record_never_raises(tmp_path: Path, monkeypatch) -> None:
    """Logging must never break a job, even if building the record fails."""
    config = _config(tmp_path)
    monkeypatch.setattr(
        run_log, "_build_record", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )

    assert run_log.write_run_record(config, "complete") == {}
