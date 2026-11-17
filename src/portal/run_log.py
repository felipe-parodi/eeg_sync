"""Per-job run records for troubleshooting and future optimization.

Every command step appends its start/end time to ``<job_dir>/step_timings.jsonl``.
When a job finishes (or fails), :func:`write_run_record` gathers what is
useful later -- inputs, timings per step and per phase (with the wait for the
parent labels kept separate), code/model versions, GPU, frame counts, the
outcome and per-task verdicts -- and writes it to:

- ``<job_dir>/run_info.json``,
- ``<session_output_dir>/run_info.json`` (so it ships in the results zip),
- one line in ``<output_root>/portal_runs.jsonl``, which survives portal
  restarts (job folders do not).

Recording never raises: a logging problem must not fail a job.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from video_inference.paths import REPO_ROOT

STEP_TIMINGS = "step_timings.jsonl"
RUN_INFO = "run_info.json"
RUNS_JSONL = "portal_runs.jsonl"
RECORD_VERSION = 1
LOG_TAIL_LINES = 30
# Step names (prefixes) that run before the parent labels; the rest run after.
PHASE1_PREFIXES = ("compression", "pose inference", "extract anchors")

_LOCK = threading.Lock()


def record_step(
    job_dir: Path,
    step: str,
    camera: str | None,
    started: float,
    ended: float,
    ok: bool,
) -> None:
    """Append one step's timing to the job's step log (never raises).

    Args:
        job_dir: Job directory.
        step: Step name.
        camera: Camera the step belongs to, if any.
        started: Start time (epoch seconds).
        ended: End time (epoch seconds).
        ok: Whether the step succeeded.
    """
    try:
        line = json.dumps(
            {
                "step": step,
                "camera": camera,
                "started": started,
                "ended": ended,
                "seconds": round(ended - started, 3),
                "ok": ok,
            }
        )
        with _LOCK, (Path(job_dir) / STEP_TIMINGS).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001 - logging must never fail a job
        pass


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_timings(job_dir: Path) -> list[dict]:
    path = job_dir / STEP_TIMINGS
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return sorted(rows, key=lambda r: r.get("started", 0))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_names() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _seconds_between(a: str | None, b: str | None) -> float | None:
    try:
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    except (TypeError, ValueError):
        return None


def _phase_durations(timings: list[dict]) -> dict[str, float | None]:
    """Phase 1, label wait, phase 2 and total processing time, in seconds."""
    phase1 = [t for t in timings if str(t.get("step", "")).startswith(PHASE1_PREFIXES)]
    phase2 = [t for t in timings if t not in phase1]

    def span(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return max(r["ended"] for r in rows) - min(r["started"] for r in rows)

    wait = None
    if phase1 and phase2:
        wait = max(
            0.0, min(r["started"] for r in phase2) - max(r["ended"] for r in phase1)
        )
    busy = [s for s in (span(phase1), span(phase2)) if s is not None]
    return {
        "phase1": span(phase1),
        "label_wait": wait,
        "phase2": span(phase2),
        "processing": sum(busy) if busy else None,
    }


def _video_info(config: Any, meta: dict) -> dict[str, dict]:
    videos: dict[str, dict] = {}
    for camera_id, info in (meta.get("videos") or {}).items():
        entry: dict[str, Any] = {
            "file_name": info.get("original_filename"),
            "size_mb": round(float(info.get("size_bytes", 0)) / 1e6, 2),
            "duration_s": info.get("duration_s"),
        }
        frames = Path(config.session_output_dir) / camera_id / "frames"
        extraction = _read_json(frames / "extraction_info.json")
        entry["source_fps"] = extraction.get("source_fps")
        entry["achieved_frame_rate"] = extraction.get("achieved_frame_rate")
        index = frames / "frame_index.csv"
        if index.exists():
            with index.open(encoding="utf-8") as fh:
                entry["frames_extracted"] = max(0, sum(1 for _ in fh) - 1)
        videos[camera_id] = entry
    return videos


def _task_results(config: Any) -> list[dict]:
    path = Path(config.session_output_dir) / "task_metrics.csv"
    if not path.exists():
        return []
    import pandas as pd

    table = pd.read_csv(path)
    cols = [
        c
        for c in ("camera_id", "task", "valid_seconds_both", "quality_verdict")
        if c in table.columns
    ]
    return table[cols].to_dict(orient="records")


def _log_tail(job_dir: Path) -> list[str]:
    path = job_dir / "job.log"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-LOG_TAIL_LINES:]


def _build_record(config: Any, state: str, error: BaseException | None) -> dict:
    job_dir = Path(config.job_dir)
    meta = _read_json(job_dir / "submission_metadata.json")
    timings = _read_timings(job_dir)
    from video_analysis.session_report import METRICS_VERSION

    record: dict[str, Any] = {
        "record_version": RECORD_VERSION,
        "job_id": config.job_id,
        "session_id": config.session_id,
        "state": state,
        "created_at": meta.get("created_at"),
        "upload_finished_at": meta.get("upload_finished_at"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "code": {
            "git_commit": _git_commit(),
            "metrics_version": METRICS_VERSION,
            "pose_model": Path(str(config.model_path)).name,
            "gaze_model": config.gaze_model if config.include_gaze else None,
            "frame_rate": config.frame_rate,
        },
        "machine": {"gpus": _gpu_names()},
        "videos": _video_info(config, meta),
        "tasks_submitted": {
            camera_id: [b["name"] for b in blocks]
            for camera_id, blocks in (
                meta.get("analysis_blocks_by_camera") or {}
            ).items()
        },
        "dropped_parts": meta.get("dropped_parts_by_camera"),
        "durations_s": {
            "upload": _seconds_between(
                meta.get("created_at"), meta.get("upload_finished_at")
            ),
            **_phase_durations(timings),
        },
        "steps": [
            {k: r.get(k) for k in ("step", "camera", "seconds", "ok")} for r in timings
        ],
        "results": _task_results(config) if state == "complete" else [],
    }
    if error is not None:
        record["failed_step"] = getattr(error, "portal_step_name", None)
        record["error"] = f"{type(error).__name__}: {error}"[-2000:]
        record["log_tail"] = _log_tail(job_dir)
    return record


def write_run_record(
    config: Any, state: str, error: BaseException | None = None
) -> dict[str, Any]:
    """Write the job's run record (job dir, session dir, portal_runs.jsonl).

    Args:
        config: The job's ``PortalJobConfig``.
        state: ``complete`` or ``failed``.
        error: The exception for a failed job.

    Returns:
        The record, or ``{}`` if it could not be built (never raises).
    """
    try:
        record = _build_record(config, state, error)
        text = json.dumps(record, indent=2, default=str)
        (Path(config.job_dir) / RUN_INFO).write_text(text, encoding="utf-8")
        session_dir = Path(config.session_output_dir)
        if session_dir.is_dir():
            (session_dir / RUN_INFO).write_text(text, encoding="utf-8")
        output_root = Path(config.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        with _LOCK, (output_root / RUNS_JSONL).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record
    except Exception:  # noqa: BLE001 - logging must never fail a job
        return {}


def now() -> float:
    """Current time in epoch seconds (wrapped for tests)."""
    return time.time()
