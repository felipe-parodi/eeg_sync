"""Recalculate finished sessions' numbers when the metric definitions change.

When the portal starts, every finished job whose results were made with older
metric definitions (``METRICS_VERSION``) gets its numbers recalculated from
the tracking it already has: pose metrics, the gaze dashboard, the session
report and study tables, and the results zip. Nothing is re-run from the
video (no inference, tracking or overlay clips), so this takes seconds per
session, and the colleague only has to download the results again.

Each job records the version its results were made with in
``metrics_version.txt``; the job page shows a note when numbers were
recalculated. A failed recalculation leaves the job as it was (still
finished, still marked outdated) and is retried at the next start.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import date
from pathlib import Path
from typing import Callable

from portal.jobs import (
    CommandStep,
    add_job_note,
    bundle_result_zip,
    read_status,
    run_command_step,
)
from video_analysis.pose_metrics import METRICS_VERSION

MARKER_NAME = "metrics_version.txt"
_refreshing = threading.Event()


def is_refreshing() -> bool:
    """True while finished sessions are being recalculated."""
    return _refreshing.is_set()


def write_marker(job_dir: Path) -> None:
    """Record that this job's results use the current metric definitions."""
    (Path(job_dir) / MARKER_NAME).write_text(METRICS_VERSION + "\n", encoding="utf-8")


def _session_id(job_dir: Path) -> str | None:
    try:
        config = json.loads((job_dir / "job_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    session_id = config.get("session_id")
    return str(session_id) if session_id else None


def _camera_dirs(session_dir: Path) -> list[Path]:
    if not session_dir.is_dir():
        return []
    return sorted(
        d
        for d in session_dir.iterdir()
        if d.is_dir()
        and d.name.startswith("camera_")
        and (d / "pose_3d_smooth.csv").exists()
    )


def needs_refresh(job_dir: Path, output_root: Path) -> bool:
    """True for a finished job whose results predate the current definitions.

    Args:
        job_dir: Portal job folder.
        output_root: Folder holding one folder per session.

    Returns:
        Whether the job's numbers should be recalculated.
    """
    job_dir = Path(job_dir)
    if read_status(job_dir).get("state") != "complete":
        return False
    session_id = _session_id(job_dir)
    if session_id is None or not (job_dir / "session_config.json").exists():
        return False
    if not _camera_dirs(Path(output_root) / session_id):
        return False
    marker = job_dir / MARKER_NAME
    current = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    return current != METRICS_VERSION


def refresh_steps(
    job_dir: Path, output_root: Path, python: str = sys.executable
) -> list[CommandStep]:
    """The numbers-only steps for one job, using current folder locations.

    Paths stored in an old ``job_config.json`` may point at an earlier folder
    layout, so everything is derived from where the job and session are now.

    Args:
        job_dir: Portal job folder.
        output_root: Folder holding one folder per session.
        python: Python interpreter to run the steps with.

    Returns:
        Pose metrics (and the gaze dashboard when gaze was run) per camera,
        then the session report.
    """
    job_dir = Path(job_dir)
    output_root = Path(output_root)
    session_dir = output_root / str(_session_id(job_dir))
    config_path = str(job_dir / "session_config.json")
    steps: list[CommandStep] = []
    for camera_dir in _camera_dirs(session_dir):
        camera_id = camera_dir.name
        common = [
            "--camera-dir",
            str(camera_dir),
            "--session-config",
            config_path,
            "--camera-id",
            camera_id,
        ]
        steps.append(
            CommandStep(
                name=f"pose metrics {camera_id}",
                argv=[
                    python,
                    "-m",
                    "video_analysis.pose_metrics",
                    *common,
                    "--pose-input",
                    "pose_3d_smooth.csv",
                ],
            )
        )
        if (camera_dir / "synchrony_metrics.csv").exists():
            steps.append(
                CommandStep(
                    name=f"gaze dashboard {camera_id}",
                    argv=[python, "-m", "gaze_analysis.plotting", *common],
                )
            )
    steps.append(
        CommandStep(
            name="session report",
            argv=[
                python,
                "-m",
                "video_analysis.session_report",
                "--session-dir",
                str(session_dir),
                "--all-sessions-root",
                str(output_root),
            ],
        )
    )
    return steps


def refresh_job(job_dir: Path, output_root: Path, repo_root: Path) -> None:
    """Recalculate one finished job's numbers and rebuild its results zip.

    Args:
        job_dir: Portal job folder.
        output_root: Folder holding one folder per session.
        repo_root: Working directory for the steps.

    Raises:
        Exception: Whatever a step raised; the job is then left unmarked.
    """
    job_dir = Path(job_dir)
    session_id = str(_session_id(job_dir))
    log_path = job_dir / "job.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n=== Recalculating numbers with metric definitions "
            f"{METRICS_VERSION} (no video processing) ===\n"
        )
    for step in refresh_steps(job_dir, output_root):
        run_command_step(step, cwd=repo_root, log_path=log_path)
    bundle_result_zip(job_dir, session_id, Path(output_root) / session_id)
    write_marker(job_dir)
    add_job_note(
        job_dir,
        f"Numbers recalculated on {date.today().isoformat()} with the updated "
        f"metric definitions ({METRICS_VERSION}). Download the results again "
        "to get them; tracking and video clips are unchanged.",
    )


def refresh_outdated_jobs(
    job_root: Path,
    output_root: Path,
    repo_root: Path,
    lock: threading.Lock | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Recalculate every finished job made with older metric definitions.

    Args:
        job_root: Folder holding the portal job folders.
        output_root: Folder holding one folder per session.
        repo_root: Working directory for the steps.
        lock: Held around each job, so recalculation never overlaps a real
            processing job.
        log: Where to print one line per job.

    Returns:
        Counts of jobs ``refreshed`` and ``failed``.
    """
    summary = {"refreshed": 0, "failed": 0}
    job_root = Path(job_root)
    if not job_root.is_dir():
        return summary
    jobs = [
        d
        for d in sorted(job_root.iterdir())
        if d.is_dir() and needs_refresh(d, output_root)
    ]
    if not jobs:
        return summary
    _refreshing.set()
    try:
        for job_dir in jobs:
            if lock is not None:
                lock.acquire()
            try:
                refresh_job(job_dir, output_root, repo_root)
                summary["refreshed"] += 1
                log(f"Recalculated numbers for {job_dir.name}.")
            except Exception as error:  # noqa: BLE001 - one job must not stop the rest
                summary["failed"] += 1
                log(
                    f"Could not recalculate {job_dir.name} ({error}); "
                    "its earlier results are kept and it is retried next start."
                )
            finally:
                if lock is not None:
                    lock.release()
    finally:
        _refreshing.clear()
    return summary
