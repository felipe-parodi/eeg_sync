"""End-to-end smoke tests that actually run the inference pipeline.

These tests catch dependency drift, install gaps, and breaking-API changes
in third-party libraries — failures that pure-unit tests miss because they
mock out ffmpeg, ultralytics, torch, and the tracker.

Activation:
    Set RUN_SMOKE_TESTS=1 in the environment to run. Default `pytest`
    runs skip the entire module so day-to-day iteration stays fast.

PowerShell:
    $env:RUN_SMOKE_TESTS = "1"; uv run pytest tests/test_smoke_end_to_end.py

Each test additionally skips itself if its prerequisites are missing
(compressed video, ffmpeg, ultralytics) so the failure message points
at what to install rather than dumping an opaque traceback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_SAMPLE_VIDEO = (
    ROOT_DIR / "video_inference" / "compressed" / "P001c_Short_Full_rapid.mp4"
)
SMOKE_TIMEOUT_S = 600

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SMOKE_TESTS") != "1",
    reason="Smoke tests are slow and need real deps. Set RUN_SMOKE_TESTS=1 to enable.",
)


def _resolve_sample_video() -> Path:
    """Return the smoke sample, honoring SMOKE_VIDEO_PATH if set."""
    override = os.environ.get("SMOKE_VIDEO_PATH")
    return Path(override) if override else DEFAULT_SAMPLE_VIDEO


def _require_sample_video() -> Path:
    video = _resolve_sample_video()
    if not video.exists():
        pytest.skip(
            f"Compressed sample video not found at {video}. "
            "Set SMOKE_VIDEO_PATH to a small compressed mp4 to enable the smoke test."
        )
    return video


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not on PATH; required for frame extraction.")


def _require_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pytest.skip(
            "ultralytics is not installed. Run "
            "`uv sync --extra portal` (or `--extra video`) to enable."
        )


def test_video_infer_cli_runs_end_to_end_on_compressed_video(tmp_path: Path) -> None:
    """Run `video-infer run` on a real 2-second window and check expected outputs."""
    video = _require_sample_video()
    _require_ffmpeg()
    _require_ultralytics()

    output_dir = tmp_path / "output"
    session_id = "smoke"
    cmd = [
        sys.executable,
        "-m",
        "video_inference.pipeline",
        "run",
        "--video-a",
        str(video),
        "--inference-backend",
        "ultralytics",
        "--ultralytics-model-path",
        "yolo11m-pose.pt",
        "--tracker-backend",
        "roboflow",
        "--tracker-name",
        "bytetrack",
        "--max-persons",
        "4",
        "--frame-rate",
        "1.0",
        "--analysis-windows-camera-a",
        "smoke,0:00,0:02",
        "--device",
        "auto",
        "--skip-compress",
        "--output-dir",
        str(output_dir),
        "--session-id",
        session_id,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=SMOKE_TIMEOUT_S
    )
    assert result.returncode == 0, (
        f"video-infer exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    camera_dir = output_dir / session_id / "camera_a"
    assert (
        camera_dir / "tracks_2d.csv"
    ).exists(), f"tracks_2d.csv missing under {camera_dir}"
    assert (
        camera_dir / "pose_3d.csv"
    ).exists(), f"pose_3d.csv missing under {camera_dir}"
    assert (
        camera_dir / "manifest.json"
    ).exists(), f"manifest.json missing under {camera_dir}"


def test_run_portal_job_end_to_end_on_compressed_video(tmp_path: Path) -> None:
    """Exercise the portal's run_portal_job path against a real video."""
    video = _require_sample_video()
    _require_ffmpeg()
    _require_ultralytics()

    from portal.jobs import PortalJobConfig, SessionBlock, run_portal_job, write_status

    job_id = "smoke_job"
    session_id = "smoke"
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    output_root = tmp_path / "output"
    output_root.mkdir()
    session_config_path = job_dir / "session_config.json"

    # Copy the sample video into a staging dir so the test does not write
    # to the gitignored source tree.
    staged_video = tmp_path / "upload" / "camera_a.mp4"
    staged_video.parent.mkdir(parents=True)
    shutil.copy2(video, staged_video)

    blocks = [SessionBlock(name="smoke", start_s=0.0, end_s=2.0, color="gray")]
    config = PortalJobConfig(
        job_id=job_id,
        session_id=session_id,
        email="",
        video_a=staged_video,
        video_b=None,
        job_dir=job_dir,
        output_root=output_root,
        session_config_path=session_config_path,
        blocks=blocks,
        blocks_by_camera={"camera_a": blocks},
        include_gaze=False,
        frame_rate=1.0,
        max_persons=4,
        device="auto",
    )
    # Seed initial status; run_portal_job updates it as it progresses.
    write_status(job_dir, {"state": "queued", "message": "smoke test"})

    run_portal_job(config, ROOT_DIR)

    camera_dir = output_root / session_id / "camera_a"
    log_path = job_dir / "job.log"
    log_excerpt = (
        log_path.read_text(errors="replace") if log_path.exists() else "(no job.log)"
    )
    assert (
        camera_dir / "tracks_2d.csv"
    ).exists(), f"tracks_2d.csv missing under {camera_dir}.\njob.log:\n{log_excerpt}"
    assert (camera_dir / "pose_3d.csv").exists()
    assert (camera_dir / "manifest.json").exists()
    # Downstream stages: filter -> smooth -> metrics -> overlay video. Each is
    # a separate subprocess, so a regression in any one breaks one of these.
    assert (camera_dir / "tracks_2d_filtered.csv").exists()
    assert (camera_dir / "tracks_2d_smooth.csv").exists()
    assert (camera_dir / "pose_metrics_summary.json").exists()
    overlay = camera_dir / "overlay_2min.mp4"
    assert overlay.exists(), f"overlay video missing under {camera_dir}"
    assert overlay.stat().st_size > 0, "overlay video is empty"
