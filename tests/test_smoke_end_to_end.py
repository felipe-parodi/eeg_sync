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

import json
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
        # Use the portal's production tracker so the smoke test catches
        # any regression in the boxmot+botsort wiring (validation allow-
        # lists, ReID backend layering, etc.) before a real upload does.
        "--tracker-backend",
        "boxmot",
        "--tracker-name",
        "botsort",
        "--max-persons",
        "3",
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
    """Exercise the full portal flow: phase 1 -> anchors -> phase 2.

    Mirrors how a real colleague would use the portal: upload, wait for
    awaiting_annotation, label the rendered anchors, get the final zip.
    The test fakes the labelling step by reading the per-camera anchor
    manifest written by phase 1 and asserting parent/child against
    whichever two track ids are present.
    """
    video = _require_sample_video()
    _require_ffmpeg()
    _require_ultralytics()

    from portal.jobs import (
        PortalJobConfig,
        SessionBlock,
        read_status,
        resume_portal_job_phase2,
        run_portal_job,
        write_status,
    )

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

    # Window must be wide enough that anchor extraction (which drops the
    # first/last ~1 s by default) still has frames to choose from.
    blocks = [SessionBlock(name="smoke", start_s=0.0, end_s=8.0, color="gray")]
    config = PortalJobConfig(
        job_id=job_id,
        session_id=session_id,
        video_a=staged_video,
        video_b=None,
        job_dir=job_dir,
        output_root=output_root,
        session_config_path=session_config_path,
        blocks=blocks,
        blocks_by_camera={"camera_a": blocks},
        include_gaze=False,
        frame_rate=2.0,
        max_persons=2,
        device="auto",
    )
    write_status(job_dir, {"state": "queued", "message": "smoke test"})

    # Phase 1: pose + tracking + anchor extraction.
    run_portal_job(config, ROOT_DIR)

    status_after_phase1 = read_status(job_dir)
    assert (
        status_after_phase1["state"] == "awaiting_annotation"
    ), f"expected awaiting_annotation, got {status_after_phase1}"

    camera_dir = output_root / session_id / "camera_a"
    assert (camera_dir / "tracks_2d.csv").exists()
    assert (camera_dir / "pose_3d.csv").exists()
    assert (camera_dir / "manifest.json").exists()

    # Anchor frames + per-camera manifest landed under the job dir.
    anchors_manifest = json.loads(
        (job_dir / "anchors" / "camera_a" / "manifest.json").read_text(encoding="utf-8")
    )
    assert anchors_manifest, "expected at least one anchor frame"

    # Synthesize an annotation: for every anchor that has 2 tracks, pick the
    # first as parent and the second as child. (For a smoke test, the chosen
    # role is arbitrary -- we only care that the anchor-aware filter runs.)
    anchors_json = []
    for entry in anchors_manifest:
        tracks = entry.get("tracks", [])
        if len(tracks) >= 2:
            anchors_json.append(
                {
                    "frame_idx": int(entry["frame_idx"]),
                    "parent_track_id": int(tracks[0]["track_id"]),
                    "child_track_id": int(tracks[1]["track_id"]),
                }
            )
    if not anchors_json:
        pytest.skip(
            "Sample video produced no anchors with >=2 tracks; cannot exercise "
            "the anchor-aware filter end-to-end."
        )
    (job_dir / "anchors" / "camera_a" / "anchors.json").write_text(
        json.dumps(anchors_json), encoding="utf-8"
    )

    # Phase 2: anchor-aware filter -> smooth -> metrics -> overlay -> zip.
    resume_portal_job_phase2(config, ROOT_DIR)

    final_status = read_status(job_dir)
    assert final_status["state"] == "complete", f"expected complete, got {final_status}"
    assert (camera_dir / "tracks_2d_filtered.csv").exists()
    assert (camera_dir / "tracks_2d_smooth.csv").exists()
    assert (camera_dir / "pose_metrics_summary.json").exists()
    overlay = camera_dir / "overlay_2min.mp4"
    assert overlay.exists(), f"overlay video missing under {camera_dir}"
    assert overlay.stat().st_size > 0, "overlay video is empty"
