"""Job progress page, failure messages, label submission, check clips.

Covers the portal-revamp Phase 4 and 5 behaviour: the stage table in
``portal/jobs.py``, ``GET /jobs/{id}/status.json``, plain failure messages
(never a raw command line), labels accepted only while a job waits for them,
check clips that cannot fail a job, and the labeling page's boxes and counter.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import portal.app as portal_app
import portal.jobs as jobs
from portal.jobs import (
    JOB_STAGE_KEYS,
    CommandStep,
    PortalJobConfig,
    SessionBlock,
    build_phase1_steps,
    build_phase2_steps,
    describe_step,
    friendly_failure_message,
    read_status,
    write_status,
)

JOB_ID = "job-001"
RAW_COMMAND = "Command '['python', '-m', 'video_analysis.pose_metrics']' returned"


@pytest.fixture
def portal_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the portal at temporary folders and configure its secrets.

    Args:
        tmp_path: Pytest temporary directory used as OUTPUT_ROOT.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The temporary output root.
    """
    monkeypatch.setenv("PORTAL_PASSWORD", "test-password")
    monkeypatch.setenv("PORTAL_SECRET_KEY", "test-secret")
    monkeypatch.setattr(portal_app, "JOB_ROOT", tmp_path / "portal_jobs")
    monkeypatch.setattr(portal_app, "UPLOAD_ROOT", tmp_path / "portal_uploads")
    monkeypatch.setattr(portal_app, "OUTPUT_ROOT", tmp_path)
    portal_app.SESSION_TOKENS.clear()
    portal_app.LOGIN_FAILURES.clear()
    return tmp_path


@pytest.fixture
def client(portal_env: Path) -> TestClient:
    """Return a logged-in test client.

    Args:
        portal_env: Temporary portal folders.

    Returns:
        A ``TestClient`` carrying a valid ``portal_session`` cookie.
    """
    test_client = TestClient(portal_app.app)
    test_client.cookies.set("portal_session", portal_app.create_session_token())
    return test_client


def _job_dir(root: Path) -> Path:
    return root / "portal_jobs" / JOB_ID


def _config(tmp_path: Path) -> PortalJobConfig:
    return PortalJobConfig(
        job_id=JOB_ID,
        session_id="P001c",
        video_a=tmp_path / "camera_a_rapid.mp4",
        video_b=tmp_path / "camera_b_rapid.mp4",
        job_dir=tmp_path / JOB_ID,
        output_root=tmp_path / "output",
        session_config_path=tmp_path / JOB_ID / "session_config.json",
        blocks=[SessionBlock(name="free_play", start_s=100.0, end_s=220.0)],
        include_gaze=True,
    )


# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------


def test_every_pipeline_step_maps_to_a_stage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    names = [s.name for s in build_phase1_steps(config)]
    names += [s.name for s in build_phase2_steps(config)]
    names += [
        "uploading",
        "queued",
        "finalize uploads",
        "compression",
        "extract anchors",
        "awaiting annotation",
        "phase 2 queued",
        "refine keypoints camera_a",
        "session report",
        "results zip",
    ]
    for name in names:
        info = describe_step(name)
        assert info["stage"] in JOB_STAGE_KEYS, name
        assert info["message"] != "Working.", name


@pytest.mark.parametrize(
    ("step", "stage", "message"),
    [
        ("compression", "prepare", "Making the videos smaller for processing."),
        ("pose inference", "find_people", "Finding the people in every frame."),
        ("awaiting annotation", "label_parent", "Waiting for you"),
        ("filter tracks camera_a", "track_refine", "Video A: Following the parent"),
        ("smooth tracks camera_b", "track_refine", "Video B: Smoothing movement."),
        ("pose metrics camera_a", "measure", "Video A: Measuring movement."),
        ("overlay video camera_a story (end)", "measure", "Video A: Making short"),
        ("gaze inference camera_b", "measure", "Video B: Estimating where"),
        ("session report", "report", "Writing the session report."),
    ],
)
def test_describe_step_gives_stage_and_plain_message(
    step: str, stage: str, message: str
) -> None:
    info = describe_step(step)
    assert info["stage"] == stage
    assert info["message"].startswith(message)


def test_unknown_step_is_described_generically() -> None:
    assert describe_step("something new") == {
        "stage": "",
        "message": "Working.",
        "failure": "processing",
    }


def test_friendly_failure_never_shows_the_command_line() -> None:
    error = subprocess.CalledProcessError(
        1, ["python", "-m", "video_analysis.pose_metrics"]
    )
    message = friendly_failure_message("pose metrics camera_a", error)
    assert message.startswith("The job stopped while measuring movement for Video A.")
    assert "Command" not in message
    assert "python" not in message


def test_friendly_failure_keeps_plain_value_errors() -> None:
    error = ValueError("The two videos ended up at different frames per second.")
    message = friendly_failure_message("compression", error)
    assert message == (
        "The job stopped while preparing the videos. "
        "The two videos ended up at different frames per second."
    )


# ---------------------------------------------------------------------------
# status.json and the job page
# ---------------------------------------------------------------------------


def test_status_json_requires_login(portal_env: Path) -> None:
    write_status(_job_dir(portal_env), {"state": "running"})
    response = TestClient(portal_app.app).get(
        f"/jobs/{JOB_ID}/status.json", follow_redirects=False
    )
    assert response.status_code in {303, 401}


def test_status_json_missing_job_is_404(client: TestClient) -> None:
    assert client.get("/jobs/nope/status.json").status_code == 404


def test_status_json_shape_for_running_job(
    client: TestClient, portal_env: Path
) -> None:
    write_status(
        _job_dir(portal_env),
        jobs.running_status(
            "pose metrics camera_a",
            phase=2,
            cameras={
                "camera_a": "Video A: Measuring movement.",
                "camera_b": "Video B: Smoothing movement.",
            },
        ),
    )

    body = client.get(f"/jobs/{JOB_ID}/status.json").json()

    assert body["state"] == "running"
    assert body["polling"] is True
    assert body["stage"] == "measure"
    assert body["message"] == "Working on both videos at the same time."
    assert body["chip_label"] == "Running"
    assert [s["key"] for s in body["stages"]] == list(JOB_STAGE_KEYS)
    progress = {s["key"]: s["status"] for s in body["stages"]}
    assert progress == {
        "upload": "done",
        "prepare": "done",
        "find_people": "done",
        "label_parent": "done",
        "track_refine": "done",
        "measure": "current",
        "report": "todo",
    }
    assert body["cameras"] == [
        {"video": "Video A", "message": "Video A: Measuring movement."},
        {"video": "Video B", "message": "Video B: Smoothing movement."},
    ]
    assert body["annotate_url"] is None
    assert "log_tail" not in body


def test_status_json_awaiting_labels_points_to_annotate(
    client: TestClient, portal_env: Path
) -> None:
    write_status(_job_dir(portal_env), {"state": "awaiting_annotation"})

    body = client.get(f"/jobs/{JOB_ID}/status.json").json()

    assert body["polling"] is False
    assert body["annotate_url"] == f"/jobs/{JOB_ID}/annotate"
    current = [s["key"] for s in body["stages"] if s["status"] == "current"]
    assert current == ["label_parent"]


def test_status_json_complete_marks_every_stage_done(
    client: TestClient, portal_env: Path
) -> None:
    write_status(_job_dir(portal_env), {"state": "complete", "stage": "report"})

    body = client.get(f"/jobs/{JOB_ID}/status.json").json()

    assert body["polling"] is False
    assert {s["status"] for s in body["stages"]} == {"done"}


def _failed_job(root: Path) -> Path:
    job_dir = _job_dir(root)
    error = subprocess.CalledProcessError(
        1, ["python", "-m", "video_analysis.pose_metrics"]
    )
    write_status(job_dir, jobs.failed_status("pose metrics camera_a", error))
    lines = [f"log line {i}" for i in range(50)] + [RAW_COMMAND + " <b>x</b>"]
    (job_dir / "job.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return job_dir


def test_status_json_failed_job_has_plain_message_and_log_tail(
    client: TestClient, portal_env: Path
) -> None:
    _failed_job(portal_env)

    body = client.get(f"/jobs/{JOB_ID}/status.json").json()

    assert body["state"] == "failed"
    assert body["message"].startswith("The job stopped while measuring movement")
    assert "Command" not in body["message"]
    assert "returned non-zero exit status" in body["error"]
    tail = body["log_tail"].splitlines()
    assert len(tail) == portal_app.FAILURE_LOG_LINES
    assert tail[-1].startswith(RAW_COMMAND)
    progress = {s["key"]: s["status"] for s in body["stages"]}
    assert progress["measure"] == "failed"
    assert progress["track_refine"] == "done"
    assert progress["report"] == "todo"


def test_failed_job_page_hides_command_behind_details(
    client: TestClient, portal_env: Path
) -> None:
    _failed_job(portal_env)

    text = client.get(f"/jobs/{JOB_ID}").text

    headline = text.split('id="job-message"', 1)[1].split("</p>", 1)[0]
    assert "The job stopped while measuring movement for Video A." in headline
    assert "Command" not in headline
    assert "<summary>Show technical details</summary>" in text
    details = text.split("<details", 1)[1].split("</details>", 1)[0]
    assert "returned non-zero exit status" in details
    assert "log line 49" in details
    assert "log line 5\n" not in details
    assert "&lt;b&gt;x&lt;/b&gt;" in details
    assert "/static/job.js" not in text


def test_job_page_lists_stages_with_current_marked(
    client: TestClient, portal_env: Path
) -> None:
    write_status(_job_dir(portal_env), jobs.running_status("pose inference", phase=1))

    text = client.get(f"/jobs/{JOB_ID}").text

    assert 'class="stage stage--current" data-stage="find_people"' in text
    assert 'class="stage stage--done" data-stage="prepare"' in text
    assert 'class="stage stage--todo" data-stage="report"' in text
    assert "Finding the people in every frame." in text
    assert "Phase 1 step" not in text


def test_job_page_lists_notes(client: TestClient, portal_env: Path) -> None:
    job_dir = _job_dir(portal_env)
    write_status(job_dir, {"state": "complete"})
    jobs.add_job_note(job_dir, "Check clip not made (Video A, story).")

    text = client.get(f"/jobs/{JOB_ID}").text

    assert "<h3>Notes</h3>" in text
    assert "Check clip not made (Video A, story)." in text


# ---------------------------------------------------------------------------
# Failure points in the runners
# ---------------------------------------------------------------------------


def _prepare_phase2(tmp_path: Path, monkeypatch, steps: list[CommandStep]):
    config = _config(tmp_path)
    config.include_gaze = False
    config.parallel_cameras = False
    config.job_dir.mkdir(parents=True, exist_ok=True)
    config.session_output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(jobs, "achieved_session_frame_rate", lambda cfg: 5.0)
    monkeypatch.setattr(jobs, "resolve_anchors_json_paths", lambda cfg: {})
    monkeypatch.setattr(
        jobs, "build_phase2_steps", lambda cfg, anchors_json_by_camera=None: steps
    )
    return config


def test_phase2_failure_names_the_stage_not_the_command(
    tmp_path: Path, monkeypatch
) -> None:
    steps = [
        CommandStep("smooth tracks camera_a", ["x"], "camera_a"),
        CommandStep("pose metrics camera_a", ["x"], "camera_a"),
    ]
    config = _prepare_phase2(tmp_path, monkeypatch, steps)

    def fake_run(step, cwd, log_path):
        if step.name.startswith("pose metrics"):
            raise subprocess.CalledProcessError(1, ["python", "-m", "pose_metrics"])

    monkeypatch.setattr(jobs, "run_command_step", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        jobs.resume_portal_job_phase2(config, tmp_path)

    status = read_status(config.job_dir)
    assert status["state"] == "failed"
    assert status["stage"] == "measure"
    assert status["failed_step"] == "pose metrics camera_a"
    assert status["message"].startswith(
        "The job stopped while measuring movement for Video A."
    )
    assert "Command" not in status["message"]
    assert "returned non-zero exit status" in status["error"]


def test_parallel_failure_reports_the_failing_step(tmp_path: Path, monkeypatch) -> None:
    steps = [
        CommandStep("smooth tracks camera_a", ["x"], "camera_a"),
        CommandStep("pose metrics camera_b", ["x"], "camera_b"),
    ]
    config = _prepare_phase2(tmp_path, monkeypatch, steps)
    config.parallel_cameras = True

    def fake_run(step, cwd, log_path):
        if step.name == "smooth tracks camera_a":
            raise subprocess.CalledProcessError(1, ["python"])

    monkeypatch.setattr(jobs, "run_command_step", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        jobs.resume_portal_job_phase2(config, tmp_path)

    status = read_status(config.job_dir)
    assert status["failed_step"] == "smooth tracks camera_a"
    assert "smoothing movement for Video A" in status["message"]


def test_background_wrapper_keeps_the_pipeline_failure_message(
    portal_env: Path, monkeypatch
) -> None:
    job_dir = _job_dir(portal_env)
    error = subprocess.CalledProcessError(1, ["python"])
    write_status(job_dir, jobs.failed_status("pose inference", error))

    def action_fails(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(portal_app, "_run_with_job_lock", action_fails)
    portal_app.resume_portal_job_phase2_background(JOB_ID, job_dir)

    status = read_status(job_dir)
    assert status["failed_step"] == "pose inference"
    assert "finding the people" in status["message"]


# ---------------------------------------------------------------------------
# Check clips (overlay) never fail a job
# ---------------------------------------------------------------------------


def test_overlay_steps_are_optional(tmp_path: Path) -> None:
    steps = build_phase2_steps(_config(tmp_path))
    overlay = [s for s in steps if s.name.startswith("overlay video")]
    assert overlay and all(s.optional for s in overlay)
    assert not any(s.optional for s in steps if s not in overlay)


def test_failed_overlay_is_noted_and_job_completes(tmp_path: Path, monkeypatch) -> None:
    steps = [
        CommandStep("pose metrics camera_a", ["x"], "camera_a"),
        CommandStep(
            "overlay video camera_a story (end)", ["x"], "camera_a", optional=True
        ),
        CommandStep("gaze inference camera_a", ["x"], "camera_a"),
    ]
    config = _prepare_phase2(tmp_path, monkeypatch, steps)
    ran: list[str] = []

    def fake_run(step, cwd, log_path):
        ran.append(step.name)
        if step.name.startswith("overlay video"):
            raise subprocess.CalledProcessError(1, ["python"])

    monkeypatch.setattr(jobs, "run_command_step", fake_run)
    jobs.resume_portal_job_phase2(config, tmp_path)

    status = read_status(config.job_dir)
    assert status["state"] == "complete"
    assert ran[-2:] == ["gaze inference camera_a", "session report"]
    assert Path(status["result_zip"]).exists()
    notes = jobs.read_job_notes(config.job_dir)
    assert len(notes) == 1
    assert "Video A, story end clip" in notes[0]


def _clip_camera(tmp_path: Path, tracked_from_s: float) -> Path:
    """Camera folder with 10 frames at 0.0-1.8 s; tracks only after a time."""
    cv2 = pytest.importorskip("cv2")
    camera_dir = tmp_path / "camera_a"
    frames = camera_dir / "frames"
    frames.mkdir(parents=True)
    rows = []
    for i in range(10):
        name = f"frame_{i + 1:06d}.jpg"
        cv2.imwrite(str(frames / name), np.zeros((60, 80, 3), dtype=np.uint8))
        rows.append((i, i * 0.2, name))
    pd.DataFrame(rows, columns=["frame_idx", "timestamp_s", "image_name"]).to_csv(
        frames / "frame_index.csv", index=False
    )
    track_rows = [
        (0, t, 0, "parent", 1, 1, 20, 40, 0.9)
        for t in np.arange(tracked_from_s, tracked_from_s + 2, 0.2)
    ]
    pd.DataFrame(
        track_rows,
        columns=[
            "frame_idx",
            "timestamp_s",
            "track_id",
            "track_label",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "track_confidence",
        ],
    ).to_csv(camera_dir / "tracks.csv", index=False)
    pd.DataFrame(
        columns=[
            "timestamp_s",
            "track_id",
            "keypoint_name",
            "x_m",
            "y_m",
            "z_m",
            "keypoint_confidence",
        ]
    ).to_csv(camera_dir / "pose.csv", index=False)
    return camera_dir


def test_overlay_with_nobody_tracked_renders_raw_frames(tmp_path: Path) -> None:
    from video_analysis.visualize_pose_tracks import (
        VisualizationConfig,
        render_pose_track_video,
    )

    camera_dir = _clip_camera(tmp_path, tracked_from_s=100.0)
    summary = render_pose_track_video(
        VisualizationConfig(
            camera_dir=str(camera_dir),
            output_video=str(camera_dir / "clip.mp4"),
            pose_csv="pose.csv",
            tracks_csv="tracks.csv",
            start_time="0:00",
            end_time="0:01.8",
        )
    )
    assert summary["rendered_frames"] == 10
    assert (camera_dir / "clip.mp4").stat().st_size > 0


def test_overlay_past_video_end_still_raises(tmp_path: Path) -> None:
    from video_analysis.visualize_pose_tracks import (
        VisualizationConfig,
        render_pose_track_video,
    )

    camera_dir = _clip_camera(tmp_path, tracked_from_s=0.0)
    with pytest.raises(ValueError, match="No frames"):
        render_pose_track_video(
            VisualizationConfig(
                camera_dir=str(camera_dir),
                output_video=str(camera_dir / "clip.mp4"),
                pose_csv="pose.csv",
                tracks_csv="tracks.csv",
                start_time="0:50",
                end_time="1:00",
            )
        )


# ---------------------------------------------------------------------------
# Label submission: only while waiting, launched once
# ---------------------------------------------------------------------------


def _anchor_job(root: Path, state: str = "awaiting_annotation") -> Path:
    job_dir = _job_dir(root)
    write_status(job_dir, {"state": state, "message": "x"})
    camera_dir = job_dir / "anchors" / "camera_a"
    camera_dir.mkdir(parents=True)
    (camera_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "anchor_idx": idx,
                    "image_filename": f"anchor_{idx:02d}.jpg",
                    "frame_idx": 10 * idx,
                    "timestamp_s": 1.0 * idx,
                    "tracks": [
                        {
                            "track_id": 0,
                            "bbox_xyxy": [863.0, 280.5, 959.0, 533.5],
                            "color_bgr": [0, 255, 0],
                        },
                        {
                            "track_id": 1,
                            "bbox_xyxy": [538.5, 0.0, 706.5, 463.75],
                            "color_bgr": [0, 165, 255],
                        },
                    ],
                }
                for idx in (1, 2, 3)
            ]
        ),
        encoding="utf-8",
    )
    return job_dir


PAYLOAD = {
    "annotations": {
        "camera_a": [
            {"anchor_idx": 1, "parent_track_id": 1, "child_track_id": 0},
            {"anchor_idx": 2, "skipped": True},
        ]
    }
}


@pytest.fixture
def launches(monkeypatch) -> list[str]:
    """Record phase-2 launches instead of running them."""
    seen: list[str] = []
    monkeypatch.setattr(
        portal_app,
        "resume_portal_job_phase2_background",
        lambda job_id, job_dir: seen.append(job_id),
    )
    return seen


def _post(client: TestClient, payload: dict):
    token = portal_app.csrf_token_for_session(client.cookies["portal_session"])
    return client.post(
        f"/jobs/{JOB_ID}/annotations",
        json=payload,
        headers={"x-csrf-token": token},
    )


def test_labels_accepted_while_waiting(
    client: TestClient, portal_env: Path, launches: list[str]
) -> None:
    job_dir = _anchor_job(portal_env)

    response = _post(client, PAYLOAD)

    assert response.status_code == 200
    assert launches == [JOB_ID]
    saved = json.loads(
        (job_dir / "anchors" / "camera_a" / "anchors.json").read_text("utf-8")
    )
    assert saved == [
        {"frame_idx": 10, "parent_track_id": 1, "child_track_id": 0},
        {"frame_idx": 20, "skipped": True},
    ]
    status = read_status(job_dir)
    assert status["state"] == "queued"
    assert status["stage"] == "track_refine"


@pytest.mark.parametrize(
    "state", ["queued", "running", "complete", "failed", "uploading"]
)
def test_labels_refused_unless_waiting(
    client: TestClient, portal_env: Path, launches: list[str], state: str
) -> None:
    job_dir = _anchor_job(portal_env, state=state)

    response = _post(client, PAYLOAD)

    assert response.status_code == 409
    assert "Nothing was changed" in response.json()["detail"]
    assert launches == []
    assert not (job_dir / "anchors" / "camera_a" / "anchors.json").exists()
    assert read_status(job_dir)["state"] == state


@pytest.mark.parametrize(
    "payload", [{}, {"annotations": {}}, {"annotations": {"camera_a": []}}]
)
def test_empty_label_submission_is_refused(
    client: TestClient, portal_env: Path, launches: list[str], payload: dict
) -> None:
    job_dir = _anchor_job(portal_env)

    response = _post(client, payload)

    assert response.status_code == 409
    assert "No frames were labeled" in response.json()["detail"]
    assert launches == []
    assert read_status(job_dir)["state"] == "awaiting_annotation"


def test_second_submit_is_refused(
    client: TestClient, portal_env: Path, launches: list[str]
) -> None:
    _anchor_job(portal_env)

    assert _post(client, PAYLOAD).status_code == 200
    second = _post(client, PAYLOAD)

    assert second.status_code == 409
    assert "already received" in second.json()["detail"]
    assert launches == [JOB_ID]


def test_simultaneous_submits_launch_phase2_once(
    client: TestClient, portal_env: Path, launches: list[str], monkeypatch
) -> None:
    _anchor_job(portal_env)
    real_read_status = portal_app.read_status

    def slow_read_status(job_dir: Path):
        status = real_read_status(job_dir)
        time.sleep(0.2)  # widen the check-then-write window
        return status

    monkeypatch.setattr(portal_app, "read_status", slow_read_status)
    codes: list[int] = []

    def submit() -> None:
        other = TestClient(portal_app.app)
        other.cookies.set("portal_session", client.cookies["portal_session"])
        codes.append(_post(other, PAYLOAD).status_code)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(codes) == [200, 409]
    assert launches == [JOB_ID]


# ---------------------------------------------------------------------------
# Labeling page
# ---------------------------------------------------------------------------


def test_annotate_page_shows_counter_boxes_and_buttons(
    client: TestClient, portal_env: Path
) -> None:
    _anchor_job(portal_env)

    text = client.get(f"/jobs/{JOB_ID}/annotate").text

    assert 'id="annotate-progress"' in text
    assert "0 of 3 labeled" in text
    assert "<h3>Video A</h3>" in text
    assert 'data-bbox="863,280.5,959,533.5"' in text
    assert 'data-bbox="538.5,0,706.5,463.75"' in text
    assert text.count('<svg aria-hidden="true"') == 3
    assert text.count('class="track-btn"') == 6
    assert text.count('class="skip-btn"') == 3


def test_annotate_page_without_boxes_still_renders(
    client: TestClient, portal_env: Path
) -> None:
    job_dir = _anchor_job(portal_env)
    manifest = job_dir / "anchors" / "camera_a" / "manifest.json"
    entries = json.loads(manifest.read_text("utf-8"))
    for entry in entries:
        for track in entry["tracks"]:
            del track["bbox_xyxy"]
    manifest.write_text(json.dumps(entries), encoding="utf-8")

    text = client.get(f"/jobs/{JOB_ID}/annotate").text

    assert "data-bbox" not in text
    assert text.count('class="track-btn"') == 6


def test_annotate_script_keeps_the_payload_shape() -> None:
    script = (Path(portal_app.__file__).parent / "static" / "annotate.js").read_text(
        encoding="utf-8"
    )
    for key in ("anchor_idx", "parent_track_id", "child_track_id", "skipped"):
        assert key in script
    assert "JSON.stringify({ annotations: payload })" in script
    assert "form.dataset.csrfToken" in script
    assert '"x-csrf-token": CSRF_TOKEN' in script


# ---------------------------------------------------------------------------
# Complete job page and support contact (portal-revamp Phases 6-7)
# ---------------------------------------------------------------------------


def _complete_job(root: Path, with_report: bool = True) -> Path:
    job_dir = _job_dir(root)
    write_status(job_dir, {"state": "complete", "stage": "report"})
    (job_dir / "job_config.json").write_text(
        json.dumps({"session_id": "P001c"}), encoding="utf-8"
    )
    session = root / "P001c"
    (session / "camera_a").mkdir(parents=True)
    pd.DataFrame(
        {"camera_id": ["camera_a"], "block": ["free_play"], "coverage_both": [0.9]}
    ).to_csv(session / "session_metrics.csv", index=False)
    if with_report:
        (session / "report.html").write_text("<html></html>", encoding="utf-8")
    return job_dir


def test_complete_job_page_links_to_results(
    client: TestClient, portal_env: Path
) -> None:
    _complete_job(portal_env)

    text = client.get(f"/jobs/{JOB_ID}").text

    assert 'class="button-link" href="/sessions/P001c"' in text
    assert "Open results" in text


def test_complete_job_without_report_has_no_results_button(
    client: TestClient, portal_env: Path
) -> None:
    _complete_job(portal_env, with_report=False)

    text = client.get(f"/jobs/{JOB_ID}").text

    assert "Open results" not in text


def test_running_job_page_has_no_results_button(
    client: TestClient, portal_env: Path
) -> None:
    _complete_job(portal_env)
    write_status(_job_dir(portal_env), jobs.running_status("pose inference", phase=1))

    assert "Open results" not in client.get(f"/jobs/{JOB_ID}").text


def test_failure_message_names_support_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = subprocess.CalledProcessError(1, ["python"])

    monkeypatch.delenv("PORTAL_SUPPORT_CONTACT", raising=False)
    assert friendly_failure_message("compression", error).endswith(
        "send them to Felipe."
    )
    monkeypatch.setenv("PORTAL_SUPPORT_CONTACT", "Dr. <Lab> Admin")
    assert friendly_failure_message("compression", error).endswith(
        "send them to Dr. <Lab> Admin."
    )


def test_failed_job_page_escapes_support_contact(
    client: TestClient, portal_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PORTAL_SUPPORT_CONTACT", "<b>Lab</b>")
    _failed_job(portal_env)

    text = client.get(f"/jobs/{JOB_ID}").text

    assert "send them to &lt;b&gt;Lab&lt;/b&gt;." in text
