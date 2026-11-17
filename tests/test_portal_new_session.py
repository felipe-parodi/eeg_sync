"""Tests for the new-session form backend (revamp Phase 3).

Covers the server-side validation of task and extra-person-window text (the
form still posts the old ``name,start,end`` / ``start,end,reason`` lines), the
10 s minimum part length, the session-ID reuse check with archive-on-replace,
the ffprobe video-length check in the finish step, and the job page only
showing results for a finished job.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import portal.app as portal_app
import portal.jobs as portal_jobs
from portal.jobs import (
    MIN_PART_S,
    ExclusionWindow,
    SessionBlock,
    format_clock,
    parse_timecode,
    split_blocks_with_min_part,
    write_status,
)
from video_analysis.session_report import build_all_sessions_table

# --- fixtures ----------------------------------------------------------------


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


def _csrf_headers(client: TestClient) -> dict[str, str]:
    """Return the CSRF header for the client's session.

    Args:
        client: Logged-in test client.

    Returns:
        Header dict with ``X-CSRF-Token``.
    """
    token = portal_app.csrf_token_for_session(client.cookies["portal_session"])
    return {"X-CSRF-Token": token}


def _metadata(
    blocks_a: str = "grocery,1:00,5:00",
    exclusions_a: str = "",
    blocks_b: str | None = None,
    exclusions_b: str | None = None,
    with_b: bool = False,
) -> dict[str, Any]:
    """Build submission metadata for camera A (and optionally B).

    Args:
        blocks_a: Video A task lines.
        exclusions_a: Video A extra-person window lines.
        blocks_b: Video B task lines.
        exclusions_b: Video B extra-person window lines.
        with_b: Whether Video B is included.

    Returns:
        The metadata dict.
    """
    return portal_app.build_submission_metadata(
        job_id="job-001",
        created_at="2026-11-16T10:00:00",
        safe_session_id="P001",
        blocks_text=blocks_a,
        exclusions_text=exclusions_a,
        blocks_text_b=blocks_b,
        exclusions_text_b=exclusions_b,
        video_a_name="a.mp4",
        video_b_name="b.mp4" if with_b else "",
        video_a_size=1000,
        video_b_size=1000 if with_b else 0,
        include_gaze=False,
    )


def _init_form(session_id: str = "P001", **extra: str) -> dict[str, str]:
    """Return a valid ``/uploads/init`` form for one camera.

    Args:
        session_id: Session ID to submit.
        **extra: Extra or overriding form fields.

    Returns:
        Form fields.
    """
    form = {
        "session_id": session_id,
        "video_a_name": "a.mp4",
        "video_a_size": "1000",
        "blocks_text_a": "grocery,1:00,5:00",
        "exclusions_text_a": "",
    }
    form.update(extra)
    return form


def _make_session(root: Path, name: str) -> Path:
    """Create a processed session folder with one metrics row.

    Args:
        root: Output root.
        name: Session folder name.

    Returns:
        The session folder.
    """
    session = root / name
    (session / "camera_a").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"session_id": name, "camera_id": "camera_a", "block": "grocery"}]
    ).to_csv(session / "session_metrics.csv", index=False)
    return session


# --- time parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("83", 83.0),
        ("83.5", 83.5),
        ("13:26", 806.0),
        ("0:05", 5.0),
        ("75:30", 4530.0),
        ("1:02:03.5", 3723.5),
        (" 1:00 ", 60.0),
    ],
)
def test_parse_timecode_accepts_valid_times(text: str, seconds: float) -> None:
    assert parse_timecode(text) == seconds


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "Time is empty."),
        ("-5", '"-5" is negative. Times count from the start of the video.'),
        ("-0:05", '"-0:05" is negative. Times count from the start of the video.'),
        ("nan", '"nan" is not a time.'),
        ("inf", '"inf" is not a time.'),
        ("1e3", '"1e3" is not a time.'),
        ("1.5:30", '"1.5:30" is not a time.'),
        ("1:2:3:4", '"1:2:3:4" is not a time.'),
        ("1:75", '"1:75" has 75 seconds. Seconds must be under 60.'),
        ("1:60", '"1:60" has 60 seconds. Seconds must be under 60.'),
        ("1:75:00", '"1:75:00" has 75 minutes. Minutes must be under 60.'),
        ("1:05:60", '"1:05:60" has 60 seconds. Seconds must be under 60.'),
        ("abc", '"abc" is not a time.'),
    ],
)
def test_parse_timecode_rejects_bad_times(text: str, message: str) -> None:
    with pytest.raises(ValueError) as error:
        parse_timecode(text)
    assert str(error.value).startswith(message)


def test_parse_timecode_bad_format_explains_accepted_forms() -> None:
    with pytest.raises(ValueError) as error:
        parse_timecode("1.5:30")
    assert str(error.value) == (
        '"1.5:30" is not a time. Use seconds (90), M:SS (1:30) or H:MM:SS (1:01:30).'
    )


def test_format_clock() -> None:
    assert format_clock(5) == "0:05"
    assert format_clock(806) == "13:26"
    assert format_clock(47160) == "13:06:00"
    assert format_clock(65.5) == "1:05.5"


# --- task and window text --------------------------------------------------------


def test_task_with_bad_time_names_task_and_field() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(blocks_a="grocery,1:00,1:75")
    assert str(error.value) == (
        'Video A task "grocery" has a bad end time. '
        '"1:75" has 75 seconds. Seconds must be under 60.'
    )


def test_task_must_end_after_start() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(blocks_a="grocery,13:26,13:20")
    assert str(error.value) == (
        'Video A task "grocery" must end after it starts (start 13:26, end 13:20).'
    )


def test_task_shorter_than_minimum_is_rejected() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(blocks_a="grocery,1:00,1:05")
    assert str(error.value) == (
        'Video A task "grocery" is only 5 s long. Tasks must be at least 10 s.'
    )


def test_task_without_name_is_rejected() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(blocks_a="grocery,1:00,2:00\n!!!,3:00,4:00")
    assert str(error.value) == ('Video A task 2 name "!!!" has no letters or numbers.')


def test_task_line_with_wrong_field_count_is_rejected() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(blocks_a="grocery,1:00")
    assert str(error.value) == "Video A task line 1 must be: name,start,end"


def test_duplicate_task_names_are_rejected() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(blocks_a="grocery,1:00,2:00\ngrocery,3:00,4:00")
    assert str(error.value) == (
        'Video A has two tasks named "grocery". Give each task a different name.'
    )


def test_task_names_that_slug_to_the_same_name_are_rejected() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(blocks_a="récré,1:00,2:00\nr cr,3:00,4:00")
    assert str(error.value) == (
        'Video A task names "récré" and "r cr" both become "r_cr" in file names. '
        "Give each task a different name."
    )


def test_duplicate_names_checked_per_camera_only() -> None:
    metadata = _metadata(
        blocks_a="grocery,1:00,2:00", blocks_b="grocery,1:30,2:30", with_b=True
    )
    assert set(metadata["analysis_blocks_by_camera"]) == {"camera_a", "camera_b"}


def test_video_b_errors_name_video_b() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(blocks_b="story,2:00,1:00", with_b=True)
    assert str(error.value).startswith('Video B task "story" must end after')


def test_exclusion_with_bad_time_is_rejected() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(exclusions_a="1:20,inf,someone walked in")
    assert str(error.value) == (
        "Video A extra-person window 1 has a bad end time. "
        '"inf" is not a time. Use seconds (90), M:SS (1:30) or H:MM:SS (1:01:30).'
    )


def test_exclusion_must_end_after_start() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(exclusions_a="1:30,1:20,experimenter")
    assert str(error.value) == (
        "Video A extra-person window 1 must end after it starts "
        "(start 1:30, end 1:20)."
    )


def test_task_wholly_inside_window_is_rejected_by_name() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(
            blocks_a="grocery,1:00,5:00\nstorybook,6:00,7:00",
            exclusions_a="5:30,7:30,experimenter",
        )
    assert str(error.value) == (
        'Video A task "storybook" (6:00 to 7:00) is entirely inside an '
        "extra-person window, so none of it can be analysed. Remove the task "
        "or correct the window."
    )


def test_task_whose_pieces_are_all_too_short_is_rejected() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(blocks_a="grocery,1:00,1:30", exclusions_a="1:05,1:25,visitor")
    assert str(error.value) == (
        'After removing extra-person windows, every piece of Video A task "grocery" '
        "is shorter than 10 s. Remove the task or correct the windows."
    )


def test_task_name_colliding_with_generated_part_is_rejected() -> None:
    with pytest.raises(ValueError) as error:
        _metadata(
            blocks_a="story,1:00,5:00\nstory_part1,10:00,12:00",
            exclusions_a="2:00,2:30,visitor",
        )
    assert str(error.value) == (
        'Video A task name "story_part1" clashes with a piece of task "story", '
        "which an extra-person window splits into story_part1 and story_part2. "
        'Rename "story_part1".'
    )


def test_task_named_like_a_part_is_fine_when_nothing_splits() -> None:
    metadata = _metadata(blocks_a="story,1:00,5:00\nstory_part1,10:00,12:00")
    names = [b["name"] for b in metadata["analysis_blocks_by_camera"]["camera_a"]]
    assert names == ["story", "story_part1"]


# --- minimum part length ----------------------------------------------------------


def test_min_part_constant() -> None:
    assert MIN_PART_S == 10.0


def test_split_with_min_part_keeps_numbering_and_reports_dropped() -> None:
    blocks = [SessionBlock("grocery", 960.0, 1610.0)]
    exclusions = [
        ExclusionWindow(1086.0, 1101.0, "experimenter"),
        ExclusionWindow(1105.0, 1200.0, "experimenter"),
    ]
    kept, dropped = split_blocks_with_min_part(blocks, exclusions)
    assert [(b.name, b.start_s, b.end_s) for b in kept] == [
        ("grocery_part1", 960.0, 1086.0),
        ("grocery_part3", 1200.0, 1610.0),
    ]
    assert [(b.name, b.start_s, b.end_s) for b in dropped] == [
        ("grocery_part2", 1101.0, 1105.0)
    ]


def test_split_with_min_part_drops_tiny_sliver() -> None:
    blocks = [SessionBlock("free_play", 100.0, 200.0)]
    exclusions = [ExclusionWindow(100.2, 150.0, "visitor")]
    kept, dropped = split_blocks_with_min_part(blocks, exclusions)
    assert [b.name for b in kept] == ["free_play_part2"]
    assert dropped[0].name == "free_play_part1"
    assert dropped[0].end_s - dropped[0].start_s == pytest.approx(0.2)


def test_metadata_records_dropped_parts() -> None:
    metadata = _metadata(
        blocks_a="grocery,16:00,26:50",
        exclusions_a="18:06,18:21,experimenter\n18:25,20:00,experimenter",
    )
    names = [b["name"] for b in metadata["analysis_blocks_by_camera"]["camera_a"]]
    assert names == ["grocery_part1", "grocery_part3"]
    assert metadata["min_part_s"] == MIN_PART_S
    assert metadata["dropped_parts_by_camera"] == {
        "camera_a": [
            {
                "name": "grocery_part2",
                "start_s": 1101.0,
                "end_s": 1105.0,
                "duration_s": 4.0,
                "reason": "shorter than 10 s after removing extra-person windows",
            }
        ]
    }


def test_metadata_without_dropped_parts_has_empty_lists() -> None:
    metadata = _metadata()
    assert metadata["dropped_parts_by_camera"] == {"camera_a": []}


# --- the form still posts the old text format ----------------------------------


def test_init_accepts_old_text_format(client: TestClient, portal_env: Path) -> None:
    response = client.post(
        "/uploads/init",
        data=_init_form(
            blocks_text_a="grocery,16:00,26:50\nstorybook,29:22,37:06",
            exclusions_text_a="18:06,18:21,experimenter entered",
        ),
        headers=_csrf_headers(client),
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    meta = json.loads(
        (portal_env / "portal_jobs" / job_id / "upload_meta.json").read_text(
            encoding="utf-8"
        )
    )
    names = [b["name"] for b in meta["analysis_blocks_by_camera"]["camera_a"]]
    assert names == ["grocery_part1", "grocery_part2", "storybook"]
    assert meta["replace_existing"] is False


def test_init_returns_plain_validation_message(client: TestClient) -> None:
    response = client.post(
        "/uploads/init",
        data=_init_form(blocks_text_a="grocery,1:00,1:75"),
        headers=_csrf_headers(client),
    )
    assert response.status_code == 400
    assert "Seconds must be under 60" in response.json()["detail"]


# --- session ID reuse --------------------------------------------------------------


def test_exists_endpoint_reports_existing_session(
    client: TestClient, portal_env: Path
) -> None:
    _make_session(portal_env, "P001")
    payload = client.get("/sessions/P001/exists").json()
    assert payload == {
        "session_id": "P001",
        "valid": True,
        "exists": True,
        "active_job": False,
        "message": "",
    }
    assert client.get("/sessions/P002/exists").json()["exists"] is False


def test_exists_endpoint_reports_slugged_id(client: TestClient) -> None:
    payload = client.get("/sessions/P 001/exists").json()
    assert payload["session_id"] == "P_001"
    assert payload["exists"] is False


def test_exists_endpoint_flags_reserved_name(client: TestClient) -> None:
    payload = client.get("/sessions/portal_jobs/exists").json()
    assert payload["valid"] is False
    assert payload["message"] == (
        '"portal_jobs" is used by the portal itself. Choose another session ID.'
    )


def test_exists_endpoint_requires_auth(portal_env: Path) -> None:
    response = TestClient(portal_app.app).get(
        "/sessions/P001/exists", headers={"accept": "application/json"}
    )
    assert response.status_code == 401


def test_init_refuses_existing_session_without_flag(
    client: TestClient, portal_env: Path
) -> None:
    _make_session(portal_env, "P001")
    response = client.post(
        "/uploads/init", data=_init_form(), headers=_csrf_headers(client)
    )
    assert response.status_code == 409
    assert response.json()["detail"] == (
        'Session "P001" already has results. Choose a new session ID, or tick '
        '"Replace the earlier results" to move the old results to the archive '
        "folder."
    )
    assert not (portal_env / "portal_jobs").exists() or not any(
        (portal_env / "portal_jobs").iterdir()
    )


def test_init_accepts_existing_session_with_flag(
    client: TestClient, portal_env: Path
) -> None:
    _make_session(portal_env, "P001")
    response = client.post(
        "/uploads/init",
        data=_init_form(replace_existing="yes"),
        headers=_csrf_headers(client),
    )
    assert response.status_code == 200, response.text
    job_dir = portal_env / "portal_jobs" / response.json()["job_id"]
    meta = json.loads((job_dir / "upload_meta.json").read_text(encoding="utf-8"))
    assert meta["replace_existing"] is True
    # Nothing is moved until the upload finishes.
    assert (portal_env / "P001" / "session_metrics.csv").is_file()


def test_init_refuses_session_with_active_job(
    client: TestClient, portal_env: Path
) -> None:
    first = client.post(
        "/uploads/init", data=_init_form(), headers=_csrf_headers(client)
    )
    assert first.status_code == 200
    second = client.post(
        "/uploads/init",
        data=_init_form(replace_existing="yes"),
        headers=_csrf_headers(client),
    )
    assert second.status_code == 409
    assert second.json()["detail"] == (
        'Session "P001" is already being processed. Wait for that job to '
        "finish, or choose a new session ID."
    )
    assert client.get("/sessions/P001/exists").json()["active_job"] is True


def test_init_refuses_reserved_session_id(client: TestClient) -> None:
    response = client.post(
        "/uploads/init",
        data=_init_form(session_id="Portal_Jobs"),
        headers=_csrf_headers(client),
    )
    assert response.status_code == 400
    assert "used by the portal itself" in response.json()["detail"]


def test_archive_session_dir_moves_folder(portal_env: Path) -> None:
    session = _make_session(portal_env, "P001")
    (session / "camera_a" / "tracks.csv").write_text("x", encoding="utf-8")
    archived = portal_app.archive_session_dir("P001", stamp="20261116_101500")
    assert archived == portal_env / "_archive" / "P001_20261116_101500"
    assert not session.exists()
    assert (archived / "camera_a" / "tracks.csv").read_text(encoding="utf-8") == "x"


def test_archive_session_dir_never_overwrites(portal_env: Path) -> None:
    _make_session(portal_env, "P001")
    first = portal_app.archive_session_dir("P001", stamp="20261116_101500")
    _make_session(portal_env, "P001")
    second = portal_app.archive_session_dir("P001", stamp="20261116_101500")
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_archive_session_dir_missing_returns_none(portal_env: Path) -> None:
    assert portal_app.archive_session_dir("P404") is None


def test_archive_is_hidden_from_home_and_study_table(
    client: TestClient, portal_env: Path
) -> None:
    _make_session(portal_env, "P001")
    portal_app.archive_session_dir("P001", stamp="20261116_101500")
    _make_session(portal_env, "P001")
    # Even a metrics file placed directly in _archive must be ignored.
    pd.DataFrame([{"session_id": "stray", "camera_id": "camera_a"}]).to_csv(
        portal_env / "_archive" / "session_metrics.csv", index=False
    )

    rows = portal_app.session_rows()
    assert [row["session_id"] for row in rows] == ["P001"]
    table = build_all_sessions_table(portal_env)
    assert list(table["session_id"]) == ["P001"]
    assert "_archive" not in client.get("/").text


# --- finish step: ffprobe length check and archive ------------------------------


def _fake_ffprobe(duration_s: float):
    """Return a ``subprocess.run`` stand-in that answers ffprobe calls.

    Args:
        duration_s: Duration reported for every probed video.

    Returns:
        A callable with the ``subprocess.run`` signature used by portal.jobs.
    """

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess:
        if "format=duration" in command:
            stdout = json.dumps({"format": {"duration": str(duration_s)}})
        else:
            stdout = json.dumps({"streams": [{"codec_type": "video"}]})
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    return fake_run


def _upload_one_camera(client: TestClient, **form: str) -> str:
    """Init a one-camera upload and send its single chunk.

    Args:
        client: Logged-in test client.
        **form: Overrides for the init form.

    Returns:
        The job ID.
    """
    response = client.post(
        "/uploads/init", data=_init_form(**form), headers=_csrf_headers(client)
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    chunk = client.post(
        f"/uploads/{job_id}/chunk",
        data={"camera_id": "camera_a", "chunk_index": "0", "total_chunks": "1"},
        files={"chunk": ("a.mp4", b"x" * 1000, "video/mp4")},
        headers=_csrf_headers(client),
    )
    assert chunk.status_code == 200, chunk.text
    return job_id


@pytest.fixture
def no_background_job(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the background pipeline with a recorder.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        List that receives each queued job ID.
    """
    queued: list[str] = []
    monkeypatch.setattr(
        portal_app,
        "run_chunked_portal_job",
        lambda job_id, _upload_dir, _job_dir: queued.append(job_id),
    )
    return queued


def test_finish_rejects_task_past_video_length(
    client: TestClient,
    portal_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_background_job: list[str],
) -> None:
    monkeypatch.setattr(portal_jobs.subprocess, "run", _fake_ffprobe(2530.0))
    job_id = _upload_one_camera(client, blocks_text_a="grocery,13:00,13:06:00")
    response = client.post(f"/uploads/{job_id}/finish", headers=_csrf_headers(client))
    assert response.status_code == 400
    assert response.json()["detail"] == (
        'Video A task "grocery" ends at 13:06:00, but Video A is only 42:10 long. '
        "Check the task times. For 13 minutes 6 seconds, type 13:06."
    )
    assert no_background_job == []
    status = portal_jobs.read_status(portal_env / "portal_jobs" / job_id)
    assert status["state"] == "failed"
    assert "only 42:10 long" in status["message"]


def test_finish_length_message_without_hour_hint(
    client: TestClient,
    portal_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_background_job: list[str],
) -> None:
    monkeypatch.setattr(portal_jobs.subprocess, "run", _fake_ffprobe(600.0))
    job_id = _upload_one_camera(client, blocks_text_a="grocery,9:00,10:30")
    response = client.post(f"/uploads/{job_id}/finish", headers=_csrf_headers(client))
    assert response.status_code == 400
    assert response.json()["detail"] == (
        'Video A task "grocery" ends at 10:30, but Video A is only 10:00 long. '
        "Check the task times."
    )


def test_finish_queues_job_and_records_duration(
    client: TestClient,
    portal_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_background_job: list[str],
) -> None:
    monkeypatch.setattr(portal_jobs.subprocess, "run", _fake_ffprobe(2530.0))
    job_id = _upload_one_camera(client)
    response = client.post(f"/uploads/{job_id}/finish", headers=_csrf_headers(client))
    assert response.status_code == 200, response.text
    assert no_background_job == [job_id]
    meta = json.loads(
        (portal_env / "portal_jobs" / job_id / "upload_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["videos"]["camera_a"]["duration_s"] == 2530.0


def test_finish_archives_old_results_when_replacing(
    client: TestClient,
    portal_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_background_job: list[str],
) -> None:
    monkeypatch.setattr(portal_jobs.subprocess, "run", _fake_ffprobe(2530.0))
    _make_session(portal_env, "P001")
    job_id = _upload_one_camera(client, replace_existing="yes")
    response = client.post(f"/uploads/{job_id}/finish", headers=_csrf_headers(client))
    assert response.status_code == 200, response.text
    assert not (portal_env / "P001").exists()
    archived = list((portal_env / "_archive").iterdir())
    assert len(archived) == 1
    assert archived[0].name.startswith("P001_")
    assert (archived[0] / "session_metrics.csv").is_file()
    meta = json.loads(
        (portal_env / "portal_jobs" / job_id / "upload_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["archived_previous_results"] == str(archived[0])


def test_finish_refuses_session_created_meanwhile(
    client: TestClient,
    portal_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_background_job: list[str],
) -> None:
    monkeypatch.setattr(portal_jobs.subprocess, "run", _fake_ffprobe(2530.0))
    job_id = _upload_one_camera(client)
    _make_session(portal_env, "P001")
    response = client.post(f"/uploads/{job_id}/finish", headers=_csrf_headers(client))
    assert response.status_code == 409
    assert "already has results" in response.json()["detail"]
    assert (portal_env / "P001" / "session_metrics.csv").is_file()
    assert no_background_job == []


def test_finish_does_not_archive_when_length_check_fails(
    client: TestClient,
    portal_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_background_job: list[str],
) -> None:
    monkeypatch.setattr(portal_jobs.subprocess, "run", _fake_ffprobe(100.0))
    _make_session(portal_env, "P001")
    job_id = _upload_one_camera(client, replace_existing="yes")
    response = client.post(f"/uploads/{job_id}/finish", headers=_csrf_headers(client))
    assert response.status_code == 400
    assert (portal_env / "P001" / "session_metrics.csv").is_file()
    assert not (portal_env / "_archive").exists()


def test_probe_video_duration_parses_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(portal_jobs.subprocess, "run", _fake_ffprobe(12.5))
    assert portal_jobs.probe_video_duration(tmp_path / "a.mp4") == 12.5


# --- job page -----------------------------------------------------------------


def _job_with_old_results(root: Path, state: str) -> Path:
    """Create a job whose session folder holds a previous run's table.

    Args:
        root: Output root.
        state: Job state to record.

    Returns:
        The job folder.
    """
    job_dir = root / "portal_jobs" / "job-001"
    write_status(job_dir, {"state": state, "message": "msg"})
    (job_dir / "job_config.json").write_text(
        json.dumps({"session_id": "P001"}), encoding="utf-8"
    )
    session = _make_session(root, "P001")
    pd.DataFrame(
        [{"camera_id": "camera_a", "block": "grocery", "coverage_both": 0.9}]
    ).to_csv(session / "session_metrics.csv", index=False)
    return job_dir


@pytest.mark.parametrize("state", ["running", "failed", "queued", "uploading"])
def test_job_page_hides_results_unless_complete(
    client: TestClient, portal_env: Path, state: str
) -> None:
    _job_with_old_results(portal_env, state)
    text = client.get("/jobs/job-001").text
    assert "Both people tracked, per block" not in text
    assert "/gaze-label" not in text


def test_job_page_shows_results_when_complete(
    client: TestClient, portal_env: Path
) -> None:
    _job_with_old_results(portal_env, "complete")
    text = client.get("/jobs/job-001").text
    assert "Both people tracked, per block" in text
    assert "/jobs/job-001/gaze-label" in text


def test_job_page_lists_dropped_parts(client: TestClient, portal_env: Path) -> None:
    job_dir = portal_env / "portal_jobs" / "job-001"
    write_status(job_dir, {"state": "running", "message": "msg"})
    metadata = _metadata(
        blocks_a="<b>grocery</b>,16:00,26:50",
        exclusions_a="18:06,18:21,experimenter\n18:25,20:00,experimenter",
    )
    portal_app._write_submission_metadata(job_dir, metadata)
    text = client.get("/jobs/job-001").text
    assert "Skipped because shorter than 10 s" in text
    assert "b_grocery_b_part2" in text
    assert "Video A, 18:21 to 18:25 (4 s)" in text
