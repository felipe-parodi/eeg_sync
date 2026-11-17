"""Tests for the Sessions home page, the new-session page and session reports.

The home page lists every processed session found under ``OUTPUT_ROOT`` (one
folder with a ``session_metrics.csv``), so it survives portal restarts. Jobs
from the current run are pinned above it. Session folders are served through
a confined file route so the per-session ``report.html`` and its figures and
clips open in the browser.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import portal.app as portal_app
from portal.jobs import write_status

EVIL = "<script>alert(1)</script>"
EVIL_ESCAPED = "&lt;script&gt;alert(1)&lt;/script&gt;"


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


def _make_session(
    root: Path,
    name: str,
    coverages: list[float],
    cameras: tuple[str, ...] = ("camera_a",),
    quality: list[str] | None = None,
) -> Path:
    """Create a processed session folder with a session_metrics.csv.

    Args:
        root: Output root.
        name: Session folder name.
        coverages: ``coverage_both`` per block row.
        cameras: Camera subfolders to create.
        quality: Optional ``quality_verdict`` per row.

    Returns:
        The session folder.
    """
    session = root / name
    for camera in cameras:
        (session / camera).mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "session_id": name,
            "camera_id": cameras[0],
            "block": [f"block{i}" for i in range(len(coverages))],
            "coverage_both": coverages,
        }
    )
    if quality is not None:
        table["quality_verdict"] = quality
    table.to_csv(session / "session_metrics.csv", index=False)
    return session


# --- overall verdict ------------------------------------------------------


def test_verdict_fallback_all_blocks_covered() -> None:
    table = pd.DataFrame({"coverage_both": [0.5, 0.9, 0.75]})

    verdict = portal_app.overall_session_verdict(table)

    assert verdict == ("ok", "Looks OK")


def test_verdict_fallback_counts_low_blocks() -> None:
    one = pd.DataFrame({"coverage_both": [0.49, 0.9]})
    two = pd.DataFrame({"coverage_both": [0.1, 0.9, 0.2]})

    assert portal_app.overall_session_verdict(one) == ("check", "Check 1 block")
    assert portal_app.overall_session_verdict(two) == ("check", "Check 2 blocks")


def test_verdict_missing_coverage_counts_as_low() -> None:
    table = pd.DataFrame({"coverage_both": [float("nan"), 0.9]})

    assert portal_app.overall_session_verdict(table) == ("check", "Check 1 block")


def test_verdict_prefers_quality_verdict_column() -> None:
    # coverage alone would say "Looks OK"; the quality column wins.
    base = {"coverage_both": [0.9, 0.9, 0.9]}
    insufficient = pd.DataFrame(
        {**base, "quality_verdict": ["usable", "check", "insufficient"]}
    )
    check = pd.DataFrame({**base, "quality_verdict": ["usable", "Check", "usable"]})
    usable = pd.DataFrame({**base, "quality_verdict": ["usable"] * 3})

    assert portal_app.overall_session_verdict(insufficient) == (
        "insufficient",
        "Not enough data",
    )
    assert portal_app.overall_session_verdict(check) == ("check", "Check")
    assert portal_app.overall_session_verdict(usable) == ("ok", "Looks OK")


def test_verdict_empty_table_is_not_enough_data() -> None:
    table = pd.DataFrame({"coverage_both": []})

    assert portal_app.overall_session_verdict(table) == (
        "insufficient",
        "Not enough data",
    )


# --- sessions list ----------------------------------------------------------


def test_session_rows_only_include_folders_with_metrics(portal_env: Path) -> None:
    _make_session(portal_env, "1019v3", [0.9], cameras=("camera_a", "camera_b"))
    _make_session(portal_env, "1020v3", [0.2, 0.8])
    write_status(portal_env / "portal_jobs" / "job-1", {"state": "running"})
    (portal_env / "session_20260101_1200").mkdir()
    (portal_env / "all_sessions.csv").write_text("a\n1\n", encoding="utf-8")

    rows = portal_app.session_rows()

    names = sorted(row["session_id"] for row in rows)
    assert names == ["1019v3", "1020v3"]
    by_name = {row["session_id"]: row for row in rows}
    assert by_name["1019v3"]["cameras"] == ["A", "B"]
    assert by_name["1019v3"]["verdict_kind"] == "ok"
    assert by_name["1020v3"]["verdict_label"] == "Check 1 block"
    assert by_name["1020v3"]["processed"]


def test_session_rows_newest_first(portal_env: Path) -> None:
    old = _make_session(portal_env, "old", [0.9])
    new = _make_session(portal_env, "new", [0.9])
    os.utime(old / "session_metrics.csv", (1_000_000_000, 1_000_000_000))
    os.utime(new / "session_metrics.csv", (1_700_000_000, 1_700_000_000))

    rows = portal_app.session_rows()

    assert [row["session_id"] for row in rows] == ["new", "old"]


def test_session_rows_missing_output_root(
    portal_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(portal_app, "OUTPUT_ROOT", portal_env / "nope")

    assert portal_app.session_rows() == []


def test_home_lists_sessions_and_hides_other_folders(
    client: TestClient, portal_env: Path
) -> None:
    _make_session(portal_env, "1019v3", [0.9, 0.95])
    _make_session(portal_env, "1020v3", [0.1, 0.2, 0.9])
    (portal_env / "session_20260101_1200").mkdir()

    response = client.get("/")

    assert response.status_code == 200
    text = response.text
    assert "<title>Sessions</title>" in text
    assert 'href="/sessions/1019v3"' in text
    assert 'href="/sessions/1020v3"' in text
    assert "Looks OK" in text
    assert "Check 2 blocks" in text
    assert "session_20260101_1200" not in text
    assert 'href="/new"' in text
    assert "New session" in text
    assert 'href="/metrics/all_sessions.csv"' in text
    assert "Download all sessions table" in text
    assert "What the results contain" in text
    assert "Movement synchrony" in text


def test_home_empty_state(client: TestClient) -> None:
    text = client.get("/").text

    assert "No processed sessions yet." in text
    assert "No jobs yet." in text


def test_home_pins_active_jobs_above_finished_ones(
    client: TestClient, portal_env: Path
) -> None:
    jobs = portal_env / "portal_jobs"
    write_status(jobs / "a-done", {"state": "complete", "message": "Finished."})
    write_status(jobs / "b-run", {"state": "running", "message": "Finding people."})
    write_status(
        jobs / "c-wait", {"state": "awaiting_annotation", "message": "Label please."}
    )

    text = client.get("/").text

    assert 'href="/jobs/b-run"' in text
    assert 'href="/jobs/c-wait"' in text
    assert "Finding people." in text
    assert "Running" in text
    assert "Waiting for you" in text
    assert "Done" in text
    assert text.index("/jobs/b-run") < text.index("/jobs/a-done")
    assert text.index("/jobs/c-wait") < text.index("/jobs/a-done")


def test_home_escapes_session_name_and_job_message(
    client: TestClient, portal_env: Path
) -> None:
    _make_session(portal_env, "odd&name'x", [0.9])
    write_status(
        portal_env / "portal_jobs" / "job-1", {"state": "failed", "message": EVIL}
    )

    text = client.get("/").text

    assert EVIL not in text
    assert EVIL_ESCAPED in text
    assert "odd&amp;name&#39;x" in text
    assert "odd&name'x" not in text
    # A folder name the session route cannot address gets no link.
    assert 'href="/sessions/odd' not in text


def test_header_has_logout_and_portal_name(client: TestClient) -> None:
    text = client.get("/").text

    assert 'id="logout-button"' in text
    assert portal_app.PORTAL_DISPLAY_NAME in text
    assert portal_app.csrf_token_for_session(client.cookies["portal_session"]) in text


def test_login_page_has_no_logout_button(portal_env: Path) -> None:
    text = TestClient(portal_app.app).get("/login").text

    assert 'id="logout-button"' not in text


# --- new session page ---------------------------------------------------------


def test_new_session_page_renders_upload_form(client: TestClient) -> None:
    response = client.get("/new")

    assert response.status_code == 200
    text = response.text
    assert "<title>New session</title>" in text
    for element_id in ('id="upload-form"', 'id="session_id"', 'id="upload-status"'):
        assert element_id in text
    assert 'src="/static/index.js"' in text


def test_new_session_page_requires_auth(portal_env: Path) -> None:
    response = TestClient(portal_app.app).get("/new", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


# --- session report and files -----------------------------------------------


@pytest.fixture
def report_session(portal_env: Path) -> Path:
    """Create a session with a report, a figure, a clip and a non-allowed file.

    Args:
        portal_env: Temporary output root.

    Returns:
        The session folder.
    """
    session = _make_session(portal_env, "1019v3", [0.9])
    (session / "report.html").write_text(
        '<html><body><img src="camera_a/fig.png"></body></html>', encoding="utf-8"
    )
    (session / "camera_a" / "fig.png").write_bytes(b"\x89PNG fake")
    (session / "camera_a" / "overlay_storybook.mp4").write_bytes(bytes(range(256)))
    (session / "camera_a" / "gaze_heatmaps.npz").write_bytes(b"npz")
    (session / "camera_a" / "notes.py").write_text("x = 1", encoding="utf-8")
    other = _make_session(portal_env, "other", [0.9])
    (other / "secret.csv").write_text("secret\n", encoding="utf-8")
    return session


def test_session_page_redirects_to_report(
    client: TestClient, report_session: Path
) -> None:
    response = client.get("/sessions/1019v3", follow_redirects=False)

    assert response.status_code in (302, 303, 307)
    assert response.headers["location"] == "/sessions/1019v3/files/report.html"


def test_session_report_is_served(client: TestClient, report_session: Path) -> None:
    response = client.get("/sessions/1019v3/files/report.html")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'src="camera_a/fig.png"' in response.text


def test_session_file_serves_allowed_figure(
    client: TestClient, report_session: Path
) -> None:
    response = client.get("/sessions/1019v3/files/camera_a/fig.png")

    assert response.status_code == 200
    assert response.content == b"\x89PNG fake"
    assert response.headers["content-type"].startswith("image/png")


def test_session_clip_supports_range_requests(
    client: TestClient, report_session: Path
) -> None:
    response = client.get(
        "/sessions/1019v3/files/camera_a/overlay_storybook.mp4",
        headers={"Range": "bytes=0-9"},
    )

    assert response.status_code == 206
    assert response.content == bytes(range(10))
    assert response.headers["content-type"].startswith("video/mp4")


@pytest.mark.parametrize(
    "path",
    [
        "/sessions/1019v3/files/..%2Fother%2Fsecret.csv",
        "/sessions/1019v3/files/camera_a%2F..%2F..%2Fother%2Fsecret.csv",
        "/sessions/1019v3/files/%2E%2E/other/secret.csv",
        "/sessions/1019v3/files/camera_a/notes.py",
        "/sessions/1019v3/files/camera_a/gaze_heatmaps.npz",
        "/sessions/1019v3/files/camera_a/missing.png",
        "/sessions/1019v3/files/camera_a",
        "/sessions/nope/files/report.html",
        "/sessions/portal_jobs/files/report.html",
        "/sessions/nope",
    ],
)
def test_session_files_reject_bad_paths(
    client: TestClient, report_session: Path, path: str
) -> None:
    assert client.get(path, follow_redirects=False).status_code == 404


def test_session_file_rejects_absolute_path(
    client: TestClient, report_session: Path
) -> None:
    target = (report_session.parent / "other" / "secret.csv").as_posix()

    response = client.get(f"/sessions/1019v3/files/{target}")

    assert response.status_code == 404


def test_session_routes_require_auth(report_session: Path) -> None:
    anonymous = TestClient(portal_app.app)

    for path in ("/sessions/1019v3", "/sessions/1019v3/files/camera_a/fig.png"):
        response = anonymous.get(path, follow_redirects=False)
        assert response.status_code == 303, path
        assert response.headers["location"] == "/login"


def test_session_file_rejects_symlink_escaping_session(
    client: TestClient, report_session: Path
) -> None:
    secret = report_session.parent / "other" / "secret.csv"
    link = report_session / "camera_a" / "linked.csv"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("creating symlinks is not permitted here")

    response = client.get("/sessions/1019v3/files/camera_a/linked.csv")

    assert response.status_code == 404


# --- study task table download -----------------------------------------------


def test_home_links_study_task_table(client: TestClient) -> None:
    text = client.get("/").text

    assert 'href="/metrics/all_sessions_task_metrics.csv"' in text
    assert "Download study task table" in text


def test_study_task_table_is_served(client: TestClient, portal_env: Path) -> None:
    (portal_env / "all_sessions_task_metrics.csv").write_text(
        "session_id,camera_id,task\n1019v3,camera_a,grocery\n", encoding="utf-8"
    )

    response = client.get("/metrics/all_sessions_task_metrics.csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "all_sessions_task_metrics.csv" in response.headers["content-disposition"]
    assert "1019v3,camera_a,grocery" in response.text


def test_study_task_table_missing_is_404(client: TestClient) -> None:
    assert client.get("/metrics/all_sessions_task_metrics.csv").status_code == 404


def test_study_task_table_requires_auth(portal_env: Path) -> None:
    (portal_env / "all_sessions_task_metrics.csv").write_text("x\n", encoding="utf-8")
    anonymous = TestClient(portal_app.app)

    response = anonymous.get(
        "/metrics/all_sessions_task_metrics.csv", follow_redirects=False
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_session_report_step_writes_study_tables(tmp_path: Path) -> None:
    from portal.jobs import PortalJobConfig, SessionBlock, build_session_report_step

    config = PortalJobConfig(
        job_id="j",
        session_id="1019v3",
        video_a=tmp_path / "a.mp4",
        video_b=None,
        job_dir=tmp_path / "j",
        output_root=tmp_path / "output",
        session_config_path=tmp_path / "j" / "session_config.json",
        blocks=[SessionBlock(name="grocery", start_s=0.0, end_s=60.0)],
    )

    command = build_session_report_step(config).argv

    root = command[command.index("--all-sessions-root") + 1]
    assert Path(root) == tmp_path / "output"
