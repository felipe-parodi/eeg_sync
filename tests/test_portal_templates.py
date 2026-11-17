"""Route-level rendering tests for the portal pages.

Every HTML page is fetched through FastAPI's ``TestClient`` with a real
session cookie, so these tests cover routing, auth, templates, and escaping
together. Untrusted strings (job messages, block names, log lines, manifest
filenames) are seeded with a script tag and must come back escaped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import portal.app as portal_app
from portal.jobs import write_status

EVIL = "<script>alert(1)</script>"
EVIL_ESCAPED = "&lt;script&gt;alert(1)&lt;/script&gt;"
JOB_ID = "job-001"


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
    """Return a logged-in test client (lifespan not run, so no job purge).

    Args:
        portal_env: Temporary portal folders.

    Returns:
        A ``TestClient`` carrying a valid ``portal_session`` cookie.
    """
    test_client = TestClient(portal_app.app)
    test_client.cookies.set("portal_session", portal_app.create_session_token())
    return test_client


def _csrf(client: TestClient) -> str:
    """Return the CSRF token tied to the client's session cookie."""
    return portal_app.csrf_token_for_session(client.cookies["portal_session"])


def _finished_job(root: Path, block_name: str = "storybook") -> Path:
    """Create a finished job with a session_metrics.csv and a log file."""
    job_dir = root / "portal_jobs" / JOB_ID
    write_status(job_dir, {"state": "complete", "message": f"Done {EVIL}"})
    (job_dir / "job_config.json").write_text(
        json.dumps({"session_id": "1012"}), encoding="utf-8"
    )
    (job_dir / "job.log").write_text(f"step 1 ok\n{EVIL} & done\n", encoding="utf-8")
    session = root / "1012"
    session.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "session_id": "1012",
                "camera_id": "camera_a",
                "block": block_name,
                "coverage_both": 0.21,
                "low_coverage": True,
            },
            {
                "session_id": "1012",
                "camera_id": "camera_b",
                "block": "grocery",
                "coverage_both": 0.9,
                "low_coverage": False,
            },
        ]
    ).to_csv(session / "session_metrics.csv", index=False)
    return job_dir


def _anchor_job(root: Path) -> Path:
    """Create a job awaiting annotation with one anchor manifest."""
    job_dir = root / "portal_jobs" / JOB_ID
    write_status(
        job_dir,
        {"state": "awaiting_annotation", "message": "Label the parent."},
    )
    camera_dir = job_dir / "anchors" / "camera_a"
    camera_dir.mkdir(parents=True)
    (camera_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "anchor_idx": 0,
                    "image_filename": f'anchor_000.jpg"{EVIL}',
                    "frame_idx": 10,
                    "timestamp_s": 1.25,
                    "tracks": [
                        {"track_id": 1, "color_bgr": [0, 255, 0]},
                        {"track_id": 2, "color_bgr": [0, 165, 255]},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return job_dir


def _gaze_job(root: Path) -> Path:
    """Create a finished job with one camera ready for gaze labeling."""
    frames = root / "1012" / "camera_a" / "frames"
    frames.mkdir(parents=True)
    index, tracks = [], []
    for i in range(120):
        t = 100.0 + i / 6
        index.append((i, t, f"frame_{i + 1:06d}.jpg"))
        g = int(round(t * 6))
        tracks.append((g, t, 0, "parent", 10, 10, 100, 300, False))
        tracks.append((g, t, 1, "child", 200, 100, 260, 300, False))
    pd.DataFrame(index, columns=["frame_idx", "timestamp_s", "image_name"]).to_csv(
        frames / "frame_index.csv", index=False
    )
    pd.DataFrame(
        tracks,
        columns=[
            "frame_idx",
            "timestamp_s",
            "track_id",
            "track_label",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "interpolated",
        ],
    ).to_csv(root / "1012" / "camera_a" / "tracks_2d_smooth.csv", index=False)
    job_dir = root / "portal_jobs" / JOB_ID
    write_status(job_dir, {"state": "complete", "message": "Done."})
    (job_dir / "job_config.json").write_text(
        json.dumps(
            {
                "session_id": "1012",
                "frame_rate": 6.0,
                "blocks_by_camera": {
                    "camera_a": [{"name": EVIL, "start_s": 100.0, "end_s": 119.0}]
                },
            }
        ),
        encoding="utf-8",
    )
    return job_dir


def test_login_page_renders_without_auth(portal_env: Path) -> None:
    response = TestClient(portal_app.app).get("/login")

    assert response.status_code == 200
    text = response.text
    assert "<title>Login</title>" in text
    assert "Video Processing Portal" in text
    assert '<form method="post" action="/login">' in text
    assert 'id="password"' in text


def test_pages_redirect_to_login_without_cookie(portal_env: Path) -> None:
    response = TestClient(portal_app.app).get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_index_renders_sessions_home_and_escapes_job_rows(
    client: TestClient, portal_env: Path
) -> None:
    write_status(
        portal_env / "portal_jobs" / JOB_ID,
        {"state": "failed", "message": EVIL},
    )

    response = client.get("/")

    assert response.status_code == 200
    text = response.text
    assert "<title>Sessions</title>" in text
    assert 'id="logout-button"' in text
    assert 'href="/new"' in text
    assert f'href="/jobs/{JOB_ID}"' in text
    assert "Download all sessions table" in text
    assert EVIL not in text
    assert EVIL_ESCAPED in text


def test_new_session_page_renders_upload_form(client: TestClient) -> None:
    response = client.get("/new")

    assert response.status_code == 200
    text = response.text
    assert "<title>New session</title>" in text
    for element_id in (
        'id="upload-form"',
        'id="logout-button"',
        'id="session_id"',
        'id="blocks_text_a"',
        'id="blocks_text_b"',
        'id="exclusions_text_a"',
        'id="exclusions_text_b"',
        'id="upload-status"',
        'id="upload-progress"',
    ):
        assert element_id in text
    assert f'const CSRF_TOKEN = "{_csrf(client)}";' in text


def test_index_shows_empty_job_table(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "No jobs yet." in response.text


def test_job_detail_renders_coverage_and_escapes(
    client: TestClient, portal_env: Path
) -> None:
    _finished_job(portal_env, block_name=EVIL)

    response = client.get(f"/jobs/{JOB_ID}")

    assert response.status_code == 200
    text = response.text
    assert f"<title>Job {JOB_ID}</title>" in text
    assert "Both people tracked, per block" in text
    assert "21%" in text
    assert "90%" in text
    assert "Low: check the overlay clips" in text
    assert f'href="/jobs/{JOB_ID}/gaze-label"' in text
    assert f'href="/jobs/{JOB_ID}/log"' in text
    assert 'href="/"' in text
    assert EVIL not in text
    assert f"Done {EVIL_ESCAPED}" in text
    assert f"<td>{EVIL_ESCAPED}</td>" in text


def test_job_detail_running_job_auto_refreshes(
    client: TestClient, portal_env: Path
) -> None:
    write_status(
        portal_env / "portal_jobs" / JOB_ID,
        {"state": "running", "message": "Tracking poses."},
    )

    text = client.get(f"/jobs/{JOB_ID}").text

    assert "This page refreshes automatically while the job runs." in text
    assert "window.location.reload()" in text
    assert "/annotate" not in text


def test_job_log_renders_escaped(client: TestClient, portal_env: Path) -> None:
    _finished_job(portal_env)

    response = client.get(f"/jobs/{JOB_ID}/log")

    assert response.status_code == 200
    text = response.text
    assert "<title>Job Log</title>" in text
    assert "<pre>" in text
    assert "step 1 ok" in text
    assert EVIL not in text
    assert f"{EVIL_ESCAPED} &amp; done" in text


def test_job_log_missing_is_404(client: TestClient, portal_env: Path) -> None:
    write_status(portal_env / "portal_jobs" / JOB_ID, {"state": "running"})

    assert client.get(f"/jobs/{JOB_ID}/log").status_code == 404


def test_annotate_page_renders_buttons_and_csrf(
    client: TestClient, portal_env: Path
) -> None:
    _anchor_job(portal_env)

    response = client.get(f"/jobs/{JOB_ID}/annotate")

    assert response.status_code == 200
    text = response.text
    assert f"<title>Annotate {JOB_ID}</title>" in text
    assert "annotate-form" in text
    assert f"/jobs/{JOB_ID}/annotations" in text
    assert 'class="anchor-card"' in text
    assert 'class="track-btn"' in text
    assert 'class="skip-btn"' in text
    assert "Track 1 is parent" in text
    assert "Track 2 is parent" in text
    assert "frame 10 (t=1.25s)" in text
    assert "background: rgba(0,255,0,0.12); color: #17202a;" in text
    assert "Submit annotations and continue processing" in text
    assert _csrf(client) in text
    assert EVIL not in text


def test_annotate_page_404_without_manifest(
    client: TestClient, portal_env: Path
) -> None:
    write_status(portal_env / "portal_jobs" / JOB_ID, {"state": "running"})

    assert client.get(f"/jobs/{JOB_ID}/annotate").status_code == 404


def test_gaze_label_page_renders_and_escapes_json(
    client: TestClient, portal_env: Path
) -> None:
    _gaze_job(portal_env)

    response = client.get(f"/jobs/{JOB_ID}/gaze-label")

    assert response.status_code == 200
    text = response.text
    assert f"<title>Gaze labels {JOB_ID}</title>" in text
    assert f"What are they looking at? ({JOB_ID})" in text
    for element_id in ('id="labeler"', 'id="cards"', 'id="save"', 'id="gaze-data"'):
        assert element_id in text
    assert "The other person's face" in text
    assert '"image_name": "frame_' in text
    assert "/gaze-frames/" in text
    assert _csrf(client) in text
    # The block name sits inside a JSON <script> block: it must not be able
    # to close that block.
    assert "alert(1)</script>" not in text


def test_static_stylesheet_is_public(portal_env: Path) -> None:
    response = TestClient(portal_app.app).get("/login")
    assert response.status_code == 200
    for href in _static_refs(response.text):
        asset = TestClient(portal_app.app).get(href)
        assert asset.status_code == 200, href


def _static_refs(text: str) -> list[str]:
    """Return every /static/ URL referenced by a page."""
    import re

    return re.findall(r'(?:href|src)="(/static/[^"]+)"', text)


def test_every_page_static_reference_resolves(
    client: TestClient, portal_env: Path
) -> None:
    _finished_job(portal_env)
    pages = ["/login", "/", "/new", f"/jobs/{JOB_ID}", f"/jobs/{JOB_ID}/log"]
    for page in pages:
        response = client.get(page)
        assert response.status_code == 200, page
        for href in _static_refs(response.text):
            assert client.get(href).status_code == 200, href


def test_static_mount_serves_only_css_and_js(tmp_path: Path) -> None:
    static_cls = portal_app.PortalStaticFiles
    (tmp_path / "site.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "page.js").write_text("1;", encoding="utf-8")
    (tmp_path / "secret.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "notes.html").write_text("<p>x</p>", encoding="utf-8")
    mini = FastAPI()
    mini.mount("/static", static_cls(directory=tmp_path), name="static")
    mini_client = TestClient(mini)

    assert mini_client.get("/static/site.css").status_code == 200
    assert mini_client.get("/static/page.js").status_code == 200
    assert mini_client.get("/static/secret.py").status_code == 404
    assert mini_client.get("/static/notes.html").status_code == 404
    assert mini_client.get("/static/").status_code == 404


def test_portal_static_dir_holds_only_css_and_js() -> None:
    static_dir = Path(portal_app.__file__).with_name("static")
    assert static_dir.is_dir()
    suffixes = {p.suffix for p in static_dir.rglob("*") if p.is_file()}
    assert suffixes <= {".css", ".js"}


def test_static_assets_have_browser_content_types(portal_env: Path) -> None:
    anonymous = TestClient(portal_app.app)

    css = anonymous.get("/static/portal.css")
    js = anonymous.get("/static/index.js")

    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert anonymous.get("/static/missing.js").status_code == 404
    assert anonymous.get("/static/../app.py").status_code == 404
