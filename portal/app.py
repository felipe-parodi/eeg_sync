"""FastAPI web app for the local processing portal."""

from __future__ import annotations

import hmac
import html
import json
import math
import os
import re
import secrets
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from .jobs import (
    PORTAL_DEFAULT_FRAME_RATE,
    PORTAL_DEFAULT_MAX_PERSONS,
    PORTAL_DEFAULT_MODEL_PATH,
    CompressionSettings,
    ExclusionWindow,
    PortalJobConfig,
    SessionBlock,
    final_upload_path_for,
    mark_chunk_received,
    parse_timecode,
    read_status,
    run_portal_job,
    split_blocks_for_exclusions,
    validate_direct_chunked_upload,
    write_status,
)

PORTAL_DISPLAY_NAME = "Video Processing Portal"
REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = REPO_ROOT / "video_inference" / "data" / "portal_uploads"
JOB_ROOT = REPO_ROOT / "video_inference" / "output" / "portal_jobs"
OUTPUT_ROOT = REPO_ROOT / "video_inference" / "output"
CHUNK_SIZE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_JOB_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024
DEFAULT_SESSION_TTL_SECONDS = 8 * 60 * 60
LOGIN_RATE_LIMIT_ATTEMPTS = 5
LOGIN_RATE_LIMIT_WINDOW_SECONDS = 5 * 60


@dataclass
class SessionState:
    """Server-side state for one authenticated portal session."""

    expires_at: float
    csrf_token: str


SESSION_TOKENS: dict[str, SessionState] = {}
LOGIN_FAILURES: dict[str, list[float]] = {}
# This is intentionally process-local. Run the portal as a single uvicorn
# worker; multiple workers would each have their own GPU job lock.
JOB_RUN_LOCK = threading.Lock()


def validate_startup_configuration() -> None:
    """Fail startup if required portal secrets are missing."""
    portal_password()
    portal_secret_key()


@asynccontextmanager
async def portal_lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate portal configuration before accepting requests."""
    validate_startup_configuration()
    yield


app = FastAPI(title=PORTAL_DISPLAY_NAME, lifespan=portal_lifespan)


def portal_password() -> str:
    """Return the configured shared portal password."""
    password = os.environ.get("PORTAL_PASSWORD")
    if not password:
        raise RuntimeError("PORTAL_PASSWORD must be set")
    return password


def portal_secret_key() -> str:
    """Return the configured secret used to derive session tokens."""
    secret = os.environ.get("PORTAL_SECRET_KEY")
    if not secret:
        raise RuntimeError("PORTAL_SECRET_KEY must be set")
    return secret


def session_ttl_seconds() -> int:
    """Return the session lifetime in seconds."""
    return int(
        os.environ.get("PORTAL_SESSION_TTL_SECONDS", DEFAULT_SESSION_TTL_SECONDS)
    )


def cookie_secure() -> bool:
    """Return whether session cookies should require HTTPS."""
    return os.environ.get("PORTAL_COOKIE_SECURE", "true").lower() not in {
        "0",
        "false",
        "no",
    }


def max_job_upload_bytes() -> int:
    """Return the maximum total upload size accepted for one portal job."""
    return int(
        os.environ.get("PORTAL_MAX_JOB_UPLOAD_BYTES", DEFAULT_MAX_JOB_UPLOAD_BYTES)
    )


def max_chunk_upload_bytes() -> int:
    """Return the maximum accepted size for one upload chunk."""
    return int(os.environ.get("PORTAL_MAX_CHUNK_BYTES", CHUNK_SIZE_BYTES))


def validate_upload_size_limits(video_a_size: int, video_b_size: int) -> None:
    """Validate browser-declared upload sizes against portal caps."""
    if video_a_size < 0 or video_b_size < 0:
        raise ValueError("Uploaded video sizes must be non-negative")
    total_size = video_a_size + video_b_size
    max_total = max_job_upload_bytes()
    if total_size > max_total:
        raise ValueError(
            f"Upload is {total_size} bytes, above the per-job limit of {max_total} bytes"
        )
    max_chunk = max_chunk_upload_bytes()
    if max_chunk > CHUNK_SIZE_BYTES:
        raise ValueError(
            "PORTAL_MAX_CHUNK_BYTES cannot exceed the browser chunk size "
            f"({CHUNK_SIZE_BYTES} bytes)"
        )


def create_session_token() -> str:
    """Create a random server-side session token with an expiry."""
    token = hmac.new(
        portal_secret_key().encode("utf-8"),
        secrets.token_bytes(32),
        "sha256",
    ).hexdigest()
    SESSION_TOKENS[token] = SessionState(
        expires_at=time.time() + session_ttl_seconds(),
        csrf_token=secrets.token_urlsafe(32),
    )
    return token


def validate_session_token(token: str) -> bool:
    """Return whether a session token exists and has not expired."""
    if not token:
        return False
    state = SESSION_TOKENS.get(token)
    if state is None:
        return False
    if state.expires_at < time.time():
        SESSION_TOKENS.pop(token, None)
        return False
    return True


def csrf_token_for_session(token: str) -> str:
    """Return the CSRF token tied to an authenticated session."""
    state = SESSION_TOKENS.get(token)
    if state is None:
        raise HTTPException(status_code=401, detail="Login required")
    return state.csrf_token


def _is_json_or_upload_request(request: Request) -> bool:
    """Return whether failed auth should be reported as JSON/XHR status."""
    accept = request.headers.get("accept", "")
    return request.url.path.startswith("/uploads/") or "application/json" in accept


def require_auth(request: Request) -> str:
    """Require the shared portal login cookie and return its session token."""
    cookie = str(request.cookies.get("portal_session", ""))
    if not validate_session_token(cookie):
        if _is_json_or_upload_request(request):
            raise HTTPException(status_code=401, detail="Login required")
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return cookie


def require_csrf(request: Request) -> str:
    """Require an authenticated session plus a matching CSRF header."""
    session_token = require_auth(request)
    expected = csrf_token_for_session(session_token)
    supplied = request.headers.get("x-csrf-token", "")
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    return session_token


def _html_page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, sans-serif;
      background: #f6f7f9;
      color: #17202a;
    }}
    body {{ margin: 0; }}
    header {{
      background: #143642;
      color: white;
      padding: 16px 24px;
      font-size: 18px;
      font-weight: 700;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }}
    form, .panel {{
      background: white;
      border: 1px solid #d9dee5;
      border-radius: 6px;
      padding: 20px;
      margin-bottom: 18px;
    }}
    label {{
      display: block;
      font-weight: 700;
      margin-top: 14px;
      margin-bottom: 6px;
    }}
    .info-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      margin-left: 6px;
      border-radius: 50%;
      border: 1px solid #8da0b3;
      color: #176b87;
      font-size: 12px;
      font-weight: 700;
      cursor: help;
    }}
    input, textarea, button {{
      font: inherit;
      box-sizing: border-box;
    }}
    input[type="text"], input[type="email"], input[type="password"], input[type="file"], textarea {{
      width: 100%;
      border: 1px solid #b9c1cc;
      border-radius: 4px;
      padding: 10px;
      background: white;
    }}
    input.file-input-native {{
      position: absolute;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    }}
    textarea {{ min-height: 120px; resize: vertical; }}
    .file-drop-zone {{
      border: 2px dashed #8da0b3;
      border-radius: 6px;
      padding: 18px;
      background: #f9fbfc;
      cursor: pointer;
    }}
    .file-drop-zone strong, .file-drop-zone span {{
      display: block;
    }}
    .file-drop-zone.dragover {{
      border-color: #176b87;
      background: #eef8fb;
    }}
    .file-name {{
      margin-top: 8px;
    }}
    button {{
      border: 0;
      border-radius: 4px;
      background: #176b87;
      color: white;
      padding: 11px 16px;
      margin-top: 16px;
      font-weight: 700;
      cursor: pointer;
    }}
    .top-actions {{
      display: flex;
      justify-content: flex-end;
      margin-bottom: 12px;
    }}
    .top-actions button {{ margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border-bottom: 1px solid #d9dee5; padding: 10px; text-align: left; }}
    .muted {{ color: #5d6d7e; font-size: 14px; }}
    .warning {{
      border-left: 4px solid #c77700;
      background: #fff8ec;
      padding: 10px 12px;
      margin-bottom: 16px;
    }}
    .error {{ color: #9b1c1c; font-weight: 700; }}
    .output-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .output-card {{
      border: 1px solid #d9dee5;
      border-radius: 6px;
      padding: 12px;
      background: #fbfcfd;
    }}
    .output-card h3 {{
      margin: 0 0 8px 0;
      font-size: 16px;
    }}
    .output-card ul {{
      margin: 8px 0 0 18px;
      padding: 0;
    }}
  </style>
</head>
<body>
  <header>{PORTAL_DISPLAY_NAME}</header>
  <main>{body}</main>
</body>
</html>""")


def _info_icon(text: str) -> str:
    return f'<span class="info-icon" title="{html.escape(text)}">?</span>'


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError("name cannot be empty")
    return cleaned


def _parse_blocks_text(text: str) -> list[SessionBlock]:
    blocks: list[SessionBlock] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) not in {3, 4}:
            raise ValueError(
                f"Block line {line_number} must be: name,start,end[,color]"
            )
        name, start_raw, end_raw = parts[:3]
        start_s = parse_timecode(start_raw)
        end_s = parse_timecode(end_raw)
        if end_s <= start_s:
            raise ValueError(f"Block line {line_number} end must be after start")
        blocks.append(
            SessionBlock(
                name=_slug(name),
                start_s=start_s,
                end_s=end_s,
                color=parts[3] if len(parts) == 4 and parts[3] else "gray",
            )
        )
    if not blocks:
        raise ValueError("At least one analysis block is required")
    return blocks


def _parse_exclusions_text(text: str) -> list[ExclusionWindow]:
    exclusions: list[ExclusionWindow] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        if len(parts) < 2:
            raise ValueError(
                f"Exclusion line {line_number} must be: start,end[,reason]"
            )
        start_s = parse_timecode(parts[0])
        end_s = parse_timecode(parts[1])
        if end_s <= start_s:
            raise ValueError(f"Exclusion line {line_number} end must be after start")
        reason = parts[2] if len(parts) == 3 else ""
        exclusions.append(ExclusionWindow(start_s=start_s, end_s=end_s, reason=reason))
    return exclusions


def build_submission_metadata(
    job_id: str,
    created_at: str,
    safe_session_id: str,
    email: str,
    blocks_text: str,
    exclusions_text: str,
    video_a_name: str,
    video_b_name: str,
    video_a_size: int,
    video_b_size: int,
    include_gaze: bool,
    blocks_text_b: str | None = None,
) -> dict[str, Any]:
    """Build the durable metadata record for one submitted portal run.

    Args:
        job_id: Portal job identifier.
        created_at: Timestamp for job creation.
        safe_session_id: Sanitized session identifier.
        email: Notification email supplied by the user.
        blocks_text: Raw analysis block text for camera A from the form.
        exclusions_text: Raw extra-person window text from the form.
        video_a_name: Browser-provided filename for camera A.
        video_b_name: Browser-provided filename for camera B.
        video_a_size: Browser-reported byte size for camera A.
        video_b_size: Browser-reported byte size for camera B.
        include_gaze: Whether optional gaze analysis was requested.
        blocks_text_b: Raw analysis block text for camera B. Defaults to
            ``blocks_text`` for backward compatibility.

    Returns:
        JSON-serializable submission metadata.

    Raises:
        ValueError: If the supplied block or exclusion text is invalid.
    """
    has_video_a = bool(video_a_name) and video_a_size > 0
    has_video_b = bool(video_b_name) and video_b_size > 0
    if not has_video_a and not has_video_b:
        raise ValueError("At least one GoPro video is required.")
    validate_upload_size_limits(video_a_size, video_b_size)

    blocks_text_a = blocks_text
    blocks_text_b = blocks_text if blocks_text_b is None else blocks_text_b
    exclusions = _parse_exclusions_text(exclusions_text)

    input_blocks_by_camera: dict[str, list[SessionBlock]] = {}
    analysis_blocks_by_camera: dict[str, list[SessionBlock]] = {}
    if has_video_a:
        camera_a_blocks = _parse_blocks_text(blocks_text_a)
        camera_a_analysis_blocks = split_blocks_for_exclusions(
            camera_a_blocks, exclusions
        )
        if not camera_a_analysis_blocks:
            raise ValueError("Exclusions removed all Video A analysis time")
        input_blocks_by_camera["camera_a"] = camera_a_blocks
        analysis_blocks_by_camera["camera_a"] = camera_a_analysis_blocks
    if has_video_b:
        camera_b_blocks = _parse_blocks_text(blocks_text_b)
        camera_b_analysis_blocks = split_blocks_for_exclusions(
            camera_b_blocks, exclusions
        )
        if not camera_b_analysis_blocks:
            raise ValueError("Exclusions removed all Video B analysis time")
        input_blocks_by_camera["camera_b"] = camera_b_blocks
        analysis_blocks_by_camera["camera_b"] = camera_b_analysis_blocks

    legacy_blocks = next(iter(analysis_blocks_by_camera.values()))
    videos: dict[str, dict[str, Any]] = {}
    if has_video_a:
        videos["camera_a"] = {
            "original_filename": video_a_name,
            "size_bytes": video_a_size,
            "chunk_count": math.ceil(video_a_size / CHUNK_SIZE_BYTES),
        }
    if has_video_b:
        videos["camera_b"] = {
            "original_filename": video_b_name,
            "size_bytes": video_b_size,
            "chunk_count": math.ceil(video_b_size / CHUNK_SIZE_BYTES),
        }

    compression_settings = CompressionSettings()
    video_a_chunks = videos.get("camera_a", {}).get("chunk_count", 0)
    video_b_chunks = videos.get("camera_b", {}).get("chunk_count", 0)
    return {
        "schema_version": "1.0.0",
        "portal_name": PORTAL_DISPLAY_NAME,
        "job_id": job_id,
        "created_at": created_at,
        "session_id": safe_session_id,
        "email": email,
        "raw_inputs": {
            "analysis_blocks_text": blocks_text,
            "analysis_blocks_text_a": blocks_text_a,
            "analysis_blocks_text_b": blocks_text_b,
            "extra_person_windows_text": exclusions_text,
        },
        "input_analysis_blocks": [
            asdict(block)
            for block in input_blocks_by_camera[next(iter(input_blocks_by_camera))]
        ],
        "input_analysis_blocks_by_camera": {
            camera_id: [asdict(block) for block in blocks]
            for camera_id, blocks in input_blocks_by_camera.items()
        },
        "extra_person_windows": [asdict(exclusion) for exclusion in exclusions],
        "analysis_blocks": [asdict(block) for block in legacy_blocks],
        "analysis_blocks_by_camera": {
            camera_id: [asdict(block) for block in blocks]
            for camera_id, blocks in analysis_blocks_by_camera.items()
        },
        "blocks": [asdict(block) for block in legacy_blocks],
        "videos": videos,
        "video_a_name": video_a_name,
        "video_b_name": video_b_name,
        "video_a_size": video_a_size,
        "video_b_size": video_b_size,
        "video_a_chunks": video_a_chunks,
        "video_b_chunks": video_b_chunks,
        "upload": {
            "protocol": "browser_chunked",
            "chunk_size_bytes": CHUNK_SIZE_BYTES,
        },
        "processing": {
            "include_gaze": include_gaze,
            "analysis_window_only": True,
            "inference_backend": "ultralytics",
            "pose_model": PORTAL_DEFAULT_MODEL_PATH,
            "tracker_backend": "roboflow",
            "tracker_name": "bytetrack",
            "frame_rate": PORTAL_DEFAULT_FRAME_RATE,
            "max_persons": PORTAL_DEFAULT_MAX_PERSONS,
            "compression_size_threshold_mb": compression_settings.size_threshold_mb,
            "compression_target_fps": compression_settings.target_fps,
            "compression_max_width": compression_settings.max_width,
            "compression_quality": compression_settings.quality,
            "compression_encoders": list(compression_settings.encoder_order),
        },
        "include_gaze": include_gaze,
    }


def _write_submission_metadata(job_dir: Path, metadata: dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("upload_meta.json", "submission_metadata.json"):
        (job_dir / filename).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )


def _read_upload_meta(job_dir: Path) -> dict:
    meta_path = job_dir / "upload_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Upload metadata not found for job {job_dir.name}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Upload metadata must be a JSON object: {meta_path}")
    return payload


def _blocks_by_camera_from_meta(meta: dict) -> dict[str, list[SessionBlock]]:
    blocks_by_camera = meta.get("analysis_blocks_by_camera")
    if blocks_by_camera:
        return {
            camera_id: [SessionBlock(**item) for item in blocks]
            for camera_id, blocks in blocks_by_camera.items()
        }
    legacy_blocks = [
        SessionBlock(**item) for item in meta.get("analysis_blocks", meta["blocks"])
    ]
    return {
        camera_id: legacy_blocks
        for camera_id in meta.get("videos", {"camera_a": {}, "camera_b": {}})
    }


def _config_from_chunked_upload(
    safe_job_id: str,
    upload_dir: Path,
    job_dir: Path,
    meta: dict,
) -> PortalJobConfig:
    videos = meta["videos"]
    video_a_path = None
    video_b_path = None
    if "camera_a" in videos:
        camera_a = videos["camera_a"]
        video_a_path = validate_direct_chunked_upload(
            upload_dir=upload_dir,
            camera_id="camera_a",
            original_filename=camera_a["original_filename"],
            total_chunks=int(camera_a["chunk_count"]),
            expected_size_bytes=int(camera_a["size_bytes"]),
            probe=True,
            ffprobe_bin=os.environ.get("PORTAL_FFPROBE_BIN", "ffprobe"),
        )
    if "camera_b" in videos:
        camera_b = videos["camera_b"]
        video_b_path = validate_direct_chunked_upload(
            upload_dir=upload_dir,
            camera_id="camera_b",
            original_filename=camera_b["original_filename"],
            total_chunks=int(camera_b["chunk_count"]),
            expected_size_bytes=int(camera_b["size_bytes"]),
            probe=True,
            ffprobe_bin=os.environ.get("PORTAL_FFPROBE_BIN", "ffprobe"),
        )
    blocks_by_camera = _blocks_by_camera_from_meta(meta)
    legacy_blocks = next(iter(blocks_by_camera.values()))
    return PortalJobConfig(
        job_id=safe_job_id,
        session_id=meta["session_id"],
        email=meta["email"],
        video_a=video_a_path,
        video_b=video_b_path,
        job_dir=job_dir,
        output_root=OUTPUT_ROOT,
        session_config_path=job_dir / "session_config.json",
        blocks=legacy_blocks,
        blocks_by_camera=blocks_by_camera,
        include_gaze=bool(meta.get("include_gaze", False)),
    )


def _run_with_job_lock(job_dir: Path, action: Callable[[], None]) -> None:
    """Run one processing action after acquiring the in-process GPU job lock."""
    acquired = JOB_RUN_LOCK.acquire(blocking=False)
    if not acquired:
        write_status(
            job_dir,
            {
                "state": "queued",
                "message": "Waiting for the currently running GPU job to finish.",
                "current_step": "queued",
            },
        )
        JOB_RUN_LOCK.acquire()
    try:
        action()
    finally:
        JOB_RUN_LOCK.release()


def run_chunked_portal_job(safe_job_id: str, upload_dir: Path, job_dir: Path) -> None:
    """Assemble chunked uploads and run the portal pipeline."""
    try:

        def action() -> None:
            meta = _read_upload_meta(job_dir)
            write_status(
                job_dir,
                {
                    "state": "running",
                    "message": "Finalizing uploaded video chunks.",
                    "current_step": "finalize uploads",
                },
            )
            config = _config_from_chunked_upload(
                safe_job_id=safe_job_id,
                upload_dir=upload_dir,
                job_dir=job_dir,
                meta=meta,
            )
            run_portal_job(config, REPO_ROOT)

        _run_with_job_lock(job_dir, action)
    except Exception as error:
        write_status(
            job_dir,
            {
                "state": "failed",
                "message": str(error),
                "current_step": None,
            },
        )
        raise


def _job_rows() -> str:
    rows = []
    for status_path in sorted(JOB_ROOT.glob("*/status.json"), reverse=True):
        job_id = status_path.parent.name
        status = read_status(status_path.parent)
        safe_job_id = html.escape(job_id, quote=True)
        state = html.escape(str(status.get("state", "unknown")), quote=True)
        message = html.escape(str(status.get("message", "")), quote=True)
        rows.append(
            f'<tr><td><a href="/jobs/{safe_job_id}">{safe_job_id}</a></td>'
            f"<td>{state}</td><td>{message}</td></tr>"
        )
    if not rows:
        return '<tr><td colspan="3" class="muted">No jobs yet.</td></tr>'
    return "\n".join(rows)


def _client_host(request: Request) -> str:
    """Return a stable rate-limit key for one client."""
    if request.client is None:
        return "unknown"
    return str(request.client.host)


def _record_failed_login(client_host: str) -> None:
    """Record a failed login attempt within the active rate-limit window."""
    now = time.time()
    failures = [
        stamp
        for stamp in LOGIN_FAILURES.get(client_host, [])
        if now - stamp < LOGIN_RATE_LIMIT_WINDOW_SECONDS
    ]
    failures.append(now)
    LOGIN_FAILURES[client_host] = failures


def _login_is_rate_limited(client_host: str) -> bool:
    """Return whether a client has too many failed logins."""
    now = time.time()
    failures = [
        stamp
        for stamp in LOGIN_FAILURES.get(client_host, [])
        if now - stamp < LOGIN_RATE_LIMIT_WINDOW_SECONDS
    ]
    LOGIN_FAILURES[client_host] = failures
    return len(failures) >= LOGIN_RATE_LIMIT_ATTEMPTS


@app.get("/login", response_class=HTMLResponse)
def login_page() -> HTMLResponse:
    """Render the login page."""
    return _html_page(
        "Login",
        """<form method="post" action="/login">
  <label for="password">Password</label>
  <input id="password" name="password" type="password" autocomplete="current-password">
  <button type="submit">Log in</button>
</form>""",
    )


@app.post("/login")
def login(request: Request, password: Annotated[str, Form()]) -> RedirectResponse:
    """Validate the shared password and set the session cookie."""
    client_host = _client_host(request)
    if _login_is_rate_limited(client_host):
        raise HTTPException(status_code=429, detail="Too many failed login attempts")
    if not secrets.compare_digest(password, portal_password()):
        _record_failed_login(client_host)
        return RedirectResponse("/login", status_code=303)
    LOGIN_FAILURES.pop(client_host, None)
    token = create_session_token()
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        "portal_session",
        token,
        max_age=session_ttl_seconds(),
        httponly=True,
        samesite="strict",
        secure=cookie_secure(),
    )
    return response


@app.post("/logout")
def logout(session_token: Annotated[str, Depends(require_csrf)]) -> RedirectResponse:
    """Clear the current portal session."""
    SESSION_TOKENS.pop(session_token, None)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("portal_session")
    return response


@app.get("/", response_class=HTMLResponse)
def index(session_token: Annotated[str, Depends(require_auth)]) -> HTMLResponse:
    """Render the upload form and recent jobs."""
    csrf_token = csrf_token_for_session(session_token)
    return _html_page(
        "Submit Job",
        f"""<div class="top-actions"><button id="logout-button" type="button">Log out</button></div>

<form id="upload-form">
  <label for="session_id">Session ID {_info_icon("Use the participant/session name that should appear on outputs.")}</label>
  <input id="session_id" name="session_id" type="text" required placeholder="P001c">

  <label for="email">Notification Email {_info_icon("The portal emails this address when processing finishes or fails.")}</label>
  <input id="email" name="email" type="email" required placeholder="researcher@example.edu">

  <div class="warning"><strong>Upload at least one GoPro video.</strong> Video A and Video B are both supported, but one camera can be omitted when only one recording is available.</div>

  <label for="video_a">GoPro Video A {_info_icon("Optional. Upload the first GoPro angle if available.")}</label>
  <div class="file-drop-zone" data-file-input="video_a" tabindex="0">
    <strong>Drop GoPro Video A here</strong>
    <span class="muted">or click to choose a file</span>
    <span id="video_a_selected" class="muted file-name">No file selected</span>
  </div>
  <input id="video_a" class="file-input-native" name="video_a" type="file" accept=".mov,.mp4,.avi,.mkv,.m4v">

  <label for="video_b">GoPro Video B {_info_icon("Optional. Upload the second GoPro angle if available.")}</label>
  <div class="file-drop-zone" data-file-input="video_b" tabindex="0">
    <strong>Drop GoPro Video B here</strong>
    <span class="muted">or click to choose a file</span>
    <span id="video_b_selected" class="muted file-name">No file selected</span>
  </div>
  <input id="video_b" class="file-input-native" name="video_b" type="file" accept=".mov,.mp4,.avi,.mkv,.m4v">

  <label for="blocks_text_a">Video A Analysis Blocks {_info_icon("Required when Video A is uploaded. These windows are the only Video A times processed.")}</label>
  <textarea id="blocks_text_a" name="blocks_text_a" placeholder="free_play,13:26,23:40,green&#10;storybook,29:22,37:06,blue"></textarea>
  <div class="muted">One block per line: name,start,end[,color]. Times may be seconds, MM:SS, or HH:MM:SS.</div>

  <label for="blocks_text_b">Video B Analysis Blocks {_info_icon("Required when Video B is uploaded. Use the same format as Video A.")}</label>
  <textarea id="blocks_text_b" name="blocks_text_b" placeholder="free_play,13:26,23:40,green&#10;storybook,29:22,37:06,blue"></textarea>
  <div class="muted">Leave blank only if Video B is not uploaded.</div>

  <label for="exclusions_text">Extra-Person Windows {_info_icon("These windows are removed from both cameras before inference.")}</label>
  <textarea id="exclusions_text" name="exclusions_text" placeholder="8:14,8:42,experimenter entered"></textarea>
  <div class="muted">One exclusion per line: start,end,reason. These windows are removed from analysis blocks.</div>

  <label><input name="include_gaze" type="checkbox" value="yes"> Run gaze analysis {_info_icon("Optional. Adds gaze heatmaps and gaze metrics after pose tracking.")}</label>
  <button type="submit">Submit Processing Job</button>
  <p id="upload-status" class="muted"></p>
  <progress id="upload-progress" value="0" max="100" style="width: 100%; display: none;"></progress>
</form>

<section class="panel">
  <h2>Result Figures And Tables</h2>
  <div class="output-grid">
    <div class="output-card">
      <h3>Proximity</h3>
      <p class="muted">Parent-child torso distance inside each valid analysis block.</p>
      <ul>
        <li>Figure: proximity over time and block comparison</li>
        <li>Table: per-frame torso distance and per-block summary</li>
      </ul>
    </div>
    <div class="output-card">
      <h3>Movement synchrony</h3>
      <p class="muted">Windowed movement cross-correlation between parent and child tracks.</p>
      <ul>
        <li>Figure: synchrony strength and lead-lag summary</li>
        <li>Table: peak correlation and lag per window</li>
      </ul>
    </div>
    <div class="output-card">
      <h3>Gaze estimation</h3>
      <p class="muted">Optional gaze categories and heatmap-based convergence metrics.</p>
      <ul>
        <li>Figure: gaze dashboard and snapshots when enabled</li>
        <li>Table: gaze category proportions and convergence values</li>
      </ul>
    </div>
  </div>
</section>

<section class="panel">
  <h2>Jobs</h2>
  <table>
    <thead><tr><th>Job</th><th>Status</th><th>Message</th></tr></thead>
    <tbody>{_job_rows()}</tbody>
  </table>
</section>

<script>
const CHUNK_SIZE = {CHUNK_SIZE_BYTES};
const CSRF_TOKEN = "{html.escape(csrf_token, quote=True)}";
const form = document.getElementById("upload-form");
const logoutButton = document.getElementById("logout-button");
const statusEl = document.getElementById("upload-status");
const progressEl = document.getElementById("upload-progress");
const dropZones = document.querySelectorAll(".file-drop-zone");
let uploadInProgress = false;

function setStatus(message) {{
  statusEl.textContent = message;
}}

function formatBytes(bytes) {{
  const gib = bytes / (1024 * 1024 * 1024);
  if (gib >= 1) {{
    return `${{gib.toFixed(2)}} GB`;
  }}
  return `${{(bytes / (1024 * 1024)).toFixed(1)}} MB`;
}}

function updateSelectedFile(input) {{
  const label = document.getElementById(`${{input.id}}_selected`);
  if (!label) {{
    return;
  }}
  label.textContent = input.files.length ? input.files[0].name : "No file selected";
}}

logoutButton.addEventListener("click", async () => {{
  await fetch("/logout", {{
    method: "POST",
    headers: {{"X-CSRF-Token": CSRF_TOKEN}},
    credentials: "same-origin"
  }});
  window.location.href = "/login";
}});

dropZones.forEach((zone) => {{
  const input = document.getElementById(zone.dataset.fileInput);
  input.addEventListener("change", () => updateSelectedFile(input));

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (event) => {{
    if (event.key === "Enter" || event.key === " ") {{
      event.preventDefault();
      input.click();
    }}
  }});
  zone.addEventListener("dragover", (event) => {{
    event.preventDefault();
    zone.classList.add("dragover");
  }});
  zone.addEventListener("dragleave", () => {{
    zone.classList.remove("dragover");
  }});
  zone.addEventListener("drop", (event) => {{
    event.preventDefault();
    zone.classList.remove("dragover");
    if (!event.dataTransfer.files.length) {{
      return;
    }}
    const transfer = new DataTransfer();
    transfer.items.add(event.dataTransfer.files[0]);
    input.files = transfer.files;
    updateSelectedFile(input);
  }});
}});

async function postFormData(url, formData) {{
  const response = await fetch(url, {{
    method: "POST",
    body: formData,
    headers: {{"X-CSRF-Token": CSRF_TOKEN}},
    credentials: "same-origin"
  }});
  if (!response.ok) {{
    const text = await response.text();
    throw new Error(text || `Request failed: ${{response.status}}`);
  }}
  return await response.json();
}}

function postChunkFormData(url, formData, onProgress) {{
  return new Promise((resolve, reject) => {{
    const request = new XMLHttpRequest();
    request.open("POST", url);
    request.withCredentials = true;
    request.setRequestHeader("X-CSRF-Token", CSRF_TOKEN);
    request.upload.onprogress = (event) => {{
      if (event.lengthComputable) {{
        onProgress(event.loaded, event.total);
      }}
    }};
    request.onload = () => {{
      if (request.status >= 200 && request.status < 300) {{
        resolve(JSON.parse(request.responseText));
        return;
      }}
      reject(new Error(request.responseText || `Request failed: ${{request.status}}`));
    }};
    request.onerror = () => reject(new Error("Network error during chunk upload."));
    request.send(formData);
  }});
}}

function updateUploadProgress(cameraId, cameraLabel, chunkNumber, totalChunks, visibleCameraBytes, uploadState) {{
  uploadState[cameraId] = visibleCameraBytes;
  const visibleBytes = uploadState.camera_a + uploadState.camera_b;
  progressEl.value = Math.round((visibleBytes / uploadState.totalBytes) * 100);
  setStatus(
    `${{cameraLabel}}: uploading chunk ${{chunkNumber}} of ${{totalChunks}} ` +
    `(${{formatBytes(visibleBytes)}} of ${{formatBytes(uploadState.totalBytes)}} total). Keep this tab open.`
  );
}}

async function uploadCamera(jobId, cameraId, file, uploadState) {{
  const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
  const cameraLabel = cameraId === "camera_a" ? "Video A" : "Video B";
  let uploadedCameraBytes = 0;
  for (let index = 0; index < totalChunks; index += 1) {{
    const start = index * CHUNK_SIZE;
    const end = Math.min(file.size, start + CHUNK_SIZE);
    const chunk = file.slice(start, end);
    const formData = new FormData();
    formData.append("camera_id", cameraId);
    formData.append("chunk_index", String(index));
    formData.append("total_chunks", String(totalChunks));
    formData.append("chunk", chunk, file.name);
    setStatus(`${{cameraLabel}}: uploading chunk ${{index + 1}} of ${{totalChunks}}. Keep this tab open.`);
    await postChunkFormData(`/uploads/${{jobId}}/chunk`, formData, (loaded) => {{
      updateUploadProgress(
        cameraId,
        cameraLabel,
        index + 1,
        totalChunks,
        uploadedCameraBytes + loaded,
        uploadState
      );
    }});
    uploadedCameraBytes += chunk.size;
    updateUploadProgress(
      cameraId,
      cameraLabel,
      index + 1,
      totalChunks,
      uploadedCameraBytes,
      uploadState
    );
  }}
}}

window.addEventListener("beforeunload", (event) => {{
  if (!uploadInProgress) {{
    return;
  }}
  event.preventDefault();
  event.returnValue = "";
}});

form.addEventListener("submit", async (event) => {{
  event.preventDefault();
  const submitButton = form.querySelector("button");
  submitButton.disabled = true;
  submitButton.textContent = "Uploading...";
  uploadInProgress = true;
  document.title = "Uploading videos";
  progressEl.style.display = "block";
  progressEl.value = 0;

  try {{
    const videoA = document.getElementById("video_a").files[0];
    const videoB = document.getElementById("video_b").files[0];
    const blocksA = document.getElementById("blocks_text_a").value.trim();
    const blocksB = document.getElementById("blocks_text_b").value.trim();
    if (!videoA && !videoB) {{
      throw new Error("Upload at least one GoPro video.");
    }}
    if (videoA && !blocksA) {{
      throw new Error("Video A analysis blocks are required when Video A is uploaded.");
    }}
    if (videoB && !blocksB) {{
      throw new Error("Video B analysis blocks are required when Video B is uploaded.");
    }}

    const initData = new FormData(form);
    initData.append("video_a_name", videoA ? videoA.name : "");
    initData.append("video_b_name", videoB ? videoB.name : "");
    initData.append("video_a_size", videoA ? String(videoA.size) : "0");
    initData.append("video_b_size", videoB ? String(videoB.size) : "0");
    initData.delete("video_a");
    initData.delete("video_b");

    setStatus("Creating upload job");
    const init = await postFormData("/uploads/init", initData);
    const totalBytes = (videoA ? videoA.size : 0) + (videoB ? videoB.size : 0);
    const uploadState = {{ camera_a: 0, camera_b: 0, totalBytes }};
    const uploadTasks = [];
    if (videoA) {{
      uploadTasks.push(uploadCamera(init.job_id, "camera_a", videoA, uploadState));
    }}
    if (videoB) {{
      uploadTasks.push(uploadCamera(init.job_id, "camera_b", videoB, uploadState));
    }}
    await Promise.all(uploadTasks);

    setStatus("Upload complete. Asking server to finalize chunks and start processing.");
    document.title = "Processing queued";
    const finish = await postFormData(`/uploads/${{init.job_id}}/finish`, new FormData());
    uploadInProgress = false;
    window.location.href = finish.status_url;
  }} catch (error) {{
    setStatus(`Upload failed: ${{error.message}}`);
    submitButton.disabled = false;
    submitButton.textContent = "Submit Processing Job";
    uploadInProgress = false;
    document.title = "Submit Job";
  }}
}});
</script>""",
    )


@app.post("/uploads/init")
async def init_chunked_upload(
    _: Annotated[str, Depends(require_csrf)],
    session_id: Annotated[str, Form()],
    email: Annotated[str, Form()],
    video_a_name: Annotated[str, Form()] = "",
    video_b_name: Annotated[str, Form()] = "",
    video_a_size: Annotated[int, Form()] = 0,
    video_b_size: Annotated[int, Form()] = 0,
    blocks_text: Annotated[str | None, Form()] = None,
    blocks_text_a: Annotated[str | None, Form()] = None,
    blocks_text_b: Annotated[str | None, Form()] = None,
    exclusions_text: Annotated[str, Form()] = "",
    include_gaze: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Create upload metadata for a chunked browser upload."""
    try:
        safe_session_id = _slug(session_id)
        active_video_names = [
            name
            for name, size in (
                (video_a_name, video_a_size),
                (video_b_name, video_b_size),
            )
            if name and size > 0
        ]
        if not active_video_names:
            raise ValueError("At least one GoPro video is required.")
        for filename in active_video_names:
            suffix = Path(filename).suffix.lower()
            if suffix not in {".avi", ".m4v", ".mkv", ".mov", ".mp4"}:
                raise ValueError(f"Unsupported video extension: {filename}")
        if (video_a_name and video_a_size <= 0) or (video_b_name and video_b_size <= 0):
            raise ValueError("Uploaded videos must be non-empty")
        resolved_blocks_text_a = blocks_text_a or blocks_text or ""
        resolved_blocks_text_b = blocks_text_b or blocks_text or resolved_blocks_text_a

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"{stamp}_{safe_session_id}_{uuid.uuid4().hex[:8]}"
        upload_dir = UPLOAD_ROOT / job_id
        job_dir = JOB_ROOT / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        job_dir.mkdir(parents=True, exist_ok=True)

        metadata = build_submission_metadata(
            job_id=job_id,
            created_at=datetime.now().isoformat(timespec="seconds"),
            safe_session_id=safe_session_id,
            email=email,
            blocks_text=resolved_blocks_text_a,
            blocks_text_b=resolved_blocks_text_b,
            exclusions_text=exclusions_text,
            video_a_name=video_a_name,
            video_b_name=video_b_name,
            video_a_size=video_a_size,
            video_b_size=video_b_size,
            include_gaze=include_gaze == "yes",
        )
        _write_submission_metadata(job_dir, metadata)
        write_status(
            job_dir,
            {
                "state": "uploading",
                "message": "Waiting for video chunks.",
                "current_step": "upload",
            },
        )
        return JSONResponse(
            {
                "job_id": job_id,
                "chunk_size": CHUNK_SIZE_BYTES,
                "status_url": f"/jobs/{job_id}",
            }
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/uploads/{job_id}/chunk")
async def upload_chunk(
    job_id: str,
    _: Annotated[str, Depends(require_csrf)],
    camera_id: Annotated[str, Form()],
    chunk_index: Annotated[int, Form()],
    total_chunks: Annotated[int, Form()],
    chunk: Annotated[UploadFile, File()],
) -> JSONResponse:
    """Store one chunk for one camera video."""
    try:
        safe_job_id = _slug(job_id)
        upload_dir = UPLOAD_ROOT / safe_job_id
        job_dir = JOB_ROOT / safe_job_id
        meta_path = job_dir / "upload_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Unknown upload job: {safe_job_id}")
        meta = _read_upload_meta(job_dir)
        if camera_id not in meta["videos"]:
            raise ValueError(f"{camera_id} was not included in this upload job")
        camera_meta = meta["videos"][camera_id]
        expected_chunks = int(camera_meta["chunk_count"])
        if total_chunks < 1:
            raise ValueError("total_chunks must be at least 1")
        if total_chunks != expected_chunks:
            raise ValueError(
                f"{camera_id} expected {expected_chunks} chunks, got {total_chunks}"
            )
        if chunk_index >= expected_chunks:
            raise ValueError(f"chunk_index {chunk_index} exceeds expected chunks")
        if chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        expected_size = int(camera_meta["size_bytes"])
        expected_chunk_size = min(
            CHUNK_SIZE_BYTES,
            max(0, expected_size - (chunk_index * CHUNK_SIZE_BYTES)),
        )
        if expected_chunk_size <= 0:
            raise ValueError(f"chunk_index {chunk_index} exceeds expected file size")

        output_path = final_upload_path_for(
            upload_dir,
            camera_id,
            camera_meta["original_filename"],
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+b" if output_path.exists() else "w+b"
        bytes_written = 0
        with output_path.open(mode) as output_file:
            output_file.seek(chunk_index * CHUNK_SIZE_BYTES)
            while True:
                data = await chunk.read(1024 * 1024)
                if not data:
                    break
                if bytes_written + len(data) > max_chunk_upload_bytes():
                    raise ValueError(
                        f"Upload chunk exceeds {max_chunk_upload_bytes()} bytes"
                    )
                output_file.write(data)
                bytes_written += len(data)
        if bytes_written != expected_chunk_size:
            raise ValueError(
                f"{camera_id} chunk {chunk_index} size mismatch: expected "
                f"{expected_chunk_size} bytes, received {bytes_written} bytes"
            )
        mark_chunk_received(upload_dir, camera_id, chunk_index, bytes_written)
        return JSONResponse(
            {
                "job_id": safe_job_id,
                "camera_id": camera_id,
                "chunk_index": chunk_index,
                "received": bytes_written,
            }
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/uploads/{job_id}/finish")
async def finish_chunked_upload(
    job_id: str,
    background_tasks: BackgroundTasks,
    _: Annotated[str, Depends(require_csrf)],
) -> JSONResponse:
    """Assemble chunks and start processing."""
    try:
        safe_job_id = _slug(job_id)
        upload_dir = UPLOAD_ROOT / safe_job_id
        job_dir = JOB_ROOT / safe_job_id
        _read_upload_meta(job_dir)
        write_status(
            job_dir,
            {
                "state": "queued",
                "message": "Upload complete. Finalizing videos will start now.",
                "current_step": "queued",
            },
        )
        background_tasks.add_task(
            run_chunked_portal_job, safe_job_id, upload_dir, job_dir
        )
        return JSONResponse(
            {"job_id": safe_job_id, "status_url": f"/jobs/{safe_job_id}"}
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(
    job_id: str,
    _: Annotated[None, Depends(require_auth)],
) -> HTMLResponse:
    """Render one job status page."""
    safe_job_id = _slug(job_id)
    display_job_id = html.escape(safe_job_id, quote=True)
    job_dir = JOB_ROOT / safe_job_id
    status = read_status(job_dir)
    state = html.escape(str(status.get("state", "unknown")), quote=True)
    message = html.escape(str(status.get("message", "")), quote=True)
    result_zip = status.get("result_zip")
    download = ""
    if result_zip and Path(result_zip).exists():
        download = (
            f'<p><a href="/jobs/{safe_job_id}/download">Download results zip</a></p>'
        )
    refresh_script = ""
    if status.get("state") in {"uploading", "queued", "running"}:
        refresh_script = (
            '<p class="muted">This page refreshes automatically while the job runs.</p>'
            "<script>setTimeout(() => window.location.reload(), 10000);</script>"
        )
    log_link = ""
    if (job_dir / "job.log").exists():
        log_link = f'<p><a href="/jobs/{safe_job_id}/log">View job log</a></p>'
    return _html_page(
        f"Job {display_job_id}",
        f"""<section class="panel">
  <h2>{display_job_id}</h2>
  <p><strong>Status:</strong> {state}</p>
  <p><strong>Message:</strong> {message}</p>
  {download}
  {log_link}
  {refresh_script}
  <p><a href="/">Back to jobs</a></p>
</section>""",
    )


@app.get("/jobs/{job_id}/download")
def download_job(
    job_id: str,
    _: Annotated[None, Depends(require_auth)],
) -> FileResponse:
    """Download a completed result archive."""
    safe_job_id = _slug(job_id)
    status = read_status(JOB_ROOT / safe_job_id)
    result_zip = status.get("result_zip")
    if not result_zip or not Path(result_zip).exists():
        raise HTTPException(status_code=404, detail="Result archive not available")
    return FileResponse(result_zip, filename=Path(result_zip).name)


@app.get("/jobs/{job_id}/log", response_class=HTMLResponse)
def job_log(
    job_id: str,
    _: Annotated[None, Depends(require_auth)],
) -> HTMLResponse:
    """Render the plain-text job log."""
    safe_job_id = _slug(job_id)
    log_path = JOB_ROOT / safe_job_id / "job.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not available")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _html_page("Job Log", f"<pre>{escaped}</pre>")


def main() -> None:
    """Run the portal with uvicorn."""
    import uvicorn

    host = os.environ.get("PORTAL_HOST", "0.0.0.0")
    port = int(os.environ.get("PORTAL_PORT", "8080"))
    uvicorn.run("portal.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
