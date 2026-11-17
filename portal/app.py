"""FastAPI web app for the local processing portal."""

from __future__ import annotations

import hmac
import json
import math
import mimetypes
import os
import re
import secrets
import shutil
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
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from starlette.responses import Response
from starlette.types import Scope

from .jobs import (
    PORTAL_DEFAULT_FRAME_RATE,
    PORTAL_DEFAULT_MAX_PERSONS,
    PORTAL_DEFAULT_MODEL_PATH,
    PORTAL_TRACKER_BACKEND,
    PORTAL_TRACKER_NAME,
    CompressionSettings,
    ExclusionWindow,
    PortalJobConfig,
    SessionBlock,
    final_upload_path_for,
    mark_chunk_received,
    parse_timecode,
    read_status,
    resume_portal_job_phase2,
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
# Upload chunk size. Sized so that one HTTP chunk completes well under the
# Cloudflare Quick Tunnel 100 s edge timeout even on slow campus uplinks:
# 8 MB takes ~13 s at 5 Mbps, ~64 s at 1 Mbps. Larger chunks (50 MB+) hit
# Cloudflare 524 on anything slower than ~5 Mbps.
CHUNK_SIZE_BYTES = 8 * 1024 * 1024
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


def purge_all_jobs() -> int:
    """Delete every job directory and matching upload-chunk directory.

    Returns:
        Number of job directories removed.

    Note:
        Restarting the portal is treated as a clean slate: no job state,
        completed downloads, or failed-job logs survive. Only direct
        subdirectories under JOB_ROOT/UPLOAD_ROOT are removed; loose log
        files (tunnel, server) are left alone.
    """
    if not JOB_ROOT.exists():
        return 0
    removed = 0
    for job_dir in JOB_ROOT.iterdir():
        if not job_dir.is_dir():
            continue
        upload_dir = UPLOAD_ROOT / job_dir.name
        shutil.rmtree(job_dir, ignore_errors=True)
        shutil.rmtree(upload_dir, ignore_errors=True)
        removed += 1
    return removed


@asynccontextmanager
async def portal_lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate portal configuration and wipe all jobs on startup."""
    validate_startup_configuration()
    removed = purge_all_jobs()
    if removed:
        print(f"Purged {removed} job(s) from a previous run.")
    yield


PORTAL_PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = PORTAL_PACKAGE_DIR / "templates"
STATIC_DIR = PORTAL_PACKAGE_DIR / "static"
STATIC_SUFFIXES = frozenset({".css", ".js"})


def script_json(value: Any) -> Markup:
    r"""Serialize ``value`` as JSON that is safe inside a ``<script>`` element.

    ``<``, ``>`` and ``&`` become ``\uXXXX`` escapes, so no string in the data
    can close the script block or open a comment. Key order is kept (unlike
    Jinja's ``tojson``, which sorts keys and also escapes apostrophes).

    Args:
        value: JSON-serializable data.

    Returns:
        Markup holding the escaped JSON text.
    """
    text = (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return Markup(text)


# Autoescaping is on for every template; values are passed raw and escaped at
# render time.
TEMPLATES = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(enabled_extensions=("html",), default=True),
)
TEMPLATES.filters["script_json"] = script_json

# Windows can map .js to text/plain through the registry; pin the types the
# static mount serves so browsers always run/apply them.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
# Same for the session files route (report figures, overlay clips, tables).
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("text/csv", ".csv")


class PortalStaticFiles(StaticFiles):
    """Static file mount that serves only stylesheets and scripts.

    The mount is public (the login page needs its stylesheet), so anything
    other than ``.css``/``.js`` returns 404, even if it is placed in the folder.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Serve ``path`` only when its suffix is an allowed static type.

        Args:
            path: Requested path relative to the static directory.
            scope: ASGI request scope.

        Returns:
            The static file response.

        Raises:
            HTTPException: 404 for any other file type or a directory.
        """
        if Path(path).suffix.lower() not in STATIC_SUFFIXES:
            raise HTTPException(status_code=404, detail="Not Found")
        return await super().get_response(path, scope)


app = FastAPI(title=PORTAL_DISPLAY_NAME, lifespan=portal_lifespan)
app.mount("/static", PortalStaticFiles(directory=STATIC_DIR), name="static")


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


def _render_page(
    template_name: str,
    title: str,
    session_token: str | None = None,
    **context: Any,
) -> HTMLResponse:
    """Render one portal page from ``portal/templates``.

    Args:
        template_name: Template file name, e.g. ``"index.html"``.
        title: Text for the page ``<title>`` (autoescaped).
        session_token: Logged-in session token. When given, the header shows
            the navigation and a Log out button carrying this session's CSRF
            token.
        **context: Extra template variables for the page.

    Returns:
        The rendered page as an HTML response.
    """
    template = TEMPLATES.get_template(template_name)
    header_csrf = csrf_token_for_session(session_token) if session_token else None
    return HTMLResponse(
        template.render(
            title=title,
            portal_name=PORTAL_DISPLAY_NAME,
            header_csrf=header_csrf,
            **context,
        )
    )


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
    blocks_text: str,
    exclusions_text: str,
    video_a_name: str,
    video_b_name: str,
    video_a_size: int,
    video_b_size: int,
    include_gaze: bool,
    blocks_text_b: str | None = None,
    exclusions_text_b: str | None = None,
) -> dict[str, Any]:
    """Build the durable metadata record for one submitted portal run.

    Args:
        job_id: Portal job identifier.
        created_at: Timestamp for job creation.
        safe_session_id: Sanitized session identifier.
        blocks_text: Raw analysis block text for camera A from the form.
        exclusions_text: Raw extra-person window text for camera A, in Video
            A's own time.
        video_a_name: Browser-provided filename for camera A.
        video_b_name: Browser-provided filename for camera B.
        video_a_size: Browser-reported byte size for camera A.
        video_b_size: Browser-reported byte size for camera B.
        include_gaze: Whether optional gaze analysis was requested.
        blocks_text_b: Raw analysis block text for camera B. Defaults to
            ``blocks_text`` for backward compatibility.
        exclusions_text_b: Raw extra-person window text for camera B, in Video
            B's own time. Defaults to ``exclusions_text`` for backward
            compatibility.

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
    exclusions_text_a = exclusions_text
    exclusions_text_b = (
        exclusions_text if exclusions_text_b is None else exclusions_text_b
    )
    # The two GoPros have different clocks, so each camera's extra-person
    # windows are entered (and removed) in that camera's own time.
    exclusions_by_camera = {
        "camera_a": _parse_exclusions_text(exclusions_text_a),
        "camera_b": _parse_exclusions_text(exclusions_text_b),
    }

    input_blocks_by_camera: dict[str, list[SessionBlock]] = {}
    analysis_blocks_by_camera: dict[str, list[SessionBlock]] = {}
    if has_video_a:
        camera_a_blocks = _parse_blocks_text(blocks_text_a)
        camera_a_analysis_blocks = split_blocks_for_exclusions(
            camera_a_blocks, exclusions_by_camera["camera_a"]
        )
        if not camera_a_analysis_blocks:
            raise ValueError("Exclusions removed all Video A analysis time")
        input_blocks_by_camera["camera_a"] = camera_a_blocks
        analysis_blocks_by_camera["camera_a"] = camera_a_analysis_blocks
    if has_video_b:
        camera_b_blocks = _parse_blocks_text(blocks_text_b)
        camera_b_analysis_blocks = split_blocks_for_exclusions(
            camera_b_blocks, exclusions_by_camera["camera_b"]
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
        "raw_inputs": {
            "analysis_blocks_text": blocks_text,
            "analysis_blocks_text_a": blocks_text_a,
            "analysis_blocks_text_b": blocks_text_b,
            "extra_person_windows_text": exclusions_text,
            "extra_person_windows_text_a": exclusions_text_a,
            "extra_person_windows_text_b": exclusions_text_b,
        },
        "input_analysis_blocks": [
            asdict(block)
            for block in input_blocks_by_camera[next(iter(input_blocks_by_camera))]
        ],
        "input_analysis_blocks_by_camera": {
            camera_id: [asdict(block) for block in blocks]
            for camera_id, blocks in input_blocks_by_camera.items()
        },
        "extra_person_windows": [
            asdict(exclusion)
            for exclusion in exclusions_by_camera[next(iter(input_blocks_by_camera))]
        ],
        "extra_person_windows_by_camera": {
            camera_id: [
                asdict(exclusion) for exclusion in exclusions_by_camera[camera_id]
            ]
            for camera_id in input_blocks_by_camera
        },
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
            "tracker_backend": PORTAL_TRACKER_BACKEND,
            "tracker_name": PORTAL_TRACKER_NAME,
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
    """Assemble chunked uploads and run the portal pipeline.

    Runs as a FastAPI BackgroundTask, so any unhandled exception is logged by
    starlette as a noisy ASGI traceback. Catch every failure, record it on
    the job's status.json, and log a single concise line for the operator.
    """
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
        log_path = job_dir / "job.log"
        print(
            f"Job {safe_job_id} failed. See {log_path} for the full subprocess log.",
            flush=True,
        )


def resume_portal_job_phase2_background(safe_job_id: str, job_dir: Path) -> None:
    """Background task: continue a portal job after annotations arrive.

    Mirrors ``run_chunked_portal_job``'s shape — runs under the GPU job
    lock and converts any failure into a status entry plus a one-line log
    so the operator can see what happened without an ASGI traceback.
    """
    try:

        def action() -> None:
            meta = _read_upload_meta(job_dir)
            upload_dir = UPLOAD_ROOT / safe_job_id
            config = _config_from_chunked_upload(
                safe_job_id=safe_job_id,
                upload_dir=upload_dir,
                job_dir=job_dir,
                meta=meta,
            )
            resume_portal_job_phase2(config, REPO_ROOT)

        _run_with_job_lock(job_dir, action)
    except Exception as error:
        write_status(
            job_dir,
            {
                "state": "failed",
                "message": str(error),
                "current_step": None,
                "phase": 2,
            },
        )
        log_path = job_dir / "job.log"
        print(
            f"Job {safe_job_id} phase 2 failed. See {log_path}.",
            flush=True,
        )


# Job state -> (chip kind, plain-language label). Chip kinds map to the
# ``chip--<kind>`` classes in portal.css (running / waiting / done / failed).
JOB_STATE_CHIPS: dict[str, tuple[str, str]] = {
    "uploading": ("running", "Uploading"),
    "queued": ("running", "Waiting its turn"),
    "running": ("running", "Running"),
    "awaiting_annotation": ("waiting", "Waiting for you"),
    "complete": ("done", "Done"),
    "failed": ("failed", "Failed"),
}
ACTIVE_JOB_STATES = frozenset({"uploading", "queued", "running", "awaiting_annotation"})


def job_state_chip(state: str) -> tuple[str, str]:
    """Return the status chip (kind, label) for a raw job state.

    Args:
        state: Raw ``state`` value from a job's ``status.json``.

    Returns:
        ``(kind, label)``; an unrecognized state keeps its raw text as the
        label (escaped by the template) under the ``"unknown"`` kind.
    """
    return JOB_STATE_CHIPS.get(state, ("unknown", state or "Unknown"))


def _job_rows() -> list[dict[str, str]]:
    """Return one display row per job folder, active jobs first.

    Jobs that are running or waiting for the user come first; within each
    group, folders keep their newest-name-first order.

    Returns:
        Dicts with ``job_id``, ``state``, ``message``, ``chip_kind`` and
        ``chip_label`` (raw strings; the template escapes them).
    """
    rows = []
    for status_path in sorted(JOB_ROOT.glob("*/status.json"), reverse=True):
        status = read_status(status_path.parent)
        state = str(status.get("state", "unknown"))
        kind, label = job_state_chip(state)
        rows.append(
            {
                "job_id": status_path.parent.name,
                "state": state,
                "message": str(status.get("message", "")),
                "chip_kind": kind,
                "chip_label": label,
            }
        )
    # Stable sort keeps the newest-first order inside each group.
    rows.sort(key=lambda row: row["state"] not in ACTIVE_JOB_STATES)
    return rows


SESSION_METRICS_CSV = "session_metrics.csv"
SESSION_REPORT_HTML = "report.html"
LOW_COVERAGE_THRESHOLD = 0.5
# File types the session route may serve: report, figures, clips, tables.
SESSION_FILE_SUFFIXES = frozenset(
    {".png", ".svg", ".jpg", ".mp4", ".csv", ".json", ".html"}
)


def overall_session_verdict(table: Any) -> tuple[str, str]:
    """Summarize a session's blocks as one verdict chip.

    When the ``quality_verdict`` column from
    ``docs/metrics-validity_261116.md`` is present it decides: any
    ``insufficient`` block gives "Not enough data", any ``check`` block gives
    "Check", otherwise "Looks OK". Until that column exists, the fallback uses
    ``coverage_both`` (share of the block with both people tracked): every
    block at or above 0.5 gives "Looks OK", otherwise "Check N blocks"
    (missing coverage counts as low).

    Args:
        table: The session's ``session_metrics.csv`` as a pandas DataFrame.

    Returns:
        ``(kind, label)`` where kind is ``"ok"``, ``"check"`` or
        ``"insufficient"`` (the ``chip--<kind>`` class in portal.css).
    """
    import pandas as pd

    if len(table) == 0:
        return ("insufficient", "Not enough data")
    if "quality_verdict" in table.columns:
        verdicts = {
            str(value).strip().lower() for value in table["quality_verdict"].dropna()
        }
        if "insufficient" in verdicts:
            return ("insufficient", "Not enough data")
        if "check" in verdicts:
            return ("check", "Check")
        return ("ok", "Looks OK")
    if "coverage_both" not in table.columns:
        return ("insufficient", "Not enough data")
    coverage = pd.to_numeric(table["coverage_both"], errors="coerce")
    n_low = int((coverage.isna() | (coverage < LOW_COVERAGE_THRESHOLD)).sum())
    if n_low == 0:
        return ("ok", "Looks OK")
    return ("check", f"Check {n_low} block{'' if n_low == 1 else 's'}")


def _session_dir_is_addressable(name: str) -> bool:
    """Return whether a folder name survives ``_slug`` unchanged.

    Only such folders can be opened through ``/sessions/{session_id}``.

    Args:
        name: Folder name under ``OUTPUT_ROOT``.

    Returns:
        True when the session route can address the folder.
    """
    try:
        return _slug(name) == name
    except ValueError:
        return False


def session_rows() -> list[dict[str, Any]]:
    """List every processed session under ``OUTPUT_ROOT``, newest first.

    A processed session is a folder holding ``session_metrics.csv``; other
    folders (job scratch, unfinished runs) and loose files are skipped. The
    list is rebuilt from disk on every call, so it survives restarts.

    Returns:
        Dicts with ``session_id``, ``cameras`` (letters, e.g. ``["A", "B"]``),
        ``processed`` (date text), ``verdict_kind``, ``verdict_label`` and
        ``link`` (None when the folder name cannot be routed).
    """
    import pandas as pd

    if not OUTPUT_ROOT.is_dir():
        return []
    found: list[tuple[float, dict[str, Any]]] = []
    for session_dir in OUTPUT_ROOT.iterdir():
        metrics_path = session_dir / SESSION_METRICS_CSV
        if not session_dir.is_dir() or not metrics_path.is_file():
            continue
        mtime = metrics_path.stat().st_mtime
        try:
            kind, label = overall_session_verdict(pd.read_csv(metrics_path))
        except (OSError, ValueError):
            kind, label = ("unknown", "Could not read results")
        cameras = sorted(
            child.name.removeprefix("camera_").upper()
            for child in session_dir.glob("camera_*")
            if child.is_dir()
        )
        name = session_dir.name
        row = {
            "session_id": name,
            "cameras": cameras,
            "processed": datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M"),
            "verdict_kind": kind,
            "verdict_label": label,
            "link": f"/sessions/{name}" if _session_dir_is_addressable(name) else None,
        }
        found.append((mtime, row))
    found.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in found]


def _session_dir(session_id: str) -> Path:
    """Return a processed session folder or raise 404.

    Args:
        session_id: Session id from the URL (sanitized with ``_slug``).

    Returns:
        The session folder under ``OUTPUT_ROOT``.

    Raises:
        HTTPException: 404 when the id is invalid or the folder holds no
            ``session_metrics.csv``.
    """
    try:
        safe_id = _slug(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    session_dir = OUTPUT_ROOT / safe_id
    if not (session_dir / SESSION_METRICS_CSV).is_file():
        raise HTTPException(status_code=404, detail="Session not found")
    return session_dir


def resolve_session_file(session_dir: Path, relative_path: str) -> Path:
    """Resolve a report-linked file strictly inside one session folder.

    Args:
        session_dir: The session folder.
        relative_path: Path from the URL, relative to the session folder.

    Returns:
        The resolved file path.

    Raises:
        HTTPException: 404 for ``..`` parts, absolute or drive paths,
            disallowed file types, missing files, and anything (including a
            symlink target) that resolves outside the session folder.
    """
    not_found = HTTPException(status_code=404, detail="File not found")
    normalized = relative_path.replace("\\", "/")
    parts = normalized.split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or ":" in normalized
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise not_found
    if Path(normalized).suffix.lower() not in SESSION_FILE_SUFFIXES:
        raise not_found
    root = session_dir.resolve()
    candidate = (root / normalized).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise not_found
    return candidate


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
    return _render_page("login.html", "Login")


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
    """Render the Sessions home page: current jobs, then processed sessions."""
    return _render_page(
        "index.html",
        "Sessions",
        session_token=session_token,
        jobs=_job_rows(),
        sessions=session_rows(),
    )


@app.get("/new", response_class=HTMLResponse)
def new_session(session_token: Annotated[str, Depends(require_auth)]) -> HTMLResponse:
    """Render the new-session upload form."""
    return _render_page(
        "new_session.html",
        "New session",
        session_token=session_token,
        chunk_size=CHUNK_SIZE_BYTES,
        csrf_token=csrf_token_for_session(session_token),
    )


@app.get("/sessions/{session_id}")
def session_report(
    session_id: str,
    _: Annotated[None, Depends(require_auth)],
) -> RedirectResponse:
    """Open a session's report.

    The report links its figures and clips relative to the session folder
    (``camera_a/overlay_x.mp4``), so it is served from the files route, where
    those links resolve.
    """
    session_dir = _session_dir(session_id)
    if not (session_dir / SESSION_REPORT_HTML).is_file():
        raise HTTPException(status_code=404, detail="This session has no report yet")
    return RedirectResponse(
        f"/sessions/{session_dir.name}/files/{SESSION_REPORT_HTML}", status_code=303
    )


@app.get("/sessions/{session_id}/files/{file_path:path}")
def session_file(
    session_id: str,
    file_path: str,
    _: Annotated[None, Depends(require_auth)],
) -> FileResponse:
    """Serve one file from a session folder (report, figure, clip or table).

    Video clips support HTTP range requests through ``FileResponse``.
    """
    path = resolve_session_file(_session_dir(session_id), file_path)
    return FileResponse(path, content_disposition_type="inline")


@app.post("/uploads/init")
async def init_chunked_upload(
    _: Annotated[str, Depends(require_csrf)],
    session_id: Annotated[str, Form()],
    video_a_name: Annotated[str, Form()] = "",
    video_b_name: Annotated[str, Form()] = "",
    video_a_size: Annotated[int, Form()] = 0,
    video_b_size: Annotated[int, Form()] = 0,
    blocks_text: Annotated[str | None, Form()] = None,
    blocks_text_a: Annotated[str | None, Form()] = None,
    blocks_text_b: Annotated[str | None, Form()] = None,
    exclusions_text: Annotated[str, Form()] = "",
    exclusions_text_a: Annotated[str | None, Form()] = None,
    exclusions_text_b: Annotated[str | None, Form()] = None,
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
            blocks_text=resolved_blocks_text_a,
            blocks_text_b=resolved_blocks_text_b,
            exclusions_text=(
                exclusions_text if exclusions_text_a is None else exclusions_text_a
            ),
            exclusions_text_b=(
                exclusions_text if exclusions_text_b is None else exclusions_text_b
            ),
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
    session_token: Annotated[str, Depends(require_auth)],
) -> HTMLResponse:
    """Render one job status page."""
    safe_job_id = _slug(job_id)
    job_dir = JOB_ROOT / safe_job_id
    status = read_status(job_dir)
    result_zip = status.get("result_zip")
    return _render_page(
        "job_detail.html",
        f"Job {safe_job_id}",
        session_token=session_token,
        job_id=safe_job_id,
        state=str(status.get("state", "unknown")),
        chip_kind=job_state_chip(str(status.get("state", "unknown")))[0],
        chip_label=job_state_chip(str(status.get("state", "unknown")))[1],
        message=str(status.get("message", "")),
        show_annotate=status.get("state") == "awaiting_annotation",
        show_download=bool(result_zip and Path(result_zip).exists()),
        coverage_rows=_coverage_rows(job_dir),
        show_log_link=(job_dir / "job.log").exists(),
        auto_refresh=status.get("state") in {"uploading", "queued", "running"},
    )


def _coverage_rows(job_dir: Path) -> list[dict[str, Any]] | None:
    """Summarize per-block tracking coverage for a finished job.

    Args:
        job_dir: Portal job directory.

    Returns:
        One dict per camera x block row (``camera_id``, ``block``,
        ``coverage`` as a percent string, ``low``), or None when the job has
        no session table yet.
    """
    import pandas as pd

    config_path = job_dir / "job_config.json"
    if not config_path.exists():
        return None
    session_id = json.loads(config_path.read_text(encoding="utf-8")).get("session_id")
    metrics_path = OUTPUT_ROOT / str(session_id) / "session_metrics.csv"
    if not session_id or not metrics_path.exists():
        return None
    table = pd.read_csv(metrics_path, dtype={"camera_id": str, "block": str})
    rows: list[dict[str, Any]] = []
    for row in table.itertuples():
        share = float(getattr(row, "coverage_both", 0) or 0)
        rows.append(
            {
                "camera_id": str(row.camera_id),
                "block": str(row.block),
                "coverage": f"{share:.0%}",
                "low": bool(getattr(row, "low_coverage", share < 0.5)),
            }
        )
    return rows


@app.get("/metrics/all_sessions.csv")
def download_all_sessions_table(
    _: Annotated[None, Depends(require_auth)],
) -> FileResponse:
    """Download the combined metrics table across every processed session."""
    from video_analysis.session_report import ALL_SESSIONS_CSV, build_all_sessions_table

    out = OUTPUT_ROOT / ALL_SESSIONS_CSV
    build_all_sessions_table(OUTPUT_ROOT).to_csv(out, index=False)
    return FileResponse(out, filename=ALL_SESSIONS_CSV, media_type="text/csv")


def _gaze_label_context(job_id: str) -> tuple[str, Path, dict[str, list], float]:
    """Return (safe job id, session dir, blocks per camera, frame rate)."""
    safe_job_id = _slug(job_id)
    config_path = JOB_ROOT / safe_job_id / "job_config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    session_dir = OUTPUT_ROOT / _slug(str(config.get("session_id", "")))
    blocks = {
        camera_id: [SessionBlock(**block) for block in items]
        for camera_id, items in (config.get("blocks_by_camera") or {}).items()
        if (session_dir / camera_id / "tracks_2d_smooth.csv").exists()
    }
    if not blocks:
        raise HTTPException(
            status_code=404, detail="This job has no results to label yet"
        )
    return safe_job_id, session_dir, blocks, float(config.get("frame_rate", 6.0))


def _gaze_label_items(job_id: str) -> tuple[str, Path, dict[str, list[dict]]]:
    """Frames to label per camera, cached in the job folder."""
    from portal.gaze_labels import sample_gaze_label_frames

    safe_job_id, session_dir, blocks, frame_rate = _gaze_label_context(job_id)
    cache = JOB_ROOT / safe_job_id / "gaze_label_frames.json"
    if cache.exists():
        return safe_job_id, session_dir, json.loads(cache.read_text(encoding="utf-8"))
    items = {
        camera_id: sample_gaze_label_frames(
            session_dir / camera_id, camera_blocks, frame_rate=frame_rate
        )
        for camera_id, camera_blocks in blocks.items()
    }
    cache.write_text(json.dumps(items), encoding="utf-8")
    return safe_job_id, session_dir, items


@app.get("/jobs/{job_id}/gaze-label", response_class=HTMLResponse)
def gaze_label_page(
    job_id: str,
    session_token: Annotated[str, Depends(require_auth)],
) -> HTMLResponse:
    """Render the optional gaze labeling page for a finished job."""
    from portal.gaze_labels import GAZE_TARGETS, load_gaze_labels

    safe_job_id, session_dir, items = _gaze_label_items(job_id)
    existing = {
        camera_id: load_gaze_labels(session_dir / camera_id) for camera_id in items
    }
    csrf = csrf_token_for_session(session_token)
    payload = {
        "job": safe_job_id,
        "csrf": csrf,
        "frames_url": f"/jobs/{safe_job_id}/gaze-frames/",
        "save_url": f"/jobs/{safe_job_id}/gaze-labels",
        "items": items,
        "existing": existing,
        "targets": GAZE_TARGETS,
    }
    # The template embeds ``payload`` with ``|script_json``, which escapes
    # < > & so nothing in the data can close the <script> block.
    return _render_page(
        "gaze_label.html",
        f"Gaze labels {safe_job_id}",
        session_token=session_token,
        job_id=safe_job_id,
        payload=payload,
    )


@app.get("/jobs/{job_id}/gaze-frames/{camera_id}/{image_name}")
def gaze_label_frame(
    job_id: str,
    camera_id: str,
    image_name: str,
    _: Annotated[None, Depends(require_auth)],
) -> FileResponse:
    """Serve one frame image offered on the gaze labeling page."""
    _, session_dir, items = _gaze_label_items(job_id)
    names = {item["image_name"] for item in items.get(camera_id, [])}
    if image_name not in names:
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(session_dir / camera_id / "frames" / image_name)


def _save_gaze_label_submission(job_id: str, payload: dict) -> dict[str, int]:
    """Validate and store submitted gaze labels; return counts per camera."""
    from portal.gaze_labels import save_gaze_labels

    _, session_dir, items = _gaze_label_items(job_id)
    labels = payload.get("labels") or {}
    if not isinstance(labels, dict):
        raise HTTPException(status_code=400, detail="labels must be a JSON object")
    saved: dict[str, int] = {}
    for camera_id, entries in labels.items():
        if camera_id not in items:
            raise HTTPException(status_code=400, detail=f"Unknown camera: {camera_id}")
        try:
            save_gaze_labels(
                session_dir / camera_id,
                entries,
                labeler=str(payload.get("labeler", ""))[:40],
            )
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        saved[camera_id] = len(entries)
    return saved


@app.post("/jobs/{job_id}/gaze-labels")
async def submit_gaze_labels(
    job_id: str,
    request: Request,
    _csrf: Annotated[str, Depends(require_csrf)],
) -> JSONResponse:
    """Save the colleague's gaze labels next to the session outputs."""
    saved = _save_gaze_label_submission(job_id, await request.json())
    return JSONResponse({"saved": saved})


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
    session_token: Annotated[str, Depends(require_auth)],
) -> HTMLResponse:
    """Render the plain-text job log."""
    safe_job_id = _slug(job_id)
    log_path = JOB_ROOT / safe_job_id / "job.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not available")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return _render_page(
        "job_log.html", "Job Log", session_token=session_token, log_text=text
    )


def _read_anchor_manifests(job_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load every per-camera anchor manifest under a job directory."""
    anchors_root = job_dir / "anchors"
    manifests: dict[str, list[dict[str, Any]]] = {}
    if not anchors_root.exists():
        return manifests
    for camera_dir in sorted(anchors_root.iterdir()):
        if not camera_dir.is_dir():
            continue
        manifest_path = camera_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            manifests[camera_dir.name] = payload
    return manifests


def _convert_anchor_submission_to_filter_format(
    submission_entries: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert UI annotations into the schema video-filter-tracks reads.

    UI:   ``{anchor_idx, parent_track_id, child_track_id}`` or
          ``{anchor_idx, skipped: true}``.
    Filter expects: ``{frame_idx, parent_track_id, child_track_id, skipped}``.
    """
    manifest_by_idx = {int(m["anchor_idx"]): m for m in manifest}
    out: list[dict[str, Any]] = []
    for entry in submission_entries:
        anchor_idx = int(entry["anchor_idx"])
        if anchor_idx not in manifest_by_idx:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown anchor_idx {anchor_idx}",
            )
        anchor_meta = manifest_by_idx[anchor_idx]
        frame_idx = int(anchor_meta["frame_idx"])
        if entry.get("skipped"):
            out.append({"frame_idx": frame_idx, "skipped": True})
            continue
        if "parent_track_id" not in entry or "child_track_id" not in entry:
            raise HTTPException(
                status_code=400,
                detail=f"anchor_idx {anchor_idx}: missing parent/child track ids",
            )
        parent_tid = int(entry["parent_track_id"])
        child_tid = int(entry["child_track_id"])
        valid_tids = {int(t["track_id"]) for t in anchor_meta["tracks"]}
        if parent_tid not in valid_tids or child_tid not in valid_tids:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"anchor_idx {anchor_idx}: track id not in this frame's tracks"
                ),
            )
        if parent_tid == child_tid:
            raise HTTPException(
                status_code=400,
                detail=f"anchor_idx {anchor_idx}: parent and child must differ",
            )
        out.append(
            {
                "frame_idx": frame_idx,
                "parent_track_id": parent_tid,
                "child_track_id": child_tid,
            }
        )
    return out


@app.get("/jobs/{job_id}/anchors/{camera_id}/{filename}")
def anchor_image(
    job_id: str,
    camera_id: str,
    filename: str,
    _: Annotated[None, Depends(require_auth)],
) -> FileResponse:
    """Serve one rendered anchor JPEG."""
    safe_job_id = _slug(job_id)
    safe_camera = _slug(camera_id)
    safe_filename = _slug(filename.replace(".jpg", "")) + ".jpg"
    candidate = JOB_ROOT / safe_job_id / "anchors" / safe_camera / safe_filename
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Anchor image not found")
    return FileResponse(candidate, media_type="image/jpeg")


@app.get("/jobs/{job_id}/annotate", response_class=HTMLResponse)
def annotate_job(
    job_id: str,
    session_token: Annotated[str, Depends(require_auth)],
) -> HTMLResponse:
    """Render the parent/child annotation page for a job."""
    safe_job_id = _slug(job_id)
    job_dir = JOB_ROOT / safe_job_id
    manifests = _read_anchor_manifests(job_dir)
    if not manifests:
        raise HTTPException(
            status_code=404, detail="No anchor frames available for this job"
        )
    cameras: list[dict[str, Any]] = []
    for camera_id, manifest in manifests.items():
        cards: list[dict[str, Any]] = []
        for entry in manifest:
            tracks = []
            for track in entry.get("tracks", []):
                bgr = track.get("color_bgr", [0, 0, 0])
                # Manifest colors are OpenCV BGR; CSS wants r,g,b.
                rgb = f"{int(bgr[2])},{int(bgr[1])},{int(bgr[0])}"
                tracks.append({"track_id": int(track["track_id"]), "rgb": rgb})
            cards.append(
                {
                    "anchor_idx": int(entry["anchor_idx"]),
                    "image_filename": str(entry["image_filename"]),
                    "frame_idx": entry["frame_idx"],
                    "timestamp": f"{float(entry.get('timestamp_s', 0.0)):.2f}",
                    "tracks": tracks,
                }
            )
        cameras.append({"camera_id": camera_id, "cards": cards})
    # The page is GET-only; the form's POST goes through require_csrf, so the
    # token is handed to annotate.js through a data attribute on the form.
    csrf = csrf_token_for_session(session_token)
    return _render_page(
        "annotate.html",
        f"Annotate {safe_job_id}",
        session_token=session_token,
        job_id=safe_job_id,
        cameras=cameras,
        csrf_token=csrf,
    )


@app.post("/jobs/{job_id}/annotations")
async def submit_annotations(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    _csrf: Annotated[str, Depends(require_csrf)],
) -> JSONResponse:
    """Accept the colleague's anchor labels, then trigger phase 2."""
    safe_job_id = _slug(job_id)
    job_dir = JOB_ROOT / safe_job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    manifests = _read_anchor_manifests(job_dir)
    if not manifests:
        raise HTTPException(
            status_code=400, detail="No anchor frames to label for this job"
        )
    body = await request.json()
    annotations = body.get("annotations") or {}
    if not isinstance(annotations, dict):
        raise HTTPException(status_code=400, detail="annotations must be a JSON object")

    for camera_id, entries in annotations.items():
        if camera_id not in manifests:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown camera: {camera_id}",
            )
        if not isinstance(entries, list):
            raise HTTPException(
                status_code=400,
                detail=f"{camera_id} annotations must be a list",
            )
        converted = _convert_anchor_submission_to_filter_format(
            entries, manifests[camera_id]
        )
        out_path = job_dir / "anchors" / _slug(camera_id) / "anchors.json"
        out_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")

    write_status(
        job_dir,
        {
            "state": "queued",
            "message": "Annotations received; phase 2 queued.",
            "current_step": "phase 2 queued",
            "phase": 2,
        },
    )
    background_tasks.add_task(resume_portal_job_phase2_background, safe_job_id, job_dir)
    return JSONResponse({"job_id": safe_job_id, "status_url": f"/jobs/{safe_job_id}"})


def main() -> None:
    """Run the portal with uvicorn."""
    import uvicorn

    host = os.environ.get("PORTAL_HOST", "0.0.0.0")
    port = int(os.environ.get("PORTAL_PORT", "8080"))
    uvicorn.run("portal.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
