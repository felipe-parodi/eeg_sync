"""FastAPI web app for the local processing portal."""

from __future__ import annotations

import asyncio
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
    JOB_STAGES,
    MIN_PART_S,
    PORTAL_DEFAULT_FRAME_RATE,
    PORTAL_DEFAULT_MAX_PERSONS,
    PORTAL_DEFAULT_MODEL_PATH,
    PORTAL_TRACKER_BACKEND,
    PORTAL_TRACKER_NAME,
    CompressionSettings,
    ExclusionWindow,
    PortalJobConfig,
    SessionBlock,
    describe_step,
    failed_status,
    final_upload_path_for,
    format_clock,
    mark_chunk_received,
    parse_timecode,
    probe_video_duration,
    read_job_notes,
    read_status,
    resume_portal_job_phase2,
    run_portal_job,
    split_blocks_with_min_part,
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
# Held while a labels submission checks the job is waiting and marks it
# queued, so two quick submits can never both start phase 2.
ANNOTATION_SUBMIT_LOCK = threading.Lock()


def validate_startup_configuration() -> None:
    """Fail startup if required portal secrets are missing."""
    portal_password()
    portal_secret_key()


# Jobs in these states are doing work; a restart interrupts them.
ACTIVE_STATES = frozenset({"uploading", "queued", "running"})
# An upload whose status has not changed for this long was abandoned (browser
# closed) and must not block a restart forever.
STALE_UPLOAD_S = 30 * 60
# Exit code that tells start-portal.ps1 to start the server again (the
# tunnel keeps running, so the public link stays the same).
RESTART_EXIT_CODE = 3
RESTART_FLAG_NAME = ".restart_requested"
RESTART_POLL_S = 10.0


def restart_flag_path() -> Path:
    """File whose presence asks the portal to restart when idle."""
    return JOB_ROOT / RESTART_FLAG_NAME


def request_restart() -> Path:
    """Ask the running portal to restart as soon as no job is running.

    Returns:
        The request file written.
    """
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    flag = restart_flag_path()
    flag.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    return flag


def _job_dirs() -> list[Path]:
    if not JOB_ROOT.exists():
        return []
    return [
        d for d in JOB_ROOT.iterdir() if d.is_dir() and (d / "status.json").exists()
    ]


def portal_is_idle(now: float | None = None) -> bool:
    """True when no job is uploading, queued or running.

    Jobs waiting for parent labels do not count: a restart keeps them.
    Uploads untouched for ``STALE_UPLOAD_S`` count as abandoned.
    """
    now = time.time() if now is None else now
    for job_dir in _job_dirs():
        state = str(read_status(job_dir).get("state", ""))
        if state not in ACTIVE_STATES:
            continue
        if state == "uploading":
            age = now - (job_dir / "status.json").stat().st_mtime
            if age > STALE_UPLOAD_S:
                continue
        return False
    return True


def restart_due() -> bool:
    """Consume a pending restart request if the portal is idle.

    Returns:
        True when a restart was requested and nothing is running (the request
        file is then removed); False otherwise.
    """
    flag = restart_flag_path()
    if not flag.exists() or not portal_is_idle():
        return False
    flag.unlink(missing_ok=True)
    return True


def tidy_jobs_on_startup() -> dict[str, int]:
    """Keep job folders across restarts and clean up what cannot continue.

    - Every job folder is kept, so finished and failed jobs stay listed.
    - Jobs that were uploading, queued or running are marked failed with a
      plain "the portal was restarted" message (their work stopped).
    - Uploaded videos are kept only for jobs waiting for parent labels (phase
      2 needs them); others, and uploads without a job folder, are removed.

    Returns:
        Counts: ``kept`` job folders, ``interrupted`` jobs, ``uploads_removed``.
    """
    summary = {"kept": 0, "interrupted": 0, "uploads_removed": 0}
    keep_uploads: set[str] = set()
    for job_dir in _job_dirs():
        summary["kept"] += 1
        status = read_status(job_dir)
        state = str(status.get("state", ""))
        if state in ACTIVE_STATES:
            write_status(
                job_dir,
                {
                    **status,
                    "state": "failed",
                    "message": (
                        "The portal was restarted while this job was in progress, "
                        "so it stopped. Please submit the session again."
                    ),
                    "error": f"Interrupted by a portal restart (state was {state}).",
                },
            )
            summary["interrupted"] += 1
        elif state == "awaiting_annotation":
            keep_uploads.add(job_dir.name)
    if UPLOAD_ROOT.exists():
        for upload_dir in UPLOAD_ROOT.iterdir():
            if upload_dir.is_dir() and upload_dir.name not in keep_uploads:
                shutil.rmtree(upload_dir, ignore_errors=True)
                summary["uploads_removed"] += 1
    return summary


async def _restart_when_idle_loop() -> None:
    """Exit with ``RESTART_EXIT_CODE`` once a restart is requested and idle."""
    while True:
        await asyncio.sleep(RESTART_POLL_S)
        try:
            due = restart_due()
        except OSError:
            due = False
        if due:
            print(
                "Restart requested and no job is running: restarting now.", flush=True
            )
            os._exit(RESTART_EXIT_CODE)


@asynccontextmanager
async def portal_lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate configuration, tidy jobs, and watch for restart requests."""
    validate_startup_configuration()
    summary = tidy_jobs_on_startup()
    if summary["interrupted"]:
        print(
            f"Marked {summary['interrupted']} job(s) interrupted by the restart.",
            flush=True,
        )
    watcher = asyncio.create_task(_restart_when_idle_loop())
    try:
        yield
    finally:
        watcher.cancel()


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


VIDEO_LABELS = {"camera_a": "Video A", "camera_b": "Video B"}
# Offered in the task-name box; any other name is allowed.
TASK_NAME_SUGGESTIONS = ("grocery", "storybook", "synchrony")


def _parse_time_field(raw: str, what: str, field_name: str) -> float:
    """Parse one form time, naming the task or window in any error.

    Args:
        raw: Time text as typed.
        what: Plain description of the row, e.g. ``'Video A task "grocery"'``.
        field_name: ``"start"`` or ``"end"``.

    Returns:
        Time in seconds.

    Raises:
        ValueError: With the row, the field and the parsing problem.
    """
    try:
        return parse_timecode(raw)
    except ValueError as error:
        raise ValueError(f"{what} has a bad {field_name} time. {error}") from error


def _parse_blocks_text(text: str, video: str = "Video A") -> list[SessionBlock]:
    """Parse one camera's task lines (``name,start,end[,color]``).

    Args:
        text: Task text as posted by the form, one task per line.
        video: Plain camera name used in messages (``"Video A"``).

    Returns:
        Parsed tasks with slugged names.

    Raises:
        ValueError: With a plain message for a malformed line, a bad time,
            an empty name, end not after start, a task under ``MIN_PART_S``,
            or two tasks with the same (slugged) name.
    """
    blocks: list[SessionBlock] = []
    raw_names: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) not in {3, 4}:
            raise ValueError(f"{video} task line {line_number} must be: name,start,end")
        name, start_raw, end_raw = parts[:3]
        try:
            safe_name = _slug(name)
        except ValueError as error:
            raise ValueError(
                f'{video} task {line_number} name "{name}" has no letters or numbers.'
                if name
                else f"{video} task {line_number} has no name."
            ) from error
        what = f'{video} task "{name}"'
        start_s = _parse_time_field(start_raw, what, "start")
        end_s = _parse_time_field(end_raw, what, "end")
        if end_s <= start_s:
            raise ValueError(
                f"{what} must end after it starts (start {start_raw}, end {end_raw})."
            )
        if end_s - start_s < MIN_PART_S:
            raise ValueError(
                f"{what} is only {end_s - start_s:g} s long. "
                f"Tasks must be at least {MIN_PART_S:g} s."
            )
        if safe_name in raw_names:
            other = raw_names[safe_name]
            if other == name:
                raise ValueError(
                    f'{video} has two tasks named "{name}". '
                    "Give each task a different name."
                )
            raise ValueError(
                f'{video} task names "{other}" and "{name}" both become '
                f'"{safe_name}" in file names. Give each task a different name.'
            )
        raw_names[safe_name] = name
        blocks.append(
            SessionBlock(
                name=safe_name,
                start_s=start_s,
                end_s=end_s,
                color=parts[3] if len(parts) == 4 and parts[3] else "gray",
            )
        )
    if not blocks:
        raise ValueError(f"{video} needs at least one task.")
    return blocks


def _parse_exclusions_text(text: str, video: str = "Video A") -> list[ExclusionWindow]:
    """Parse one camera's extra-person windows (``start,end[,reason]``).

    Args:
        text: Window text as posted by the form, one window per line.
        video: Plain camera name used in messages (``"Video A"``).

    Returns:
        Parsed windows.

    Raises:
        ValueError: With a plain message for a malformed line, a bad time,
            or end not after start.
    """
    exclusions: list[ExclusionWindow] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        if len(parts) < 2:
            raise ValueError(
                f"{video} extra-person window line {line_number} must be: "
                "start,end,reason"
            )
        what = f"{video} extra-person window {line_number}"
        start_s = _parse_time_field(parts[0], what, "start")
        end_s = _parse_time_field(parts[1], what, "end")
        if end_s <= start_s:
            raise ValueError(
                f"{what} must end after it starts (start {parts[0]}, end {parts[1]})."
            )
        reason = parts[2] if len(parts) == 3 else ""
        exclusions.append(ExclusionWindow(start_s=start_s, end_s=end_s, reason=reason))
    return exclusions


def _part_number(name: str) -> int:
    """Return N for ``<task>_partN`` names, else 0 (for sorting)."""
    tail = name.rsplit("_part", 1)[-1]
    return int(tail) if "_part" in name and tail.isdigit() else 0


def plan_camera_blocks(
    blocks: list[SessionBlock],
    exclusions: list[ExclusionWindow],
    video: str = "Video A",
) -> tuple[list[SessionBlock], list[SessionBlock]]:
    """Split one camera's tasks around its extra-person windows and check them.

    Args:
        blocks: The camera's parsed tasks.
        exclusions: The camera's extra-person windows.
        video: Plain camera name used in messages (``"Video A"``).

    Returns:
        ``(kept, dropped)``: analysis parts to process, and parts shorter
        than ``MIN_PART_S`` that are skipped.

    Raises:
        ValueError: If a task lies entirely inside a window, if every piece
            of a task is too short, or if a task's name equals a part name
            generated for another task (``story_part1`` when ``story`` is
            split).
    """
    kept: list[SessionBlock] = []
    dropped: list[SessionBlock] = []
    generated_by: dict[str, tuple[str, list[str]]] = {}
    for block in blocks:
        block_kept, block_dropped = split_blocks_with_min_part([block], exclusions)
        if not block_kept and not block_dropped:
            raise ValueError(
                f'{video} task "{block.name}" ({format_clock(block.start_s)} to '
                f"{format_clock(block.end_s)}) is entirely inside an extra-person "
                "window, so none of it can be analysed. Remove the task or "
                "correct the window."
            )
        if not block_kept:
            raise ValueError(
                "After removing extra-person windows, every piece of "
                f'{video} task "{block.name}" is shorter than {MIN_PART_S:g} s. '
                "Remove the task or correct the windows."
            )
        part_names = sorted(
            (part.name for part in block_kept + block_dropped), key=_part_number
        )
        if part_names != [block.name]:
            for part_name in part_names:
                generated_by[part_name] = (block.name, part_names)
        kept.extend(block_kept)
        dropped.extend(block_dropped)
    for block in blocks:
        if block.name in generated_by:
            owner, part_names = generated_by[block.name]
            listed = ", ".join(part_names[:-1]) + f" and {part_names[-1]}"
            raise ValueError(
                f'{video} task name "{block.name}" clashes with a piece of task '
                f'"{owner}", which an extra-person window splits into {listed}. '
                f'Rename "{block.name}".'
            )
    return kept, dropped


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
    replace_existing: bool = False,
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
        replace_existing: Whether the user agreed to move an existing
            ``OUTPUT_ROOT/<session_id>`` folder to the archive.

    Returns:
        JSON-serializable submission metadata. ``dropped_parts_by_camera``
        lists the parts skipped for being shorter than ``MIN_PART_S``.

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
        "camera_a": _parse_exclusions_text(exclusions_text_a, "Video A"),
        "camera_b": _parse_exclusions_text(exclusions_text_b, "Video B"),
    }

    input_blocks_by_camera: dict[str, list[SessionBlock]] = {}
    analysis_blocks_by_camera: dict[str, list[SessionBlock]] = {}
    dropped_parts_by_camera: dict[str, list[dict[str, Any]]] = {}
    for camera_id, has_video, camera_blocks_text in (
        ("camera_a", has_video_a, blocks_text_a),
        ("camera_b", has_video_b, blocks_text_b),
    ):
        if not has_video:
            continue
        video = VIDEO_LABELS[camera_id]
        camera_blocks = _parse_blocks_text(camera_blocks_text, video)
        kept, dropped = plan_camera_blocks(
            camera_blocks, exclusions_by_camera[camera_id], video
        )
        input_blocks_by_camera[camera_id] = camera_blocks
        analysis_blocks_by_camera[camera_id] = kept
        dropped_parts_by_camera[camera_id] = [
            {
                "name": part.name,
                "start_s": part.start_s,
                "end_s": part.end_s,
                "duration_s": round(part.end_s - part.start_s, 3),
                "reason": (
                    f"shorter than {MIN_PART_S:g} s after removing "
                    "extra-person windows"
                ),
            }
            for part in dropped
        ]

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
        "min_part_s": MIN_PART_S,
        "dropped_parts_by_camera": dropped_parts_by_camera,
        "replace_existing": replace_existing,
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
        stage = read_status(job_dir).get("stage") or "upload"
        write_status(
            job_dir,
            {
                "state": "queued",
                "message": "Waiting for another job to finish first.",
                "stage": stage,
                "current_step": "queued",
            },
        )
        JOB_RUN_LOCK.acquire()
    try:
        action()
    finally:
        JOB_RUN_LOCK.release()


def _record_background_failure(job_dir: Path, error: BaseException, phase: int) -> None:
    """Mark a job failed with a plain message, keeping a richer one.

    The pipeline functions already write a failed status naming the step that
    stopped; that one is kept. Otherwise the step comes from the last status.

    Args:
        job_dir: Job working directory.
        error: The exception that stopped the job.
        phase: Pipeline phase (1 before labeling, 2 after).
    """
    status = read_status(job_dir)
    if status.get("state") == "failed" and status.get("error"):
        return
    write_status(
        job_dir,
        failed_status(status.get("current_step"), error, phase=phase),
    )


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
        _record_background_failure(job_dir, error, phase=1)
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
        _record_background_failure(job_dir, error, phase=2)
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
    ``docs/plans/metrics-validity_261116.md`` is present it decides: any
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
        if session_dir.name.startswith("_"):
            # ``_archive`` (replaced runs) and other portal-internal folders.
            continue
        metrics_path = session_dir / SESSION_METRICS_CSV
        if not session_dir.is_dir() or not metrics_path.is_file():
            continue
        mtime = metrics_path.stat().st_mtime
        # The task table (pieces of a split task joined) is what the lab
        # analyses; pieces can each be short while their task has enough data.
        tasks_path = session_dir / "task_metrics.csv"
        verdict_path = tasks_path if tasks_path.is_file() else metrics_path
        try:
            kind, label = overall_session_verdict(pd.read_csv(verdict_path))
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


ARCHIVE_DIR_NAME = "_archive"
# A task may end this much after the measured video end (container rounding,
# browser vs ffprobe disagreement) before it counts as past the end.
VIDEO_END_TOLERANCE_S = 1.0


class SessionIdInUseError(ValueError):
    """The session ID already has results or a running job (HTTP 409)."""


def _reserved_session_ids() -> set[str]:
    """Return folder names under ``OUTPUT_ROOT`` that are not sessions.

    Returns:
        Lower-cased names (compared case-insensitively, as on Windows).
    """
    return {JOB_ROOT.name.lower(), ARCHIVE_DIR_NAME}


def _active_job_for_session(safe_id: str, ignore_job_id: str = "") -> str | None:
    """Return a job that is still uploading or processing this session.

    Args:
        safe_id: Slugged session ID.
        ignore_job_id: Job to leave out (the caller's own job).

    Returns:
        The job ID, or None.
    """
    if not JOB_ROOT.is_dir():
        return None
    for meta_path in JOB_ROOT.glob("*/upload_meta.json"):
        job_dir = meta_path.parent
        if job_dir.name == ignore_job_id:
            continue
        if read_status(job_dir).get("state") not in ACTIVE_JOB_STATES:
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(meta.get("session_id", "")).lower() == safe_id.lower():
            return job_dir.name
    return None


def session_id_status(session_id: str) -> dict[str, Any]:
    """Describe whether a session ID can be used for a new run.

    Args:
        session_id: Session ID as typed.

    Returns:
        ``session_id`` (slugged, as used for folders), ``valid`` (false for
        empty or reserved IDs), ``exists`` (``OUTPUT_ROOT/<id>`` exists),
        ``active_job`` (another job is uploading or processing it) and a
        plain ``message`` for invalid IDs.
    """
    try:
        safe_id = _slug(session_id)
    except ValueError:
        return {
            "session_id": "",
            "valid": False,
            "exists": False,
            "active_job": False,
            "message": "The session ID needs at least one letter or number.",
        }
    if safe_id.lower() in _reserved_session_ids():
        return {
            "session_id": safe_id,
            "valid": False,
            "exists": False,
            "active_job": False,
            "message": (
                f'"{safe_id}" is used by the portal itself. '
                "Choose another session ID."
            ),
        }
    return {
        "session_id": safe_id,
        "valid": True,
        "exists": (OUTPUT_ROOT / safe_id).exists(),
        "active_job": _active_job_for_session(safe_id) is not None,
        "message": "",
    }


def check_session_id_free(
    safe_id: str, replace_existing: bool, ignore_job_id: str = ""
) -> bool:
    """Refuse a session ID that would mix a new run with earlier results.

    Args:
        safe_id: Slugged session ID.
        replace_existing: Whether the user ticked "Replace the earlier
            results".
        ignore_job_id: The caller's own job, left out of the running-job
            check.

    Returns:
        True when ``OUTPUT_ROOT/<safe_id>`` exists (and replacing was
        allowed), so the caller should archive it.

    Raises:
        ValueError: For a reserved ID.
        SessionIdInUseError: When results exist and replacing was not allowed, or
            another job is still working on the same session.
    """
    if safe_id.lower() in _reserved_session_ids():
        raise ValueError(
            f'"{safe_id}" is used by the portal itself. Choose another session ID.'
        )
    if _active_job_for_session(safe_id, ignore_job_id) is not None:
        raise SessionIdInUseError(
            f'Session "{safe_id}" is already being processed. Wait for that job '
            "to finish, or choose a new session ID."
        )
    exists = (OUTPUT_ROOT / safe_id).exists()
    if exists and not replace_existing:
        raise SessionIdInUseError(
            f'Session "{safe_id}" already has results. Choose a new session ID, '
            'or tick "Replace the earlier results" to move the old results to '
            "the archive folder."
        )
    return exists


def archive_session_dir(safe_id: str, stamp: str | None = None) -> Path | None:
    """Move a session's earlier results to ``OUTPUT_ROOT/_archive``.

    Nothing is deleted and nothing is overwritten: the folder becomes
    ``_archive/<id>_<YYYYmmdd_HHMMSS>`` (with ``_2``, ``_3``... if that name
    is taken). The home page and the study-wide table ignore ``_archive``.

    Args:
        safe_id: Slugged session ID.
        stamp: Timestamp text; defaults to now.

    Returns:
        The archived folder, or None when there was nothing to move.
    """
    source = OUTPUT_ROOT / safe_id
    if not source.exists():
        return None
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = OUTPUT_ROOT / ARCHIVE_DIR_NAME
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f"{safe_id}_{stamp}"
    suffix = 2
    while target.exists():
        target = archive_root / f"{safe_id}_{stamp}_{suffix}"
        suffix += 1
    shutil.move(str(source), str(target))
    return target


def _plural(count: int, unit: str) -> str:
    """Return ``"1 minute"`` / ``"6 minutes"``."""
    return f"{count} {unit}{'' if count == 1 else 's'}"


def check_uploaded_video_lengths(upload_dir: Path, meta: dict) -> dict[str, float]:
    """Check every uploaded video and that no task ends after its video does.

    Runs ffprobe on each assembled upload (``PORTAL_FFPROBE_BIN``). This is
    the server's re-check of the browser's length check, and also catches
    hours typed where minutes were meant (``13:06:00`` for ``13:06``).

    Args:
        upload_dir: The job's upload folder.
        meta: The job's ``upload_meta.json`` contents.

    Returns:
        Video length in seconds per camera ID.

    Raises:
        ValueError: Naming the task and the video length when a task ends
            after the video does.
        FileNotFoundError, RuntimeError: When an upload is incomplete or
            ffprobe cannot read it.
    """
    ffprobe_bin = os.environ.get("PORTAL_FFPROBE_BIN", "ffprobe")
    tasks_by_camera = meta.get("input_analysis_blocks_by_camera", {})
    durations: dict[str, float] = {}
    for camera_id, video_meta in meta["videos"].items():
        path = validate_direct_chunked_upload(
            upload_dir=upload_dir,
            camera_id=camera_id,
            original_filename=video_meta["original_filename"],
            total_chunks=int(video_meta["chunk_count"]),
            expected_size_bytes=int(video_meta["size_bytes"]),
            probe=True,
            ffprobe_bin=ffprobe_bin,
        )
        duration_s = probe_video_duration(path, ffprobe_bin)
        durations[camera_id] = duration_s
        video = VIDEO_LABELS.get(camera_id, camera_id)
        for task in tasks_by_camera.get(camera_id, []):
            end_s = float(task["end_s"])
            if end_s <= duration_s + VIDEO_END_TOLERANCE_S:
                continue
            message = (
                f'{video} task "{task["name"]}" ends at {format_clock(end_s)}, '
                f"but {video} is only {format_clock(duration_s)} long. "
                "Check the task times."
            )
            hours, rest = divmod(int(end_s), 3600)
            minutes, seconds = divmod(rest, 60)
            if hours and seconds == 0 and end_s == int(end_s):
                message += (
                    f" For {_plural(hours, 'minute')} {_plural(minutes, 'second')}, "
                    f"type {hours}:{minutes:02d}."
                )
            raise ValueError(message)
    return durations


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
        min_part_s=MIN_PART_S,
        video_end_tolerance_s=VIDEO_END_TOLERANCE_S,
        task_suggestions=TASK_NAME_SUGGESTIONS,
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


@app.get("/sessions/{session_id}/exists")
def session_exists(
    session_id: str,
    _: Annotated[None, Depends(require_auth)],
) -> JSONResponse:
    """Tell the new-session form whether this ID already has results.

    The form shows a warning and asks for "Replace the earlier results";
    ``/uploads/init`` enforces the same rule.
    """
    return JSONResponse(session_id_status(session_id))


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
    replace_existing: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Create upload metadata for a chunked browser upload.

    Refuses (409) a session ID that already has results unless
    ``replace_existing`` is ``"yes"``, and any ID another job is still
    working on. The earlier results are archived only when the upload
    finishes.
    """
    try:
        try:
            safe_session_id = _slug(session_id)
        except ValueError as error:
            raise ValueError(
                "The session ID needs at least one letter or number."
            ) from error
        replace = replace_existing == "yes"
        check_session_id_free(safe_session_id, replace)
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
            replace_existing=replace,
        )
        upload_dir.mkdir(parents=True, exist_ok=True)
        job_dir.mkdir(parents=True, exist_ok=True)
        _write_submission_metadata(job_dir, metadata)
        write_status(
            job_dir,
            {
                "state": "uploading",
                "message": "Receiving the videos.",
                "stage": "upload",
                "current_step": "uploading",
            },
        )
        return JSONResponse(
            {
                "job_id": job_id,
                "chunk_size": CHUNK_SIZE_BYTES,
                "status_url": f"/jobs/{job_id}",
            }
        )
    except SessionIdInUseError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
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
    """Check the uploaded videos and start processing.

    Before queueing: ffprobe re-checks every task against its video's real
    length, the session ID is re-checked, and when the user chose to replace
    earlier results they are moved to ``OUTPUT_ROOT/_archive``. A failed
    check marks the job failed with the same plain message.
    """
    try:
        safe_job_id = _slug(job_id)
        upload_dir = UPLOAD_ROOT / safe_job_id
        job_dir = JOB_ROOT / safe_job_id
        meta = _read_upload_meta(job_dir)
        try:
            durations = check_uploaded_video_lengths(upload_dir, meta)
            must_archive = check_session_id_free(
                str(meta["session_id"]),
                bool(meta.get("replace_existing", False)),
                ignore_job_id=safe_job_id,
            )
        except Exception as error:
            write_status(
                job_dir,
                {
                    "state": "failed",
                    "message": str(error),
                    "stage": "upload",
                    "current_step": None,
                },
            )
            raise
        for camera_id, duration_s in durations.items():
            meta["videos"][camera_id]["duration_s"] = duration_s
        meta["upload_finished_at"] = datetime.now().isoformat(timespec="seconds")
        if must_archive:
            archived = archive_session_dir(str(meta["session_id"]))
            meta["archived_previous_results"] = str(archived) if archived else None
        _write_submission_metadata(job_dir, meta)
        write_status(
            job_dir,
            {
                "state": "queued",
                "message": "Upload complete. Processing starts shortly.",
                "stage": "upload",
                "current_step": "queued",
            },
        )
        background_tasks.add_task(
            run_chunked_portal_job, safe_job_id, upload_dir, job_dir
        )
        return JSONResponse(
            {"job_id": safe_job_id, "status_url": f"/jobs/{safe_job_id}"}
        )
    except SessionIdInUseError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


# Job states the page keeps polling for; any other state is final for now.
POLLING_JOB_STATES = frozenset({"uploading", "queued", "running"})
# Lines of job.log shown under "Show technical details" on a failed job.
FAILURE_LOG_LINES = 30


def job_stage_rows(status: dict[str, Any]) -> list[dict[str, str]]:
    """Return the job page's stage list with each stage's progress.

    Args:
        status: The job's ``status.json`` payload.

    Returns:
        One dict per ``JOB_STAGES`` entry with ``key``, ``label`` and
        ``status`` (``done``, ``current``, ``failed`` or ``todo``).
    """
    state = str(status.get("state", ""))
    stage = str(
        status.get("stage") or describe_step(status.get("current_step"))["stage"]
    )
    if state == "awaiting_annotation":
        stage = "label_parent"
    keys = [key for key, _ in JOB_STAGES]
    current = keys.index(stage) if stage in keys else -1
    rows = []
    for index, (key, label) in enumerate(JOB_STAGES):
        if state == "complete" or index < current:
            progress = "done"
        elif index == current:
            progress = "failed" if state == "failed" else "current"
        else:
            progress = "todo"
        rows.append({"key": key, "label": label, "status": progress})
    return rows


def _log_tail(job_dir: Path, lines: int = FAILURE_LOG_LINES) -> str:
    """Return the last lines of a job's log (empty when there is no log).

    Args:
        job_dir: Job working directory.
        lines: How many lines to keep.

    Returns:
        The log tail as one string.
    """
    log_path = job_dir / "job.log"
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.rstrip().splitlines()[-lines:])


def job_status_view(job_id: str, job_dir: Path) -> dict[str, Any]:
    """Build the job status shown on the job page and in ``status.json``.

    Args:
        job_id: Sanitized job id.
        job_dir: Job working directory.

    Returns:
        JSON-safe dict with ``job_id``, ``state``, ``chip_kind``,
        ``chip_label``, plain ``message``, ``stage``, ``stages``, per-video
        ``cameras`` messages, ``notes``, ``polling``, ``annotate_url`` (only
        while waiting for labels) and, for a failed job, ``error`` and
        ``log_tail``.
    """
    status = read_status(job_dir)
    state = str(status.get("state", "unknown"))
    kind, label = job_state_chip(state)
    cameras = status.get("cameras") if state == "running" else None
    message = str(status.get("message", ""))
    if cameras and len(cameras) > 1:
        # Each video's step is listed under the headline.
        message = "Working on both videos at the same time."
    view: dict[str, Any] = {
        "job_id": job_id,
        "state": state,
        "chip_kind": kind,
        "chip_label": label,
        "message": message,
        "stage": str(status.get("stage") or ""),
        "stages": job_stage_rows(status),
        "cameras": [
            {"video": VIDEO_LABELS.get(str(cid), str(cid)), "message": str(msg)}
            for cid, msg in (cameras or {}).items()
        ],
        "notes": read_job_notes(job_dir),
        "polling": state in POLLING_JOB_STATES,
        "annotate_url": (
            f"/jobs/{job_id}/annotate" if state == "awaiting_annotation" else None
        ),
    }
    if state == "failed":
        view["error"] = str(status.get("error", ""))
        view["log_tail"] = _log_tail(job_dir)
    return view


@app.get("/jobs/{job_id}/status.json")
def job_status_json(
    job_id: str,
    _: Annotated[None, Depends(require_auth)],
) -> JSONResponse:
    """Return the job's current status for the job page to poll."""
    safe_job_id = _slug(job_id)
    job_dir = JOB_ROOT / safe_job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job_status_view(safe_job_id, job_dir))


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(
    job_id: str,
    session_token: Annotated[str, Depends(require_auth)],
) -> HTMLResponse:
    """Render one job status page."""
    safe_job_id = _slug(job_id)
    job_dir = JOB_ROOT / safe_job_id
    status = read_status(job_dir)
    view = job_status_view(safe_job_id, job_dir)
    result_zip = status.get("result_zip")
    is_complete = view["state"] == "complete"
    return _render_page(
        "job_detail.html",
        f"Job {safe_job_id}",
        session_token=session_token,
        job=view,
        job_id=safe_job_id,
        show_annotate=view["annotate_url"] is not None,
        show_download=bool(result_zip and Path(result_zip).exists()),
        results_url=_results_url(job_dir) if is_complete else None,
        # A new or failed run must not show an earlier run's table.
        coverage_rows=_coverage_rows(job_dir) if is_complete else None,
        dropped_parts=_dropped_part_rows(job_dir),
        min_part_s=f"{MIN_PART_S:g}",
        show_log_link=(job_dir / "job.log").exists(),
    )


def _results_url(job_dir: Path) -> str | None:
    """Return the session report link for a finished job.

    Args:
        job_dir: Portal job directory.

    Returns:
        ``/sessions/<session_id>`` when the job's session has a
        ``report.html``, else None.
    """
    config_path = job_dir / "job_config.json"
    if not config_path.exists():
        return None
    try:
        session_id = json.loads(config_path.read_text(encoding="utf-8")).get(
            "session_id"
        )
    except ValueError:
        return None
    if not session_id or not _session_dir_is_addressable(str(session_id)):
        return None
    if not (OUTPUT_ROOT / str(session_id) / SESSION_REPORT_HTML).is_file():
        return None
    return f"/sessions/{session_id}"


def _dropped_part_rows(job_dir: Path) -> list[dict[str, str]]:
    """List the analysis parts skipped for being shorter than ``MIN_PART_S``.

    Args:
        job_dir: Portal job directory.

    Returns:
        Dicts with ``video``, ``name``, ``start``, ``end`` and ``duration``
        (display text); empty when nothing was skipped or the job has no
        metadata.
    """
    try:
        meta = _read_upload_meta(job_dir)
    except (FileNotFoundError, ValueError):
        return []
    rows: list[dict[str, str]] = []
    for camera_id, parts in (meta.get("dropped_parts_by_camera") or {}).items():
        for part in parts:
            rows.append(
                {
                    "video": VIDEO_LABELS.get(camera_id, camera_id),
                    "name": str(part.get("name", "")),
                    "start": format_clock(float(part["start_s"])),
                    "end": format_clock(float(part["end_s"])),
                    "duration": f"{float(part['duration_s']):g} s",
                }
            )
    return rows


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


@app.get("/metrics/all_sessions_task_metrics.csv")
def download_study_task_table(
    _: Annotated[None, Depends(require_auth)],
) -> FileResponse:
    """Download the study task table (one row per session, camera and task).

    The session-report step rewrites it after every finished job.
    """
    from video_analysis.session_report import ALL_SESSIONS_TASK_CSV

    path = OUTPUT_ROOT / ALL_SESSIONS_TASK_CSV
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail="No study task table yet: finish a session first"
        )
    return FileResponse(path, filename=ALL_SESSIONS_TASK_CSV, media_type="text/csv")


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
                box = track.get("bbox_xyxy")
                bbox = (
                    ",".join(f"{float(v):g}" for v in box)
                    if isinstance(box, list) and len(box) == 4
                    else ""
                )
                tracks.append(
                    {"track_id": int(track["track_id"]), "rgb": rgb, "bbox": bbox}
                )
            cards.append(
                {
                    "anchor_idx": int(entry["anchor_idx"]),
                    "image_filename": str(entry["image_filename"]),
                    "frame_idx": entry["frame_idx"],
                    "timestamp": f"{float(entry.get('timestamp_s', 0.0)):.2f}",
                    "tracks": tracks,
                }
            )
        cameras.append(
            {
                "camera_id": camera_id,
                "label": VIDEO_LABELS.get(camera_id, camera_id),
                "cards": cards,
            }
        )
    # The page is GET-only; the form's POST goes through require_csrf, so the
    # token is handed to annotate.js through a data attribute on the form.
    csrf = csrf_token_for_session(session_token)
    return _render_page(
        "annotate.html",
        f"Annotate {safe_job_id}",
        session_token=session_token,
        job_id=safe_job_id,
        cameras=cameras,
        total_cards=sum(len(camera["cards"]) for camera in cameras),
        csrf_token=csrf,
    )


@app.post("/jobs/{job_id}/annotations")
async def submit_annotations(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    _csrf: Annotated[str, Depends(require_csrf)],
) -> JSONResponse:
    """Accept the colleague's anchor labels, then trigger phase 2.

    Labels are taken only while the job is waiting for them; any other state
    (queued, running, complete, failed), an empty submission, or a second
    submit returns 409 with a plain message, so phase 2 starts once.
    """
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
    annotations = body.get("annotations") if isinstance(body, dict) else None
    annotations = annotations or {}
    if not isinstance(annotations, dict):
        raise HTTPException(status_code=400, detail="annotations must be a JSON object")
    if not any(isinstance(v, list) and v for v in annotations.values()):
        raise HTTPException(
            status_code=409,
            detail=(
                "No frames were labeled. Mark the parent (or skip) on at "
                "least one frame, then submit again."
            ),
        )

    with ANNOTATION_SUBMIT_LOCK:
        state = str(read_status(job_dir).get("state", ""))
        if state != "awaiting_annotation":
            raise HTTPException(
                status_code=409, detail=_not_waiting_for_labels_message(state)
            )
        converted_by_camera: dict[str, list[dict[str, Any]]] = {}
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
            converted_by_camera[camera_id] = (
                _convert_anchor_submission_to_filter_format(
                    entries, manifests[camera_id]
                )
            )
        for camera_id, converted in converted_by_camera.items():
            out_path = job_dir / "anchors" / _slug(camera_id) / "anchors.json"
            out_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")
        write_status(
            job_dir,
            {
                "state": "queued",
                "message": describe_step("phase 2 queued")["message"],
                "stage": "track_refine",
                "current_step": "phase 2 queued",
                "phase": 2,
            },
        )
    background_tasks.add_task(resume_portal_job_phase2_background, safe_job_id, job_dir)
    return JSONResponse({"job_id": safe_job_id, "status_url": f"/jobs/{safe_job_id}"})


def _not_waiting_for_labels_message(state: str) -> str:
    """Plain reason a labels submission was refused.

    Args:
        state: The job's current state.

    Returns:
        One or two short sentences for the person labeling.
    """
    reasons = {
        "queued": "Labels for this job were already received.",
        "running": "This job is already running with the labels it received.",
        "complete": "This job is already finished.",
        "failed": "This job stopped with an error.",
        "uploading": "This job is still uploading.",
    }
    reason = reasons.get(state, "This job is not waiting for labels.")
    return f"{reason} Nothing was changed. Go back to the job page to see where it is."


def main() -> None:
    """Run the portal with uvicorn."""
    import uvicorn

    host = os.environ.get("PORTAL_HOST", "0.0.0.0")
    port = int(os.environ.get("PORTAL_PORT", "8080"))
    uvicorn.run("portal.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
