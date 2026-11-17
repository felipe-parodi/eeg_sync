# Portal Revamp Plan (UI/UX and Aesthetics)

## Goal

A non-technical research assistant can submit a session, follow progress, label
parent vs child, and decide which blocks to use, without reading logs or CSVs.
Runs in parallel with `metrics-validity_261116.md`; the results page consumes
that plan's column contract (`valid_seconds_both`, `quality_verdict`,
`quality_reasons`, `task_metrics.csv`) and falls back to `coverage_both` until
those columns exist.

Branch: `feat/portal-revamp` (from `feat/pose-quality-overhaul`).

## Current State (review, 2026-11-16)

- All pages are HTML/CSS/JS inside Python f-strings in `portal/app.py` (2,140
  lines); one inline stylesheet in `_html_page`, page-specific styles scattered.
- Upload form: free-text blocks/exclusions (`name,start,end[,color]`), one long
  single-column form; no preview of how exclusions split a task into
  `_part1/2/3`, no validation against video length.
- Job page: raw `state` and step names ("Phase 2 step 7 of 15: interpolate tracks
  camera_a"), full-page reload every 10 s.
- Anchor labeling: `.anchor-grid`, `.anchor-card`, `.chosen` have no CSS, so
  clicking a button gives almost no visible feedback; labels via "Track N is
  parent" buttons rather than clicking the person.
- Results: coverage table with "Low: check the overlay clips" / "OK"; overlay
  clips only inside the zip; no recommendation per block; no task-level view.
- Job list wiped on every restart (`purge_all_jobs`), so links to finished
  sessions disappear even though outputs are kept.
- `report.html` (in the zip) has its own separate styling.

## Design Principles

- Plain language everywhere; technical names only in the log.
- One primary action per page.
- Show, don't tell: timeline for blocks, inline clips for quality.
- Server-rendered Jinja templates + one stylesheet + small vanilla JS. No build
  step, no JS framework, same FastAPI process.
- Accessible: keyboard operable, visible focus, color never the only signal
  (verdict chips carry text + icon), colorblind-safe palette.
- All visual results must be **checked by the user** in a browser before a phase
  is called done.

## Status

- Phase 1 done (templates). Phase 2 done (design system, Sessions home page,
  session report route). Visual check by the user pending.
- Phase 3 done (2026-11-16): numbered step form with task / extra-person rows,
  per-camera timeline preview, copy A to B, browser checks mirrored on the
  server (`parse_timecode`, `_parse_blocks_text`, `plan_camera_blocks`),
  `MIN_PART_S = 10` with dropped parts recorded and listed on the job page,
  ffprobe length re-check in `/uploads/{id}/finish`, session-ID reuse check
  (`GET /sessions/{id}/exists`, 409 at init, archive to
  `output/_archive/<id>_<stamp>` at finish), job page shows results only when
  complete. M:SS accepts minutes >= 60 (`75:30`); only H:MM:SS limits minutes.
  Visual check by the user pending.

## Phase 1: Template infrastructure (pure refactor)

- `uv add jinja2` (already installed transitively; declare it).
- `portal/templates/` (`base.html`, one template per page) and
  `portal/static/portal.css`, `portal/static/*.js`; mount `/static`.
- Port every page 1:1 (login, index, job, log, annotate, gaze-label) with no
  visual change. CSRF token passed as template variable, not string replace.
- Tests: existing `tests/test_portal_*.py` pass unchanged; add a test that every
  route renders with autoescaping on.

## Phase 2: Design system and home page

- CSS tokens (color, spacing, type scale), system font stack, light theme;
  status/verdict chips; buttons; cards; tables.
- Home = "Sessions" list built from `OUTPUT_ROOT` (survives restarts): session,
  cameras, date, overall verdict, link. Running jobs pinned on top with a live
  status chip. Keep purging upload/job scratch on restart.
- "New session" as the single primary button.

## Phase 3: New-session form

- Step layout: 1 Session ID -> 2 Videos (drop zones; read duration in-browser
  from `<video>` metadata) -> 3 Tasks per camera -> 4 Extra-person windows ->
  5 Options (gaze) -> Review & start.
- Tasks and exclusions as structured rows (task name with suggestions
  grocery / storybook / synchrony, start, end) with live checks: end > start,
  inside video length, no overlaps.
- Timeline preview per camera: task bars, exclusion hatching, resulting
  `_partN` labels, and a note that parts are rejoined in the task summary.
- "Copy camera A tasks to camera B" helper (clocks differ, so it pre-fills
  for editing, never silently).
- Still posts the existing text format, so the backend contract does not change.
- Validation from the bug hunt (finding 9): reject `13:06:00`-style hours when
  minutes were meant (confirm with the user), negative, `nan`/`inf`, minutes or
  seconds >= 60, duplicate or colliding block names (`story_part1` vs generated
  parts), parts shorter than 10 s, blocks wholly inside an exclusion (say so),
  and times past the video length (browser-read duration; ffprobe re-check on
  the server after upload).
- Session ID reuse (finding 5): warn when `output/<session_id>` exists and ask
  to pick a new ID or confirm replacing (archive, never mix).
- Tests: rendering; server-side validation unchanged; JS kept small enough to
  review by hand.

## Phase 4: Job progress page

- Stepper: Upload -> Prepare video -> Find people -> Label parent (you) ->
  Track and refine -> Measure -> Report. Map internal step names to these in one
  table in `jobs.py`.
- Plain-language current step, elapsed time, per-camera progress.
- `GET /jobs/{id}/status.json`; poll with `fetch` instead of full reload.
- Failure state: one-sentence explanation naming the step + "show technical
  log" disclosure; never a raw command line (finding 10).
- Accept parent labels only while the job is waiting for them; otherwise say
  so (finding 8: submits relaunched phase 2 on finished/running jobs).
- An overlay clip with nobody tracked must not fail the job: render the raw
  frames or skip that clip with a note (finding 4).

## Phase 5: Parent labeling page

- Click the person in the image (boxes overlaid from `bbox_xyxy`), or keys
  `1`/`2` and `S` to skip; clear selected state; "8 of 12 labeled" counter;
  next-unlabeled auto-scroll.
- Confirm dialog only if fewer than the minimum anchors are labeled.
- Test: payload shape unchanged (`parent_track_id`, `child_track_id`,
  `skipped`).

## Phase 6: Results page

- Header: session, cameras, overall verdict, Download results.
- Per camera x task cards: verdict chip (`usable` / `check` / `insufficient`),
  plain reasons, valid time ("3 min 12 s with both people visible"), key
  numbers, and the overlay clips playable inline (new authenticated mp4 route
  with range requests).
- Task-level summary (from `task_metrics.csv`) above the per-part detail.
- Guidance box: what to include, what to watch in clips. Synchrony shown only
  with its chance level.
- Fallback to `coverage_both` when contract columns are missing.

## Phase 7: Gaze labeling and report.html

- Restyle gaze labeling with the shared CSS and the same interaction pattern as
  Phase 5.
- `report.html` reuses the same tokens (inlined, since it must work offline from
  the zip) and the same verdict cards.

## Open Questions

- Portal name/branding for the header (currently `PORTAL_DISPLAY_NAME`).
- Should research assistants be able to delete or rename sessions from the
  portal, or is that admin-only?
- Dark mode: needed, or light only?

## Test Strategy

pytest with FastAPI `TestClient` for every route (status, key text, escaping);
existing portal tests stay green after Phase 1. Visual review by the user at the
end of Phases 2, 3, 5 and 6 (screenshots via the local portal).
