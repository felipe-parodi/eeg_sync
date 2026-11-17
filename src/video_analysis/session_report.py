"""Per-session metrics table and one-page HTML report.

Gathers the per-camera summaries written by the metrics steps
(``pose_metrics_summary.json``, ``gaze_metrics_summary.json``,
``two_person_summary.json``) into one tidy table with one row per camera and
analysis block, and renders a self-contained ``report.html`` next to it.
``build_all_sessions_table`` stacks every session's table so a whole study can
be loaded into stats software as a single file.

CLI:
    python -m video_analysis.session_report --session-dir data/output/1010 \\
        --all-sessions-root data/output
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from video_analysis.pose_metrics import MIN_SYNC_WINDOWS, quality_verdict

SESSION_METRICS_CSV = "session_metrics.csv"
ALL_SESSIONS_CSV = "all_sessions_metrics.csv"
REPORT_HTML = "report.html"
TASK_METRICS_CSV = "task_metrics.csv"
# Bump when metric definitions change, so study tables never silently mix
# sessions processed under different definitions. 2026-11-16: 5 fps even frame
# timing, short-gap-only filling, whole-body synchrony, task table, verdict.
METRICS_VERSION = "2026-11-16"
LEGACY_METRICS_VERSION = "before-2026-11-16"
ALL_SESSIONS_TASK_CSV = "all_sessions_task_metrics.csv"

# Leading columns, in order; any other metric columns follow.
LEAD_COLUMNS = [
    "session_id",
    "camera_id",
    "block",
    "block_start_s",
    "block_end_s",
    "block_frames",
    "coverage_both",
    "low_coverage",
]

FIGURES = [
    ("pose_proximity.png", "Distance between parent and child over time"),
    ("pose_synchrony.png", "Movement synchrony over time"),
    ("synchrony_dashboard.png", "Gaze and synchrony dashboard"),
    ("gaze_categories.png", "Gaze categories per block"),
    ("gaze_convergence.png", "Gaze convergence per block"),
]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _camera_dirs(session_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in session_dir.glob("camera_*")
        if p.is_dir() and (p / "pose_metrics_summary.json").exists()
    )


def collect_session_rows(session_dir: Path) -> pd.DataFrame:
    """Build the tidy metrics table for one session.

    Args:
        session_dir: ``data/output/<session_id>``.

    Returns:
        One row per camera and analysis block. Pose metrics keep their names,
        gaze metrics are prefixed ``gaze_`` and whole-camera tracking stats
        ``tracking_``.
    """
    session_dir = Path(session_dir)
    rows: list[dict[str, Any]] = []
    for camera_dir in _camera_dirs(session_dir):
        pose_blocks = _read_json(camera_dir / "pose_metrics_summary.json") or []
        gaze_blocks = {
            item["block"]: item
            for item in _read_json(camera_dir / "gaze_metrics_summary.json") or []
        }
        tracking = _read_json(camera_dir / "two_person_summary.json") or {}
        for block_stats in pose_blocks:
            row: dict[str, Any] = {
                "session_id": session_dir.name,
                "camera_id": camera_dir.name,
            }
            row.update(block_stats)
            for key, value in gaze_blocks.get(block_stats["block"], {}).items():
                if key != "block":
                    row[f"gaze_{key}"] = value
            for key, value in tracking.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[f"tracking_{key}"] = value
            rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    lead = [c for c in LEAD_COLUMNS if c in table.columns]
    return table[lead + [c for c in table.columns if c not in lead]]


_PART_SUFFIX = re.compile(r"_part\d+$")


def task_name(block: str) -> str:
    """Task a block belongs to: the block name without a ``_partN`` suffix."""
    return _PART_SUFFIX.sub("", str(block))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    keep = values.notna() & weights.notna() & (weights > 0)
    if not keep.any():
        return float("nan")
    return float((values[keep] * weights[keep]).sum() / weights[keep].sum())


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"block": str}) if path.exists() else pd.DataFrame()


def collect_task_rows(session_dir: Path) -> pd.DataFrame:
    """One row per camera and task, rejoining blocks split by extra-person windows.

    Parts ``<task>_part1``, ``<task>_part2``, ... are pooled, not averaged:

    - distance: mean, median and sd recomputed from the per-frame file
      (``pose_metrics_proximity.csv``), so a median is never an average of
      part medians;
    - movement synchrony: the 20 s windows of all parts pooled
      (``pose_metrics_sync.csv``); the chance level is the parts' chance
      levels weighted by their window counts; windows never cross the gaps
      between parts;
    - gaze shares, their chance levels and convergence: weighted by each
      part's gaze frame count;
    - coverage: frame counts summed.

    Args:
        session_dir: ``data/output/<session_id>``.

    Returns:
        Task table (empty if the session has no metrics).
    """
    session_dir = Path(session_dir)
    blocks = collect_session_rows(session_dir)
    if blocks.empty:
        return pd.DataFrame()
    blocks = blocks.assign(task=blocks["block"].map(task_name))
    rows: list[dict[str, Any]] = []
    for (camera_id, task), parts in blocks.groupby(["camera_id", "task"], sort=False):
        camera_dir = session_dir / str(camera_id)
        names = list(parts["block"])
        row: dict[str, Any] = {
            "session_id": session_dir.name,
            "camera_id": camera_id,
            "task": task,
            "n_parts": len(names),
            "parts": ";".join(names),
            "task_start_s": float(parts["block_start_s"].min()),
            "task_end_s": float(parts["block_end_s"].max()),
            "duration_s": float((parts["block_end_s"] - parts["block_start_s"]).sum()),
        }
        for col in ("block_frames", "frames_with_both", "n_segments"):
            if col in parts:
                row[col] = int(parts[col].fillna(0).sum())
        if row.get("block_frames"):
            row["coverage_both"] = row.get("frames_with_both", 0) / row["block_frames"]
        if "valid_seconds_both" in parts:
            row["valid_seconds_both"] = float(
                parts["valid_seconds_both"].fillna(0).sum()
            )
        if "longest_segment_s" in parts:
            row["longest_segment_s"] = float(parts["longest_segment_s"].max())

        prox = _read_csv(camera_dir / "pose_metrics_proximity.csv")
        if not prox.empty and "block" in prox:
            dist = prox.loc[prox["block"].isin(names), "torso_distance_px"]
            if len(dist):
                row["proximity_mean"] = float(dist.mean())
                row["proximity_median"] = float(dist.median())
                row["proximity_std"] = float(dist.std())
                row["proximity_n_frames"] = int(len(dist))

        sync = _read_csv(camera_dir / "pose_metrics_sync.csv")
        if not sync.empty and "block" in sync:
            windows = sync.loc[sync["block"].isin(names), "sync_r"]
            if len(windows):
                row["sync_mean"] = float(windows.mean())
                row["sync_std"] = float(windows.std())
                row["sync_n_windows"] = int(len(windows))
                if {"sync_chance_mean", "sync_n_windows"} <= set(parts.columns):
                    chance = _weighted_mean(
                        parts["sync_chance_mean"], parts["sync_n_windows"]
                    )
                    row["sync_chance_mean"] = chance
                    row["sync_minus_chance"] = row["sync_mean"] - chance

        if "gaze_n_frames" in parts:
            weight = parts["gaze_n_frames"]
            row["gaze_n_frames"] = int(weight.fillna(0).sum())
            for col in parts.columns:
                if col.startswith("gaze_") and col.endswith("_pct"):
                    row[col] = _weighted_mean(parts[col], weight)
        if "gaze_convergence_n_frames" in parts:
            weight = parts["gaze_convergence_n_frames"]
            row["gaze_convergence_n_frames"] = int(weight.fillna(0).sum())
            for col in ("gaze_convergence_mean", "gaze_convergence_chance"):
                if col in parts:
                    row[col] = _weighted_mean(parts[col], weight)
        row["quality_verdict"], row["quality_reasons"] = quality_verdict(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _mark_legacy(table: pd.DataFrame) -> pd.DataFrame:
    """Label rows written before metrics versions existed."""
    if "metrics_version" not in table.columns:
        table["metrics_version"] = LEGACY_METRICS_VERSION
    table["metrics_version"] = table["metrics_version"].fillna(LEGACY_METRICS_VERSION)
    return table


def build_all_sessions_task_table(output_root: Path) -> pd.DataFrame:
    """Stack every session's ``task_metrics.csv`` under ``output_root``.

    Args:
        output_root: Directory holding one folder per session.

    Returns:
        Combined task table (empty if no session has one yet).
    """
    tables = [
        pd.read_csv(path, dtype={"session_id": str, "camera_id": str, "task": str})
        for path in sorted(Path(output_root).glob(f"*/{TASK_METRICS_CSV}"))
        if not path.parent.name.startswith("_")
    ]
    if not tables:
        return pd.DataFrame(columns=["session_id", "camera_id", "task"])
    return _mark_legacy(pd.concat(tables, ignore_index=True, sort=False))


def build_all_sessions_table(output_root: Path) -> pd.DataFrame:
    """Stack every session's ``session_metrics.csv`` under ``output_root``.

    Args:
        output_root: Directory holding one folder per session. Folders whose
            name starts with ``_`` (such as ``_archive``) are skipped.

    Returns:
        Combined table (empty if no session has metrics yet).
    """
    # Folders starting with "_" (the portal's ``_archive`` of replaced runs)
    # are not sessions.
    tables = [
        pd.read_csv(path, dtype={"session_id": str, "camera_id": str, "block": str})
        for path in sorted(Path(output_root).glob(f"*/{SESSION_METRICS_CSV}"))
        if not path.parent.name.startswith("_")
    ]
    if not tables:
        return pd.DataFrame(columns=LEAD_COLUMNS)
    return _mark_legacy(pd.concat(tables, ignore_index=True, sort=False))


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    return f"{int(seconds // 60)}:{int(round(seconds % 60)):02d}"


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    return f"{float(value):.{digits}f}{suffix}"


def _chance(row: pd.Series, column: str) -> str:
    chance = row.get(column.replace("_pct", "_chance_pct"))
    if chance is None or (isinstance(chance, float) and pd.isna(chance)):
        return ""
    if column.endswith("unknown_pct"):
        return ""
    return f' <span class="muted">(chance {float(chance):.0f})</span>'


def _sync_cell(row: pd.Series) -> str:
    windows = row.get("sync_n_windows")
    if windows is None or pd.isna(windows) or int(windows) < MIN_SYNC_WINDOWS:
        return (
            '<span class="muted">too few unbroken 20-second stretches '
            "to measure</span>"
        )
    return f"{_fmt(row.get('sync_mean'), 2)}{_sync_chance(row)}"


def _sync_chance(row: pd.Series) -> str:
    chance = row.get("sync_chance_mean")
    if chance is None or (isinstance(chance, float) and pd.isna(chance)):
        return ""
    return f' <span class="muted">(chance {float(chance):.2f})</span>'


def _coverage_cell(row: pd.Series) -> str:
    share = float(row.get("coverage_both", 0) or 0)
    low = bool(row.get("low_coverage", share < 0.5))
    color = "var(--bad)" if low else "var(--good)"
    flag = ' <span class="flag">Low coverage</span>' if low else ""
    return (
        f'<div class="bar"><span style="width:{share * 100:.0f}%;background:{color}">'
        f"</span></div>{share:.0%}{flag}"
    )


# Styling copied from portal/static/portal.css (same tokens), inlined because
# report.html must work offline from the results zip.
REPORT_CSS = """
:root { color-scheme: light;
  --color-bg:#f7f7f5; --color-surface:#ffffff; --color-surface-muted:#fafaf8;
  --color-border:#e2e4e8; --color-border-strong:#c6cad1;
  --color-text:#1b2430; --color-text-muted:#5a6472;
  --color-accent:#0072b2; --color-accent-strong:#005a8c;
  --color-ok-text:#00704f; --color-ok-bg:#e6f4ef; --color-ok-edge:#009e73;
  --color-warn-text:#8a5a00; --color-warn-bg:#fdf3da; --color-warn-edge:#e69f00;
  --color-bad-text:#a63f00; --color-bad-bg:#fbeae0; --color-bad-edge:#d55e00;
  --color-neutral-text:#4b5563; --color-neutral-bg:#f0f1f3; --color-neutral-edge:#9aa1ab;
  --good:var(--color-ok-edge); --bad:var(--color-bad-edge);
  --space-1:4px; --space-2:8px; --space-3:12px; --space-4:16px; --space-5:24px;
  --space-6:32px; --radius-sm:4px; --radius-md:8px; --radius-lg:12px;
  --font-sans:system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --text-xs:0.75rem; --text-sm:0.875rem; --text-lg:1.25rem; --text-xl:1.5rem;
  --text-2xl:1.875rem; }
*, *::before, *::after { box-sizing:border-box; }
body { margin:0 auto; max-width:1040px; padding:var(--space-6) var(--space-5);
  font-family:var(--font-sans); line-height:1.5; color:var(--color-text);
  background:var(--color-bg); -webkit-font-smoothing:antialiased; }
h1, h2, h3, h4 { line-height:1.25; margin:0 0 var(--space-3); }
h1 { font-size:var(--text-2xl); letter-spacing:-0.01em; }
h2 { font-size:var(--text-lg); }
h4 { font-size:var(--text-sm); margin:var(--space-4) 0 var(--space-2); }
p { margin:0 0 var(--space-3); }
a { color:var(--color-accent-strong); text-underline-offset:2px; }
code { font-size:0.9em; }
.card { background:var(--color-surface); border:1px solid var(--color-border);
  border-radius:var(--radius-lg); box-shadow:0 1px 2px rgba(27,36,48,.06);
  padding:var(--space-5); margin-bottom:var(--space-5); }
.muted, figcaption { color:var(--color-text-muted); font-size:var(--text-sm); }
.camera-tag { display:inline-block; padding:0 6px; border-radius:var(--radius-sm);
  background:var(--color-neutral-bg); color:var(--color-neutral-text);
  font-size:var(--text-xs); font-weight:700; margin-right:var(--space-1); }
.table-wrap { overflow-x:auto; margin-bottom:var(--space-3); }
table { width:100%; border-collapse:collapse; font-size:var(--text-sm);
  background:var(--color-surface); }
th, td { border-bottom:1px solid var(--color-border); padding:var(--space-2) var(--space-3);
  text-align:left; vertical-align:top; }
th { font-size:var(--text-xs); font-weight:600; text-transform:uppercase;
  letter-spacing:0.04em; color:var(--color-text-muted);
  background:var(--color-surface-muted); }
.chip { display:inline-flex; align-items:center; gap:6px; padding:2px 10px 2px 6px;
  border-radius:999px; border:1px solid; font-size:var(--text-xs); font-weight:600;
  line-height:1.6; white-space:nowrap; }
.chip__icon { display:inline-flex; align-items:center; justify-content:center;
  width:15px; height:15px; border-radius:50%; font-size:10px; line-height:1;
  font-weight:800; color:#fff; }
.chip--ok { color:var(--color-ok-text); background:var(--color-ok-bg); border-color:var(--color-ok-edge); }
.chip--ok .chip__icon { background:var(--color-ok-text); }
.chip--check { color:var(--color-warn-text); background:var(--color-warn-bg); border-color:var(--color-warn-edge); }
.chip--check .chip__icon { background:var(--color-warn-text); }
.chip--insufficient, .chip--unknown { color:var(--color-neutral-text);
  background:var(--color-neutral-bg); border-color:var(--color-neutral-edge); }
.chip--insufficient .chip__icon, .chip--unknown .chip__icon { background:var(--color-neutral-text); }
.reasons { margin-top:var(--space-1); color:var(--color-text-muted);
  font-size:var(--text-xs); max-width:28em; }
.task-table td[data-label="Data check"] { min-width:14em; }
.task-table td[data-label="Movement synchrony"] { min-width:11em; }
.task-table td[data-label="Task"], .task-table td[data-label="Usable time"] {
  white-space:nowrap; }
.bar { display:inline-block; width:64px; height:8px; background:var(--color-border);
  border-radius:4px; margin-right:6px; vertical-align:middle; }
.bar span { display:block; height:100%; border-radius:4px; }
.flag { color:var(--color-bad-text); font-weight:600; }
details.pieces > summary { cursor:pointer; font-weight:600; padding:var(--space-2) 0; }
details.pieces[open] > summary { margin-bottom:var(--space-3); }
.notes { padding-left:1.2em; }
.notes li { margin-bottom:var(--space-2); }
.clips { display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%, 260px), 1fr));
  gap:var(--space-3); }
figure { margin:0; }
.clip video { display:block; width:100%; aspect-ratio:16/9; background:#1b2430;
  border-radius:var(--radius-md); }
.figs { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 300px), 1fr));
  gap:var(--space-3); margin-top:var(--space-4); }
.figs img { width:100%; height:auto; border:1px solid var(--color-border);
  border-radius:var(--radius-md); }
@media (max-width:640px) {
  body { padding:var(--space-5) var(--space-4); }
  h1 { font-size:var(--text-xl); }
  .card { padding:var(--space-4); }
  /* The task table becomes one stacked card per row. */
  .task-table thead { display:none; }
  .task-table tr { display:block; padding:var(--space-3) 0;
    border-bottom:1px solid var(--color-border-strong); }
  .task-table td[data-label] { min-width:0; white-space:normal; }
  .task-table td { display:grid; grid-template-columns:8.5em minmax(0, 1fr);
    gap:var(--space-2); border:0; padding:2px 0; }
  .task-table td::before { content:attr(data-label); font-size:var(--text-xs);
    color:var(--color-text-muted); font-weight:600; }
}
"""

# quality_verdict value -> (chip kind, glyph, label). The glyphs match the
# portal's chips, so the verdict never relies on color alone.
VERDICT_CHIPS = {
    "usable": ("ok", "✓", "Usable"),
    "check": ("check", "!", "Check"),
    "insufficient": ("insufficient", "–", "Not enough data"),
}

GAZE_COLUMNS = [
    ("gaze_mutual_gaze_pct", "Looking at each other"),
    ("gaze_joint_attention_pct", "Joint attention"),
    ("gaze_parent_watching_pct", "Parent watching child"),
    ("gaze_child_watching_pct", "Child watching parent"),
    ("gaze_unknown_pct", "Gaze unknown"),
]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _fmt_duration(seconds: Any) -> str:
    """Format seconds as "3 min 12 s" ("45 s" under a minute)."""
    if _is_missing(seconds):
        return "–"
    total = int(round(float(seconds)))
    minutes, secs = divmod(total, 60)
    if not minutes:
        return f"{secs} s"
    return f"{minutes} min {secs} s" if secs else f"{minutes} min"


def _verdict_cell(row: pd.Series) -> str:
    """Verdict chip (text and symbol) with its reasons as plain sentences."""
    verdict = row.get("quality_verdict")
    kind, glyph, label = VERDICT_CHIPS.get(
        "" if _is_missing(verdict) else str(verdict), ("unknown", "?", "Not checked")
    )
    chip = (
        f'<span class="chip chip--{kind}"><span class="chip__icon" '
        f'aria-hidden="true">{glyph}</span>{label}</span>'
    )
    reasons = row.get("quality_reasons")
    if _is_missing(reasons) or not str(reasons).strip():
        return chip
    sentences = [r.strip() for r in str(reasons).split(";") if r.strip()]
    text = " ".join(f"{s[0].upper()}{s[1:]}." for s in sentences)
    return f'{chip}<div class="reasons">{html.escape(text)}</div>'


def _clip_groups(camera_dir: Path, blocks: list[str]) -> list[tuple[str, list[Path]]]:
    """Group a camera's overlay clips by block, in table order.

    ``overlay_<block>.mp4`` is the start of the block and
    ``overlay_<block>_end.mp4`` its end.
    """
    groups: dict[str, list[Path]] = {}
    for path in sorted(camera_dir.glob("overlay_*.mp4")):
        block = path.stem[len("overlay_") :]
        if block.endswith("_end"):
            block = block[: -len("_end")]
        groups.setdefault(block, []).append(path)
    order = [b for b in blocks if b in groups] + [b for b in groups if b not in blocks]
    return [
        (b, sorted(groups[b], key=lambda p: p.stem.endswith("_end"))) for b in order
    ]


def _clips_html(camera_dir: Path, blocks: list[str]) -> str:
    """Playable overlay clips for one camera, one group per block."""
    cam = html.escape(camera_dir.name)
    groups = []
    for block, paths in _clip_groups(camera_dir, blocks):
        figures = []
        for path in paths:
            src = f"{cam}/{html.escape(path.name)}"
            which = "End" if path.stem.endswith("_end") else "Start"
            figures.append(
                f'<figure class="clip"><video controls preload="none" src="{src}">'
                f"</video><figcaption>{which} · "
                f'<a href="{src}" download>Download</a></figcaption></figure>'
            )
        groups.append(
            f'<div class="clip-group" data-block="{html.escape(block)}">'
            f"<h4>{html.escape(block)}</h4>"
            f'<div class="clips">{"".join(figures)}</div></div>'
        )
    return "".join(groups) or '<p class="muted">No overlay clips.</p>'


def _task_table_html(tasks: pd.DataFrame) -> str:
    """One row per camera and task, with verdict, usable time and key numbers."""
    if tasks.empty:
        return '<p class="muted">No tasks were measured in this session.</p>'
    gaze_cols = [(c, label) for c, label in GAZE_COLUMNS if c in tasks.columns]
    labels = [
        "Camera",
        "Task",
        "Data check",
        "Usable time",
        "Both people tracked",
        "Distance (median px)",
        "Movement synchrony",
    ] + [f"{label} (%)" for _, label in gaze_cols]
    head = "".join(f'<th scope="col">{label}</th>' for label in labels)
    body = []
    for _, row in tasks.iterrows():
        n_parts = row.get("n_parts")
        pieces = (
            f'<div class="muted">{int(n_parts)} pieces joined</div>'
            if not _is_missing(n_parts) and int(n_parts) > 1
            else ""
        )
        span = ""
        if not _is_missing(row.get("task_start_s")):
            span = (
                f'<div class="muted">{_fmt_time(row["task_start_s"])}–'
                f"{_fmt_time(row['task_end_s'])}</div>"
            )
        cells = [
            html.escape(str(row["camera_id"])),
            f"<strong>{html.escape(str(row['task']))}</strong>{span}{pieces}",
            _verdict_cell(row),
            _fmt_duration(row.get("valid_seconds_both")),
            _coverage_cell(row),
            _fmt(row.get("proximity_median")),
            _sync_cell(row),
        ] + [f"{_fmt(row.get(c))}{_chance(row, c)}" for c, _ in gaze_cols]
        body.append(
            "<tr>"
            + "".join(
                f'<td data-label="{html.escape(label)}"><div>{cell}</div></td>'
                for label, cell in zip(labels, cells)
            )
            + "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="task-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def render_report_html(
    session_dir: Path, rows: pd.DataFrame, tasks: pd.DataFrame | None = None
) -> str:
    """Render the one-page session report (the portal's results page).

    The task table (one row per camera and task, pieces joined) comes first,
    then the per-piece table collapsed, then the overlay clips playable in the
    page and the figures. Links (figures, clips) are relative to the session
    folder and the styling is inlined, so the report works offline after the
    results zip is extracted.

    Args:
        session_dir: Session output folder.
        rows: Table from :func:`collect_session_rows`.
        tasks: Table from :func:`collect_task_rows`; built from
            ``session_dir`` when omitted.

    Returns:
        HTML document text.
    """
    session_dir = Path(session_dir)
    if tasks is None:
        tasks = collect_task_rows(session_dir)
    sid = html.escape(session_dir.name)
    has_gaze = any(c.startswith("gaze_") for c in rows.columns)
    gaze_cols = [(c, label) for c, label in GAZE_COLUMNS if c in rows.columns]
    camera_dirs = _camera_dirs(session_dir)
    cameras = (
        "".join(
            f'<span class="camera-tag">{html.escape(p.name)}</span>'
            for p in camera_dirs
        )
        or '<span class="muted">none</span>'
    )

    head = (
        "<tr><th>Camera</th><th>Piece</th><th>Time in video</th>"
        "<th>Both people tracked</th><th>Mean distance (px)</th>"
        "<th>Movement synchrony</th>"
        + "".join(f"<th>{label} (%)</th>" for _, label in gaze_cols)
        + "</tr>"
    )
    body = []
    for _, row in rows.iterrows():
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['camera_id']))}</td>"
            f"<td>{html.escape(str(row['block']))}</td>"
            f"<td>{_fmt_time(row.get('block_start_s', 0))}–"
            f"{_fmt_time(row.get('block_end_s', 0))}</td>"
            f"<td>{_coverage_cell(row)}</td>"
            f"<td>{_fmt(row.get('proximity_mean'))}</td>"
            f"<td>{_sync_cell(row)}</td>"
            + "".join(
                f"<td>{_fmt(row.get(c))}{_chance(row, c)}</td>" for c, _ in gaze_cols
            )
            + "</tr>"
        )

    media = []
    for camera_dir in camera_dirs:
        cam = html.escape(camera_dir.name)
        blocks = (
            [str(b) for b in rows.loc[rows["camera_id"] == camera_dir.name, "block"]]
            if {"camera_id", "block"} <= set(rows.columns)
            else []
        )
        figures = "".join(
            f'<figure><img src="{cam}/{name}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            for name, caption in FIGURES
            if (camera_dir / name).exists()
        )
        media.append(
            f'<section class="card"><h2>{cam}: clips and figures</h2>'
            '<p class="muted">Overlay clips show the video with skeletons drawn. '
            "Check that the boxes follow the parent and the child, not someone "
            "else or an object.</p>"
            f"{_clips_html(camera_dir, blocks)}"
            f'<div class="figs">{figures}</div></section>'
        )

    gaze_note = (
        "<li><strong>Gaze columns</strong>: share of frames (with both people "
        "tracked) in each gaze category, from where each person's estimated "
        "gaze lands relative to the other person. <em>Chance</em> is the same "
        "share with the child's gaze shifted 60 s in time: what seating "
        "position alone would produce. A category is meaningful only when it "
        "clearly exceeds its chance level. <em>Gaze unknown</em> = the model "
        "judged someone to be looking out of view.</li>"
        if has_gaze
        else ""
    )
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{sid} session report</title>
<style>{REPORT_CSS}</style></head><body>
<h1>Session {sid}</h1>
<p>Cameras: {cameras}</p>
<p class="muted">Generated {generated}. The same numbers are in
<code>{TASK_METRICS_CSV}</code> (one row per camera and task) and
<code>{SESSION_METRICS_CSV}</code> (one row per camera and piece).</p>
<section class="card">
<h2>Results per task</h2>
{_task_table_html(tasks)}
<p class="muted">When extra-person windows cut a task into pieces, the pieces
are joined here: distances and synchrony windows are pooled across them. The
data check says whether there is enough data behind the numbers and which
clips to watch. <strong>Recommended rule:</strong> for each task use the camera
with more usable time; include distance and gaze when both people are tracked
for at least 2 minutes, and movement synchrony only with at least 10 windows
(both together give "Usable"); exclude any task whose clips show the wrong
person or an object.</p>
<details class="pieces"><summary>Per piece ({len(rows)} rows)</summary>
<div class="table-wrap"><table>{head}{"".join(body)}</table></div>
</details>
<ul class="muted notes">
<li><strong>Usable time</strong>: time with both the parent and the child
found. <strong>Both people tracked</strong>: the same as a share of the task.
Every other number uses only those frames.</li>
<li><strong>Distance</strong>: distance between the parent's and the child's
torso centers, in image pixels (compare within a camera, not across
cameras).</li>
<li><strong>Movement synchrony</strong>: how closely the amount of whole-body
movement of the two people rises and falls together, as a correlation in
20-second windows (0 = unrelated, 1 = identical). <em>Chance</em> is the same
measure with the child's movement taken from other moments. Only the part
above chance reflects the interaction. It is not shown when there are fewer
than {MIN_SYNC_WINDOWS} unbroken 20-second stretches.</li>
{gaze_note}
</ul>
</section>
{"".join(media)}
</body></html>
"""


def write_session_outputs(session_dir: Path) -> pd.DataFrame:
    """Write ``session_metrics.csv``, ``task_metrics.csv`` and ``report.html``.

    Args:
        session_dir: Session output folder.

    Returns:
        The session table.
    """
    session_dir = Path(session_dir)
    rows = collect_session_rows(session_dir)
    rows = rows.assign(metrics_version=METRICS_VERSION)
    rows.to_csv(session_dir / SESSION_METRICS_CSV, index=False)
    tasks = collect_task_rows(session_dir).assign(metrics_version=METRICS_VERSION)
    tasks.to_csv(session_dir / TASK_METRICS_CSV, index=False)
    (session_dir / REPORT_HTML).write_text(
        render_report_html(session_dir, rows, tasks), encoding="utf-8"
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument(
        "--all-sessions-root",
        type=Path,
        default=None,
        help=f"Also rebuild <root>/{ALL_SESSIONS_CSV} from every session.",
    )
    args = parser.parse_args(argv)
    rows = write_session_outputs(args.session_dir)
    print(f"[session-report] {len(rows)} camera/block rows -> {args.session_dir}")
    if args.all_sessions_root is not None:
        table = build_all_sessions_table(args.all_sessions_root)
        out = args.all_sessions_root / ALL_SESSIONS_CSV
        table.to_csv(out, index=False)
        print(f"[session-report] {len(table)} rows across sessions -> {out}")
        tasks = build_all_sessions_task_table(args.all_sessions_root)
        tasks_out = args.all_sessions_root / ALL_SESSIONS_TASK_CSV
        tasks.to_csv(tasks_out, index=False)
        print(f"[session-report] {len(tasks)} task rows across sessions -> {tasks_out}")


if __name__ == "__main__":
    main()
