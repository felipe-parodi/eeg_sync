"""Per-session metrics table and one-page HTML report.

Gathers the per-camera summaries written by the metrics steps
(``pose_metrics_summary.json``, ``gaze_metrics_summary.json``,
``two_person_summary.json``) into one tidy table with one row per camera and
analysis block, and renders a self-contained ``report.html`` next to it.
``build_all_sessions_table`` stacks every session's table so a whole study can
be loaded into stats software as a single file.

CLI:
    python -m video_analysis.session_report --session-dir video_inference/output/1010 \\
        --all-sessions-root video_inference/output
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SESSION_METRICS_CSV = "session_metrics.csv"
ALL_SESSIONS_CSV = "all_sessions_metrics.csv"
REPORT_HTML = "report.html"

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
        session_dir: ``video_inference/output/<session_id>``.

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


def build_all_sessions_table(output_root: Path) -> pd.DataFrame:
    """Stack every session's ``session_metrics.csv`` under ``output_root``.

    Args:
        output_root: Directory holding one folder per session.

    Returns:
        Combined table (empty if no session has metrics yet).
    """
    tables = [
        pd.read_csv(path, dtype={"session_id": str, "camera_id": str, "block": str})
        for path in sorted(Path(output_root).glob(f"*/{SESSION_METRICS_CSV}"))
    ]
    if not tables:
        return pd.DataFrame(columns=LEAD_COLUMNS)
    return pd.concat(tables, ignore_index=True, sort=False)


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    return f"{int(seconds // 60)}:{int(round(seconds % 60)):02d}"


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    return f"{float(value):.{digits}f}{suffix}"


def _coverage_cell(row: pd.Series) -> str:
    share = float(row.get("coverage_both", 0) or 0)
    low = bool(row.get("low_coverage", share < 0.5))
    color = "var(--bad)" if low else "var(--good)"
    flag = ' <span class="flag">Low coverage</span>' if low else ""
    return (
        f'<div class="bar"><span style="width:{share * 100:.0f}%;background:{color}">'
        f"</span></div>{share:.0%}{flag}"
    )


def render_report_html(session_dir: Path, rows: pd.DataFrame) -> str:
    """Render the one-page session report.

    Links (figures, overlay clips) are relative to the session folder, so the
    report works after the results zip is extracted.

    Args:
        session_dir: Session output folder.
        rows: Table from :func:`collect_session_rows`.

    Returns:
        HTML document text.
    """
    session_dir = Path(session_dir)
    sid = html.escape(session_dir.name)
    has_gaze = any(c.startswith("gaze_") for c in rows.columns)
    gaze_cols = [
        ("gaze_mutual_gaze_pct", "Looking at each other"),
        ("gaze_joint_attention_pct", "Joint attention"),
        ("gaze_parent_watching_pct", "Parent watching child"),
        ("gaze_child_watching_pct", "Child watching parent"),
    ]
    gaze_cols = [(c, label) for c, label in gaze_cols if c in rows.columns]

    head = (
        "<tr><th>Camera</th><th>Block</th><th>Time in video</th>"
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
            f"<td>{_fmt(row.get('xcorr_mean'), 2)}</td>"
            + "".join(f"<td>{_fmt(row.get(c))}</td>" for c, _ in gaze_cols)
            + "</tr>"
        )

    low = rows[rows.get("low_coverage", pd.Series(False, index=rows.index)) == True]  # noqa: E712
    if len(low):
        items = "".join(
            f"<li>{html.escape(str(r.camera_id))} / {html.escape(str(r.block))}: "
            f"both people tracked in only {float(r.coverage_both):.0%} of the block.</li>"
            for r in low.itertuples()
        )
        warning = (
            '<div class="warn"><strong>Check before using these blocks.</strong> '
            "Their numbers describe only the moments when both people were "
            f"visible, which may not represent the whole block.<ul>{items}</ul>"
            "Watch the overlay clips below to see why (someone off camera, "
            "hidden behind the other person, or missed by the tracker).</div>"
        )
    else:
        warning = (
            '<p class="ok">Both people were tracked in at least half of every '
            "block.</p>"
        )

    media = []
    for camera_dir in _camera_dirs(session_dir):
        cam = html.escape(camera_dir.name)
        clips = "".join(
            f'<li><a href="{cam}/{html.escape(p.name)}">{html.escape(p.stem)}</a></li>'
            for p in sorted(camera_dir.glob("overlay_*.mp4"))
        )
        figures = "".join(
            f'<figure><img src="{cam}/{name}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            for name, caption in FIGURES
            if (camera_dir / name).exists()
        )
        media.append(
            f"<h3>{cam}</h3><p>Overlay clips (skeletons drawn on the video):</p>"
            f"<ul>{clips or '<li>none</li>'}</ul>"
            f'<div class="figs">{figures}</div>'
        )

    gaze_note = (
        "<li><strong>Gaze columns</strong>: share of frames (with both people "
        "tracked) in each gaze category, estimated from head position and the "
        "image.</li>"
        if has_gaze
        else ""
    )
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{sid} session report</title>
<style>
:root {{ --fg:#1d2433; --muted:#5b6475; --bg:#ffffff; --line:#d9dee7;
  --good:#2b8a5e; --bad:#c2410c; --warnbg:#fff4e5; }}
body {{ font-family: system-ui, sans-serif; color:var(--fg); background:var(--bg);
  max-width: 1100px; margin: 0 auto; padding: 16px; line-height:1.45; }}
table {{ border-collapse: collapse; width:100%; font-size:14px; }}
th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left;
  vertical-align: middle; }}
.table-wrap {{ overflow-x:auto; }}
.bar {{ display:inline-block; width:80px; height:8px; background:var(--line);
  border-radius:4px; margin-right:6px; vertical-align:middle; }}
.bar span {{ display:block; height:100%; border-radius:4px; }}
.flag {{ color:var(--bad); font-weight:600; }}
.warn {{ background:var(--warnbg); border-left:4px solid var(--bad); padding:8px 12px; }}
.ok {{ color:var(--good); }}
.figs {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:12px; }}
figure {{ margin:0; }} img {{ width:100%; border:1px solid var(--line); }}
figcaption, .muted {{ color:var(--muted); font-size:13px; }}
</style></head><body>
<h1>Session {sid}</h1>
<p class="muted">Generated {generated}. The same numbers are in
<code>{SESSION_METRICS_CSV}</code> (one row per camera and block).</p>
<h2>How much to trust each block</h2>
{warning}
<h2>Results per block</h2>
<div class="table-wrap"><table>{head}{"".join(body)}</table></div>
<ul class="muted">
<li><strong>Both people tracked</strong>: share of the block's frames in which
both the parent and the child were found. Every other number uses only those
frames.</li>
<li><strong>Mean distance</strong>: average distance between the parent's and
the child's torso centers, in image pixels (not comparable across cameras).</li>
<li><strong>Movement synchrony</strong>: how closely the two people's movements
follow each other (0 = unrelated, 1 = identical), in 5-second windows.</li>
{gaze_note}
</ul>
<h2>Clips and figures</h2>
{"".join(media)}
</body></html>
"""


def write_session_outputs(session_dir: Path) -> pd.DataFrame:
    """Write ``session_metrics.csv`` and ``report.html`` for one session.

    Args:
        session_dir: Session output folder.

    Returns:
        The session table.
    """
    session_dir = Path(session_dir)
    rows = collect_session_rows(session_dir)
    rows.to_csv(session_dir / SESSION_METRICS_CSV, index=False)
    (session_dir / REPORT_HTML).write_text(
        render_report_html(session_dir, rows), encoding="utf-8"
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


if __name__ == "__main__":
    main()
