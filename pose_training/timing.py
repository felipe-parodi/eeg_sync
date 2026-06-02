"""Load per-session analysis windows and multi-person exclusions.

The ``Timing_20 Participants.xlsx`` workbook (one row per participant x camera)
tells us, for every session, the two task windows worth analysing -- Grocery and
Storybook -- and, in the free-text NOTES, the spans where a third person is
visible. Frame sampling for the training set should stay inside the task windows
and outside the exclusion spans, so the multi-teacher labels never see the noisy
3+-person sections.

Mapping to the rest of the pipeline:
    - Camera "SHORT" -> ``camera_a`` -> ``{pid}_FULL_SHORT_compressed.mp4``
    - Camera "TALL"  -> ``camera_b`` -> ``{pid}_FULL_TALL_compressed.mp4``

Window / exclusion rules (confirmed with the user):
    - Sample both task windows (Grocery and Storybook).
    - Exclusions are only the explicit ``Extra Person MM:SS-MM:SS`` ranges plus
      the 1018 EEG-headband re-fit gap (a manual override; the pair leaves the
      interaction area). Vague "possible background person ... outside the
      interaction area" notes carry no window and are not excluded.
    - A task window with a missing end time is recovered from a cut/stop time in
      the NOTES when present (e.g. "video stops recording at 23:04"); otherwise
      that single window is dropped and the other task window is kept.

Spreadsheet times are the compressed videos' wall-clock (verified: 1013 TALL is
23.07 min long and its note reads "stops recording at 23:04"), so the times here
can be used directly against the compressed mp4s.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = REPO_ROOT / "compressed_video" / "Timing_20 Participants.xlsx"
DEFAULT_VIDEO_DIR = REPO_ROOT / "compressed_video"
DEFAULT_SHEET = "AllParticipants"

TASK_COLUMNS = (
    ("grocery", "grocery_start", "grocery_end"),
    ("storybook", "storybook_start", "storybook_end"),
)
CAMERA_TO_ID = {"SHORT": "camera_a", "TALL": "camera_b"}

_MMSS = re.compile(r"(\d{1,2}):(\d{2})")
_RANGE = re.compile(r"(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})")

# Manual exclusions that are not encoded as "Extra Person" ranges. The 1018 pair
# left to re-fit the EEG headband mid-Grocery; those frames have 0-1 people, so
# they are noise for a 2-person set. Keyed by (participant_id, camera).
MANUAL_EXCLUSIONS: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {
    ("1018", "SHORT"): [("23:40", "31:06", "EEG headband re-fit gap")],
    ("1018", "TALL"): [("23:18", "31:05", "EEG headband re-fit gap")],
}


def parse_mmss(value: object) -> Optional[float]:
    """Return seconds for the first ``MM:SS`` found in ``value``, else None.

    Tolerates trailing prose so cells like ``"22:52 video ends"`` parse to the
    embedded time. Returns None for empty cells or cells with no time.
    """
    if value is None:
        return None
    match = _MMSS.search(str(value))
    if match is None:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


@dataclass(frozen=True)
class TimeWindow:
    """A named task window in compressed-video wall-clock seconds."""

    name: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Exclusion:
    """A span to drop because a third person is visible (or subjects absent)."""

    start_s: float
    end_s: float
    reason: str = ""


@dataclass
class SessionTiming:
    """All timing for one participant x camera session."""

    participant_id: str
    camera: str  # "SHORT" | "TALL"
    camera_id: str  # "camera_a" | "camera_b"
    video_path: Optional[Path]
    windows: List[TimeWindow] = field(default_factory=list)
    exclusions: List[Exclusion] = field(default_factory=list)
    notes: str = ""

    @property
    def session_id(self) -> str:
        """Stable id used for filenames and split grouping."""
        return f"{self.participant_id}_{self.camera}"

    @property
    def is_sampleable(self) -> bool:
        """True when there is a video on disk and at least one valid span."""
        return self.video_path is not None and bool(self.sampling_spans())

    def sampling_spans(self, min_len_s: float = 1.0) -> List[Tuple[float, float]]:
        """Task windows with exclusion spans removed (fragments >= min_len_s)."""
        spans = [(w.start_s, w.end_s) for w in self.windows]
        excl = [(e.start_s, e.end_s) for e in self.exclusions]
        return subtract_exclusions(spans, excl, min_len_s=min_len_s)

    def valid_seconds(self) -> float:
        """Total sampleable duration in seconds (windows minus exclusions)."""
        return sum(b - a for a, b in self.sampling_spans())


def subtract_exclusions(
    spans: Sequence[Tuple[float, float]],
    exclusions: Sequence[Tuple[float, float]],
    min_len_s: float = 1.0,
) -> List[Tuple[float, float]]:
    """Remove exclusion spans from ``spans``, fragmenting where they overlap.

    Args:
        spans: Task windows as ``(start_s, end_s)`` pairs.
        exclusions: Spans to remove as ``(start_s, end_s)`` pairs.
        min_len_s: Drop result fragments shorter than this.

    Returns:
        Sorted, non-overlapping ``(start_s, end_s)`` fragments.
    """
    result: List[Tuple[float, float]] = []
    for start, end in spans:
        fragments = [(start, end)]
        for ex_start, ex_end in exclusions:
            next_fragments: List[Tuple[float, float]] = []
            for frag_start, frag_end in fragments:
                if ex_end <= frag_start or ex_start >= frag_end:
                    next_fragments.append((frag_start, frag_end))
                    continue
                if frag_start < ex_start:
                    next_fragments.append((frag_start, ex_start))
                if ex_end < frag_end:
                    next_fragments.append((ex_end, frag_end))
            fragments = next_fragments
        result.extend(f for f in fragments if f[1] - f[0] >= min_len_s)
    return sorted(result)


def extra_person_exclusions(notes: str) -> List[Exclusion]:
    """Parse explicit ``Extra Person MM:SS-MM:SS`` ranges from a NOTES cell."""
    if "extra person" not in notes.lower():
        return []
    out: List[Exclusion] = []
    for start_text, end_text in _RANGE.findall(notes):
        start_s, end_s = parse_mmss(start_text), parse_mmss(end_text)
        if start_s is not None and end_s is not None and end_s > start_s:
            out.append(Exclusion(start_s, end_s, "extra person"))
    return out


def _build_windows(row: Dict[str, object]) -> List[TimeWindow]:
    """Build task windows for a row, recovering a missing end from the note."""
    notes = str(row.get("notes") or "")
    note_cut = parse_mmss(notes)  # first MM:SS in the note (the cut time)
    windows: List[TimeWindow] = []
    for name, start_key, end_key in TASK_COLUMNS:
        start_s = parse_mmss(row.get(start_key))
        end_s = parse_mmss(row.get(end_key))
        if start_s is None:
            continue
        if end_s is None:  # camera cut / "video ends": recover from the note
            end_s = note_cut
        if end_s is None or end_s <= start_s:
            continue
        windows.append(TimeWindow(name, start_s, end_s))
    return windows


def build_sessions(
    rows: Sequence[Dict[str, object]], video_dir: Path = DEFAULT_VIDEO_DIR
) -> List[SessionTiming]:
    """Turn raw spreadsheet rows into ``SessionTiming`` records.

    Rows with no task windows at all (e.g. "No camera footage") are skipped.
    A session with windows but no video on disk is returned with
    ``video_path=None`` so the mismatch is visible (``is_sampleable`` is False).
    """
    sessions: List[SessionTiming] = []
    for row in rows:
        pid = str(row.get("participant_id") or "").strip()
        camera = str(row.get("camera") or "").strip().upper()
        if not pid or camera not in CAMERA_TO_ID:
            continue
        windows = _build_windows(row)
        if not windows:
            continue
        notes = str(row.get("notes") or "")
        exclusions = extra_person_exclusions(notes)
        for start_text, end_text, reason in MANUAL_EXCLUSIONS.get((pid, camera), []):
            exclusions.append(
                Exclusion(parse_mmss(start_text), parse_mmss(end_text), reason)
            )
        video_path = video_dir / f"{pid}_FULL_{camera}_compressed.mp4"
        sessions.append(
            SessionTiming(
                participant_id=pid,
                camera=camera,
                camera_id=CAMERA_TO_ID[camera],
                video_path=video_path if video_path.is_file() else None,
                windows=windows,
                exclusions=exclusions,
                notes=notes,
            )
        )
    return sessions


def read_xlsx_rows(
    xlsx_path: Path = DEFAULT_XLSX, sheet: str = DEFAULT_SHEET
) -> List[Dict[str, object]]:
    """Read the timing workbook into row dicts (lazy openpyxl import)."""
    import openpyxl

    workbook = openpyxl.load_workbook(xlsx_path, data_only=True)
    worksheet = workbook[sheet]
    rows = list(worksheet.iter_rows(values_only=True))
    keys = (
        "participant_id",
        "camera",
        "grocery_start",
        "grocery_end",
        "storybook_start",
        "storybook_end",
        "notes",
    )
    out: List[Dict[str, object]] = []
    for values in rows[1:]:  # skip header
        out.append({key: values[index] for index, key in enumerate(keys)})
    return out


def load_session_timings(
    xlsx_path: Path = DEFAULT_XLSX, video_dir: Path = DEFAULT_VIDEO_DIR
) -> List[SessionTiming]:
    """Load and parse all sessions from the timing workbook."""
    return build_sessions(read_xlsx_rows(xlsx_path), video_dir=video_dir)


def _format_mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def write_summary_csv(sessions: Sequence[SessionTiming], out_path: Path) -> None:
    """Write a per-session summary (windows, exclusions, valid minutes)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "session_id",
                "participant_id",
                "camera",
                "camera_id",
                "has_video",
                "windows",
                "exclusions",
                "valid_minutes",
                "sampleable",
            ]
        )
        for session in sessions:
            windows = "; ".join(
                f"{w.name} {_format_mmss(w.start_s)}-{_format_mmss(w.end_s)}"
                for w in session.windows
            )
            exclusions = "; ".join(
                f"{_format_mmss(e.start_s)}-{_format_mmss(e.end_s)} ({e.reason})"
                for e in session.exclusions
            )
            writer.writerow(
                [
                    session.session_id,
                    session.participant_id,
                    session.camera,
                    session.camera_id,
                    session.video_path is not None,
                    windows,
                    exclusions,
                    f"{session.valid_seconds() / 60:.2f}",
                    session.is_sampleable,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize session timing windows.")
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX), type=str)
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), type=str)
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "pose_training" / "data" / "session_windows.csv"),
        type=str,
    )
    args = parser.parse_args()
    sessions = load_session_timings(Path(args.xlsx), Path(args.video_dir))
    sampleable = [s for s in sessions if s.is_sampleable]
    no_video = [s for s in sessions if s.video_path is None]
    total_minutes = sum(s.valid_seconds() for s in sampleable) / 60
    print(f"Sessions with windows: {len(sessions)}")
    print(f"  sampleable (video + valid spans): {len(sampleable)}")
    print(f"  windows but no video on disk: {len(no_video)}")
    if no_video:
        print("   ", ", ".join(s.session_id for s in no_video))
    print(f"Total sampleable footage: {total_minutes:.1f} min")
    write_summary_csv(sessions, Path(args.csv))
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
