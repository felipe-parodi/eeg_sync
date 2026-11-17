"""Automatic detection of the red sync light (IR-blaster LED) in video.

Protocol (as seen in the session videos): before the first task an
experimenter holds the IR-blaster board, a green PCB with one red LED and a
red lead, a few centimetres in front of camera A (the "SHORT" GoPro), usually
with a hand behind it, for about 10-30 s. Pressing the blaster puts an IR pulse
into the EEG stream; the sync point in the video is the frame where the red LED
lights. ``sync_pipeline.sync_eeg_to_video`` asks a person to find that frame by
hand; this module finds it automatically.

Two detectors:

* :func:`find_presentation_windows` finds the *hold*: a close-up of at least
  ``min_hold_s`` (most of the frame differs from the camera's usual view) in
  which an LED-shaped red blob is seen. It works on the 8-10 fps portal
  uploads.
* :func:`detect_red_light_events` finds LED *onsets* inside holds: the LED
  box brightens sharply relative to the preceding second. Lit LEDs saturate
  red and lift green/blue (pink-white core); the unlit red body stays dark in
  green/blue. A short flash is usually dropped when a
  video is decimated to 8-10 fps, so run this on the full-rate original.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .util import _require_cv2, format_time, parse_timestamp

try:
    import cv2
except ImportError:  # pragma: no cover - checked by _require_cv2
    cv2 = None

# Red-pixel mask: red well above green and blue relative to its own level.
# The ratio keeps skin (R-max(G,B) ~ 0.5 R under the warm room light) out,
# while the LED body and nail polish (~0.65 R) pass.
RED_MIN = 110
RED_RATIO_MIN = 0.58
# A lit LED is pink-white: red near saturation, still well above green/blue.
LIT_PIXEL_RED_MIN = 230
LIT_PIXEL_MARGIN_MIN = 50
# LED blob size as a fraction of the frame area. Red clothing and furniture
# are larger or (for small toys) far from the lens.
MIN_AREA_FRAC = 2e-4
MAX_AREA_FRAC = 3e-2
# A hold is a close-up (most of the view covered) in which the LED is seen at
# least HOLD_LED_MIN_SAMPLES times. The board is often tilted or half out of
# view, so the LED shows up in only some samples; across 22 hand-checked
# close-ups the LED was seen in 9/11 holds and 0/11 people passing the lens.
HOLD_LED_MIN_AREA_FRAC = 3e-4
HOLD_LED_MIN_SAMPLES = 2
# Share of pixels that differ from the camera's usual view during a hold
# (measured 0.47-0.73 on holds, <= 0.28 on people walking or bending nearby).
HOLD_COVERED_MIN = 0.35
COVER_DIFF_MIN = 90  # summed |BGR - background| for a pixel to count as covered
# Lit score = median of (R + min(G, B)) / 2 inside the LED box. A lit LED
# turns pink-white and leaves the red mask, so the box is carried over from
# the last frame where the LED body was found.
LIT_ABS_MIN = 150.0
LIT_RED_MIN = 200  # a lit LED saturates the red channel; walls and skin do not
CARRY_BOX_S = 0.15  # how long the LED box is kept after the LED is last seen
LIT_RISE_MIN = 40.0
BASELINE_S = 1.0
WORK_WIDTH = 320


@dataclass
class RedLightEvent:
    """One LED onset.

    Attributes:
        onset_s: Video time of the first lit frame, in seconds.
        frame: Index of the first lit frame in the source video.
        duration_s: How long the LED stayed lit.
        box: LED box (x1, y1, x2, y2) in source-video pixels.
        score: Rise of the lit score over the preceding baseline.
    """

    onset_s: float
    frame: int
    duration_s: float
    box: tuple[int, int, int, int]
    score: float


@dataclass
class _Candidate:
    lit: float
    area_frac: float
    box: tuple[int, int, int, int]


def _frame_candidates(frame: np.ndarray, scale: float = 1.0) -> list[_Candidate]:
    """Compact, LED-shaped red blobs in one BGR frame."""
    x = frame.astype(np.int16)
    b, g, r = x[..., 0], x[..., 1], x[..., 2]
    margin = r - np.maximum(g, b)
    red = ((margin > RED_RATIO_MIN * r) & (r > RED_MIN)) | (
        (r >= LIT_PIXEL_RED_MIN) & (margin >= LIT_PIXEL_MARGIN_MIN)
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
    frame_area = frame.shape[0] * frame.shape[1]
    out = []
    for j in range(1, n):
        bx, by, w, h, area = (int(v) for v in stats[j])
        frac = area / frame_area
        if not (MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC):
            continue
        if not (0.6 <= h / w <= 5.0) or area / (w * h) < 0.4:
            continue
        lit = _lit_score(frame, (bx, by, bx + w, by + h))
        box = tuple(int(round(v / scale)) for v in (bx, by, bx + w, by + h))
        out.append(_Candidate(lit=lit, area_frac=frac, box=box))
    return out


def _lit_score(frame: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Median of (R + min(G, B)) / 2 inside a box; 0 unless red is near saturation."""
    x1, y1, x2, y2 = box
    patch = frame[y1:y2, x1:x2].astype(np.int16)
    if patch.size == 0:
        return 0.0
    red = patch[..., 2]
    low = np.minimum(patch[..., 0], patch[..., 1])
    # White walls and skin are bright but not red-dominant.
    if np.median(red) < LIT_RED_MIN or np.median(red - low) < LIT_PIXEL_MARGIN_MIN:
        return 0.0
    return float(np.median((red + low) / 2.0))


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Whether two (x1, y1, x2, y2) boxes intersect."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _open(video_path: str | Path) -> tuple[object, float, int]:
    _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1 << 31
    return cap, fps, total


def _resize(frame: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, width / frame.shape[1])
    if scale < 1.0:
        frame = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    return frame, scale


def _background(
    cap: object, fps: float, total: int, start_s: float, end_s: float
) -> np.ndarray:
    """Median frame of the camera's usual view, sampled every few seconds."""
    lo = max(0, int((start_s - 600) * fps))
    hi = min(total, int((end_s + 600) * fps))
    picks = np.linspace(
        lo, max(lo, hi - 1), num=min(150, max(1, (hi - lo) // max(1, int(fps))))
    )
    frames = []
    for idx in picks.astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(_resize(frame, WORK_WIDTH)[0])
    return np.median(np.stack(frames), axis=0).astype(np.int16)


def find_presentation_windows(
    video_path: str | Path,
    start_s: float | None = None,
    end_s: float | None = None,
    min_hold_s: float = 3.0,
    max_gap_s: float = 1.0,
    step_s: float = 0.2,
) -> list[tuple[float, float]]:
    """Find periods where the LED board is held in front of the camera.

    Args:
        video_path: Video file (works on low-fps portal uploads).
        start_s: Only search from this video time (seconds).
        end_s: Only search up to this video time (seconds).
        min_hold_s: Shortest hold to report.
        max_gap_s: Bridge detection drop-outs up to this long.
        step_s: Sampling interval.

    Returns:
        (start_s, end_s) windows where the LED was seen, the window with the
        most LED sightings first.
    """
    cap, fps, total = _open(video_path)
    start_s = start_s or 0.0
    end_s = min(end_s if end_s is not None else total / fps, total / fps)
    bg = _background(cap, fps, total, start_s, end_s)
    samples: list[tuple[float, bool]] = []  # (time, LED seen) for covered samples
    t = start_s
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    stride = max(1, int(round(step_s * fps)))
    idx = int(round(t * fps))
    while idx / fps <= end_s:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - int(round(start_s * fps))) % stride == 0:
            small, _ = _resize(frame, WORK_WIDTH)
            covered = (
                np.abs(small.astype(np.int16) - bg).sum(-1) > COVER_DIFF_MIN
            ).mean()
            if covered >= HOLD_COVERED_MIN:
                led = max((c.area_frac for c in _frame_candidates(small)), default=0.0)
                samples.append((idx / fps, led >= HOLD_LED_MIN_AREA_FRAC))
        idx += 1
    cap.release()
    runs: list[tuple[float, float, list[float]]] = []  # close-ups + LED sighting times
    for t, led_seen in samples:
        if runs and t - runs[-1][1] <= max_gap_s:
            runs[-1] = (runs[-1][0], t, runs[-1][2])
        else:
            runs.append((t, t, []))
        if led_seen:
            runs[-1][2].append(t)
    holds = [
        run[2]
        for run in runs
        if run[1] - run[0] >= min_hold_s and len(run[2]) >= HOLD_LED_MIN_SAMPLES
    ]
    # Report the span where the LED was seen (+- 1 sample), most sightings first.
    holds.sort(key=len, reverse=True)
    return [
        (max(start_s, led[0] - step_s), min(end_s, led[-1] + step_s)) for led in holds
    ]


def _onsets_in_range(
    cap: object, fps: float, first: int, last: int, min_gap_s: float
) -> list[RedLightEvent]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    lit, boxes, indices, dim_boxes = [], [], [], []
    led_box: tuple[int, int, int, int] | None = None
    seen = first
    for idx in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        small, scale = _resize(frame, 2 * WORK_WIDTH)
        cands = _frame_candidates(small, scale)
        dim = [c for c in cands if c.lit == 0.0]
        dim_boxes.append(max(dim, key=lambda c: c.area_frac).box if dim else None)
        if cands:
            # The LED is the largest compact red blob this close to the lens.
            led_box = max(cands, key=lambda c: c.area_frac).box
            seen = idx
        elif led_box is not None and idx - seen > CARRY_BOX_S * fps:
            led_box = None
        if led_box is None:
            lit.append(0.0)
            boxes.append((0, 0, 0, 0))
        else:
            small_box = tuple(int(round(v * scale)) for v in led_box)
            lit.append(_lit_score(small, small_box))
            boxes.append(led_box)
        indices.append(idx)
    lit_arr = np.asarray(lit)
    base_n = max(1, int(round(BASELINE_S * fps)))
    events: list[RedLightEvent] = []
    i = 0
    while i < len(lit_arr):
        baseline = float(np.median(lit_arr[max(0, i - base_n) : i])) if i else 0.0
        rise = lit_arr[i] - baseline
        # An onset is the unlit LED body, seen just before at the same spot,
        # turning bright; red things entering the view already bright are not.
        before = dim_boxes[max(0, i - base_n // 2) : i]
        was_dim = any(b is not None and _overlap(b, boxes[i]) for b in before)
        if lit_arr[i] >= LIT_ABS_MIN and rise >= LIT_RISE_MIN and was_dim:
            keep = max(LIT_ABS_MIN, baseline + LIT_RISE_MIN / 2)
            j = i
            while j + 1 < len(lit_arr) and lit_arr[j + 1] >= keep:
                j += 1
            onset = indices[i] / fps
            if (
                events
                and onset - (events[-1].onset_s + events[-1].duration_s) < min_gap_s
            ):
                events[-1].duration_s = (indices[j] + 1) / fps - events[-1].onset_s
            else:
                events.append(
                    RedLightEvent(
                        onset_s=onset,
                        frame=indices[i],
                        duration_s=(j - i + 1) / fps,
                        box=boxes[i],
                        score=float(rise),
                    )
                )
            i = j + 1
            continue
        i += 1
    return events


def detect_red_light_events(
    video_path: str | Path,
    start_s: float | None = None,
    end_s: float | None = None,
    min_gap_s: float = 0.5,
    holds: Sequence[tuple[float, float]] | None = None,
    margin_s: float = 1.0,
) -> list[RedLightEvent]:
    """Find frames where the red sync LED switches on.

    Only the board holds are searched (see :func:`find_presentation_windows`),
    which keeps red clothing and toys elsewhere in the session from counting.

    Args:
        video_path: Video file (ideally the full-rate original).
        start_s: Only search from this video time (seconds).
        end_s: Only search up to this video time (seconds).
        min_gap_s: Flashes closer than this are merged into one event.
        holds: Hold windows to search; found automatically when omitted.
        margin_s: Also search this long before and after each hold.

    Returns:
        LED onsets in time order.
    """
    if holds is None:
        holds = find_presentation_windows(video_path, start_s, end_s)
    cap, fps, total = _open(video_path)
    events: list[RedLightEvent] = []
    for hold_start, hold_end in sorted(holds):
        lo = max(hold_start - margin_s, start_s or 0.0)
        hi = min(hold_end + margin_s, end_s if end_s is not None else total / fps)
        first, last = int(round(lo * fps)), min(total, int(round(hi * fps)) + 1)
        events.extend(_onsets_in_range(cap, fps, first, last, min_gap_s))
    cap.release()
    return sorted(events, key=lambda e: e.onset_s)


def camera_offset_from_events(
    events_a: Sequence[RedLightEvent],
    events_b: Sequence[RedLightEvent],
    tolerance_s: float = 0.5,
) -> float:
    """Offset between two cameras from LED onsets seen by both.

    Events are matched in order. Convention as in ``sync_results.json``:
    ``time_a = time_b + offset``.

    Args:
        events_a: Onsets in camera A.
        events_b: Onsets in camera B.
        tolerance_s: Allowed disagreement between matched pairs.

    Returns:
        Median offset in seconds.

    Raises:
        ValueError: If there are no pairs or the pairs disagree.
    """
    n = min(len(events_a), len(events_b))
    if n == 0:
        raise ValueError("Both cameras need at least one red-light event.")
    diffs = np.array([events_a[k].onset_s - events_b[k].onset_s for k in range(n)])
    offset = float(np.median(diffs))
    if np.any(np.abs(diffs - offset) > tolerance_s):
        raise ValueError(
            "Red-light events do not line up between cameras "
            f"(offsets {diffs.round(3).tolist()})."
        )
    return offset


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: find the red sync light in one video and write JSON."""
    parser = argparse.ArgumentParser(
        description="Find the red sync light (IR-blaster LED) in a video."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--hint", help="Approximate time of the light (M:SS or seconds)."
    )
    parser.add_argument(
        "--window",
        type=float,
        default=120.0,
        help="Search +- this many seconds around --hint.",
    )
    parser.add_argument("--start", help="Search from this time (M:SS or seconds).")
    parser.add_argument("--end", help="Search up to this time (M:SS or seconds).")
    parser.add_argument("--out", type=Path, help="Write results to this JSON file.")
    args = parser.parse_args(argv)

    start_s = parse_timestamp(args.start) if args.start else None
    end_s = parse_timestamp(args.end) if args.end else None
    if args.hint:
        hint = parse_timestamp(args.hint)
        start_s, end_s = max(0.0, hint - args.window), hint + args.window

    holds = find_presentation_windows(args.video, start_s, end_s)
    events = detect_red_light_events(args.video, start_s, end_s, holds=holds)
    print(f"Light board held in front of the camera: {len(holds)} time(s)")
    for s, e in holds:
        print(f"  {format_time(s)} - {format_time(e)}")
    print(f"Red light switching on: {len(events)} time(s)")
    for ev in events:
        print(
            f"  {format_time(ev.onset_s)} (frame {ev.frame}, lit {ev.duration_s:.2f} s)"
        )
    if not events:
        print("  None found. On low-frame-rate copies a short flash is often missing;")
        print("  run this on the original GoPro file for an exact time.")
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "video": str(args.video),
                    "presentation_windows_s": holds,
                    "events": [asdict(ev) for ev in events],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
