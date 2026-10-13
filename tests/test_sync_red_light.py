"""Tests for automatic red-light (IR-blaster LED) detection in video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sync_eeg_vid.red_light import (
    RedLightEvent,
    camera_offset_from_events,
    detect_red_light_events,
    find_presentation_windows,
)

FPS = 30.0
W, H = 320, 240
LED_DIM = (40, 40, 150)  # BGR: red plastic, LED off
LED_LIT = (170, 170, 255)  # BGR: overexposed core, LED on
HAND = (110, 140, 210)  # BGR: skin behind the board, close to the lens


def _write_video(
    path: Path,
    n_frames: int,
    led_on: set[int],
    hold: set[int],
    red_shirt: set[int] | None = None,
) -> Path:
    """Room view; during ``hold`` a hand covers the lens with the LED board."""
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(n_frames):
        frame = np.full((H, W, 3), (170, 180, 185), np.uint8)
        cv2.rectangle(frame, (0, 160), (W, H), (90, 110, 120), -1)  # floor
        frame = np.clip(frame + rng.integers(-6, 6, frame.shape), 0, 255).astype(
            np.uint8
        )
        if red_shirt and i in red_shirt:
            cv2.rectangle(frame, (10, 120), (130, 235), (50, 40, 160), -1)
        if i in hold:
            cv2.rectangle(frame, (60, 10), (270, 235), HAND, -1)
            x, y = 150, 100
            cv2.rectangle(frame, (x - 40, y + 16), (x + 40, y + 24), (40, 160, 40), -1)
            color = LED_LIT if i in led_on else LED_DIM
            cv2.rectangle(frame, (x - 6, y - 12), (x + 6, y + 15), color, -1)
        writer.write(frame)
    writer.release()
    return path


def test_presentation_window_spans_the_device_hold(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "a.avi", 450, set(), set(range(150, 300)))

    windows = find_presentation_windows(video, min_hold_s=2.0)

    assert len(windows) == 1
    start_s, end_s = windows[0]
    assert start_s == pytest.approx(150 / FPS, abs=0.25)
    assert end_s == pytest.approx(299 / FPS, abs=0.25)


def test_red_clothing_is_not_a_hold(tmp_path: Path) -> None:
    video = _write_video(
        tmp_path / "a.avi", 300, set(), set(), red_shirt=set(range(300))
    )

    assert find_presentation_windows(video) == []


def test_detects_led_turning_on_at_the_right_frame(tmp_path: Path) -> None:
    video = _write_video(
        tmp_path / "a.avi", 450, set(range(210, 220)), set(range(150, 300))
    )

    events = detect_red_light_events(video)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, RedLightEvent)
    assert event.frame == 210
    assert event.onset_s == pytest.approx(210 / FPS, abs=1e-6)
    assert event.duration_s == pytest.approx(10 / FPS, abs=1.5 / FPS)
    x1, y1, x2, y2 = event.box
    assert x1 <= 150 <= x2 and y1 <= 100 <= y2


def test_counts_each_flash_separately(tmp_path: Path) -> None:
    on = set(range(180, 186)) | set(range(240, 246))
    video = _write_video(tmp_path / "a.avi", 450, on, set(range(150, 300)))

    events = detect_red_light_events(video)

    assert [e.frame for e in events] == [180, 240]


def test_dim_led_during_hold_is_not_an_event(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "a.avi", 450, set(), set(range(150, 300)))

    assert detect_red_light_events(video) == []


def test_search_window_limits_where_events_are_found(tmp_path: Path) -> None:
    on = set(range(180, 186)) | set(range(300, 306))
    video = _write_video(tmp_path / "a.avi", 540, on, set(range(150, 360)))

    events = detect_red_light_events(video, start_s=8.0, end_s=15.0)

    assert [e.frame for e in events] == [300]


def _ev(t: float) -> RedLightEvent:
    return RedLightEvent(onset_s=t, frame=0, duration_s=0.2, box=(0, 0, 1, 1), score=1)


def test_camera_offset_from_matched_events() -> None:
    offset = camera_offset_from_events(
        [_ev(100.0), _ev(160.0)], [_ev(62.5), _ev(122.5)]
    )

    # time_a = time_b + offset
    assert offset == pytest.approx(37.5)


def test_camera_offset_rejects_inconsistent_events() -> None:
    with pytest.raises(ValueError):
        camera_offset_from_events([_ev(100.0), _ev(160.0)], [_ev(62.5), _ev(140.0)])
