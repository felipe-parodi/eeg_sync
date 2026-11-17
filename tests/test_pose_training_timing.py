"""Tests for the session-timing loader (window + exclusion parsing)."""

from __future__ import annotations

from pose_training import timing


def test_parse_mmss_handles_embedded_and_empty() -> None:
    assert timing.parse_mmss("13:42") == 822
    assert timing.parse_mmss("22:52 video ends") == 1372  # trailing prose
    assert timing.parse_mmss("video ends") is None
    assert timing.parse_mmss(None) is None
    assert timing.parse_mmss("") is None


def test_subtract_exclusions_fragments_overlapping_window() -> None:
    # A single window with an exclusion in the middle splits into two fragments.
    spans = [(0.0, 100.0)]
    excl = [(40.0, 60.0)]
    assert timing.subtract_exclusions(spans, excl) == [(0.0, 40.0), (60.0, 100.0)]


def test_subtract_exclusions_drops_tiny_fragments() -> None:
    spans = [(0.0, 100.0)]
    excl = [(0.5, 100.0)]  # leaves only a 0.5s sliver at the start
    assert timing.subtract_exclusions(spans, excl, min_len_s=1.0) == []


def test_extra_person_only_with_keyword() -> None:
    assert timing.extra_person_exclusions("nothing here") == []
    out = timing.extra_person_exclusions("Extra Person 18:33-19:07 21:50-22:35")
    assert [(e.start_s, e.end_s) for e in out] == [(1113, 1147), (1310, 1355)]


def test_build_sessions_maps_camera_and_windows() -> None:
    rows = [
        {
            "participant_id": "P001",
            "camera": "Short",
            "grocery_start": "13:42",
            "grocery_end": "23:40",
            "storybook_start": "29:22",
            "storybook_end": "37:06",
            "notes": "",
        }
    ]
    sessions = timing.build_sessions(rows, video_dir=timing.DEFAULT_VIDEO_DIR)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.camera_id == "camera_a"  # SHORT -> A
    assert [w.name for w in s.windows] == ["grocery", "storybook"]
    assert s.windows[0].start_s == 822


def test_build_sessions_recovers_missing_end_from_note() -> None:
    rows = [
        {
            "participant_id": "1011",
            "camera": "short",
            "grocery_start": "17:26",
            "grocery_end": "27:44",
            "storybook_start": "32:03",
            "storybook_end": None,  # blank end
            "notes": "35:30 video cuts",
        }
    ]
    s = timing.build_sessions(rows)[0]
    story = [w for w in s.windows if w.name == "storybook"][0]
    assert story.end_s == 35 * 60 + 30  # recovered from the note


def test_build_sessions_skips_window_with_no_recoverable_end() -> None:
    # P007 TALL: storybook starts at the cutoff -> empty window, dropped.
    rows = [
        {
            "participant_id": "P007",
            "camera": "tall",
            "grocery_start": "10:22",
            "grocery_end": "21:33",
            "storybook_start": "26:35",
            "storybook_end": None,
            "notes": "Camera cut off at 26:35",
        }
    ]
    s = timing.build_sessions(rows)[0]
    assert [w.name for w in s.windows] == ["grocery"]  # storybook dropped


def test_build_sessions_skips_no_footage_row() -> None:
    rows = [
        {
            "participant_id": "P004",
            "camera": "Short",
            "grocery_start": None,
            "grocery_end": None,
            "storybook_start": None,
            "storybook_end": None,
            "notes": "No camera footage",
        }
    ]
    assert timing.build_sessions(rows) == []


def test_manual_exclusion_applied_for_1018_gap() -> None:
    rows = [
        {
            "participant_id": "1018",
            "camera": "short",
            "grocery_start": "19:55",
            "grocery_end": "37:17",
            "storybook_start": "44:54",
            "storybook_end": "52:56",
            "notes": "bathroom break ...",
        }
    ]
    s = timing.build_sessions(rows)[0]
    reasons = [e.reason for e in s.exclusions]
    assert "EEG headband re-fit gap" in reasons
    # The grocery window is split by the gap into two fragments.
    grocery_spans = [sp for sp in s.sampling_spans() if sp[0] < 44 * 60]
    assert len(grocery_spans) == 2
