import sys
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="requires Python 3.11+")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sync_eeg_vid import (
    extract_eeg_segment,
    find_sync_from_csv,
    find_sync_pulse,
    validate_file_paths,
)


def _write_raw_eeg_txt(path: Path) -> None:
    lines = [
        "%OpenBCI Raw EEG Data",
        "%Sample Rate = 250 Hz",
        "Sample Index,Analog Channel 0",
        "0,257",
        "1,",
        "2,257",
        "3,1",
        "4,1",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_pyproject_entrypoint_points_to_existing_module():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["eeg-sync"] == "sync_eeg_vid.cli:main"


def test_validate_file_paths_accepts_txt_eeg_extension():
    files = {
        "eeg_file_a": "/tmp/session_a.txt",
        "eeg_file_b": None,
        "video_a": "/tmp/video_a.mov",
        "video_b": None,
    }
    is_valid, errors = validate_file_paths(files)
    assert is_valid
    assert errors == []


def test_find_sync_from_csv_ignores_nan_values(tmp_path: Path):
    csv_path = tmp_path / "ir_cleaned.csv"
    csv_path.write_text(
        "Time (sec),Value\n"
        "0.000,257\n"
        "0.004,\n"
        "0.008,257\n"
        "0.012,1\n",
        encoding="utf-8",
    )

    sync_time = find_sync_from_csv(csv_path)
    assert sync_time == pytest.approx(0.012)


def test_find_sync_pulse_falls_back_to_txt_when_preferred_csv_is_invalid(tmp_path: Path):
    txt_path = tmp_path / "session.txt"
    _write_raw_eeg_txt(txt_path)

    # Matched by the *.irBlaster.csv lookup when starting from session.txt
    bad_csv = tmp_path / "sessionfixed_irBlaster.csv"
    bad_csv.write_text("bad_col,another\n1,2\n", encoding="utf-8")

    sync_time, source = find_sync_pulse(txt_path, prefer_csv=True)

    assert source == "TXT: session.txt"
    assert sync_time == pytest.approx(3 / 250)


def test_find_sync_pulse_csv_input_falls_back_to_matching_txt(tmp_path: Path):
    txt_path = tmp_path / "session.txt"
    _write_raw_eeg_txt(txt_path)

    # Name includes fixed_irBlaster token, so fallback should recover session.txt
    bad_csv = tmp_path / "session_fixed_irBlaster.csv"
    bad_csv.write_text("Time,WrongValue\n0,0\n", encoding="utf-8")

    sync_time, source = find_sync_pulse(bad_csv, prefer_csv=True)

    assert source == "TXT: session.txt"
    assert sync_time == pytest.approx(3 / 250)


def test_extract_eeg_segment_rejects_non_increasing_video_range(tmp_path: Path):
    txt_path = tmp_path / "session.txt"
    _write_raw_eeg_txt(txt_path)

    with pytest.raises(ValueError, match="video_end must be greater than video_start"):
        extract_eeg_segment(
            eeg_filepath=txt_path,
            video_start=10.0,
            video_end=5.0,
            eeg_video_offset=0.0,
        )
