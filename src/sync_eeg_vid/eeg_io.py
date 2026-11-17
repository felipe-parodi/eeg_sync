"""EEG file I/O: reading, parsing, and sync-pulse detection."""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from .util import _validate_file_exists

# EEG data constants
DEFAULT_IR_COLUMN = "Analog Channel 0"
IR_BASELINE_VALUE = 257
HEADER_COMMENT_CHAR = "%"
DEFAULT_SAMPLE_RATE = 250  # Hz
NUM_EEG_CHANNELS = 8


def _read_eeg_sample_rate(filepath: Path) -> int:
    """
    Extract sample rate from EEG file header.

    Args:
        filepath: Path to EEG .txt file

    Returns:
        Sample rate in Hz

    Raises:
        ValueError: If sample rate not found in header
    """
    with open(filepath, "r") as f:
        for line in f:
            if "Sample Rate" in line:
                try:
                    # Extract from format: "%Sample Rate = 250 Hz"
                    rate_str = line.split("=")[1].strip().split(" ")[0]
                    return int(rate_str)
                except (IndexError, ValueError) as e:
                    raise ValueError(
                        f"Could not parse sample rate from line: {line}"
                    ) from e
    raise ValueError("Sample rate not found in file header")


def _read_eeg_header(filepath: Path) -> Tuple[int, int, List[str]]:
    """
    Read EEG file header and extract metadata.

    Args:
        filepath: Path to EEG .txt file

    Returns:
        Tuple of (sample_rate, header_lines, column_names)

    Raises:
        ValueError: If required header information is missing
    """
    sample_rate = None
    header_lines = 0
    column_names = []

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(HEADER_COMMENT_CHAR):
                header_lines += 1
                if "Sample Rate" in line:
                    try:
                        rate_str = line.split("=")[1].strip().split(" ")[0]
                        sample_rate = int(rate_str)
                    except (IndexError, ValueError) as e:
                        raise ValueError(f"Could not parse sample rate: {line}") from e
            else:
                # First non-comment line should contain column names
                if "Sample Index" in line:
                    raw_names = [col.strip() for col in line.split(",")]
                    # Handle duplicate column names
                    column_names = _make_unique_column_names(raw_names)
                    header_lines += 1
                break

    if sample_rate is None:
        raise ValueError("Sample rate not found in file header")
    if not column_names:
        raise ValueError("Column headers not found in file")

    return sample_rate, header_lines, column_names


def _make_unique_column_names(names: List[str]) -> List[str]:
    """
    Make column names unique by appending suffix to duplicates.

    Args:
        names: List of potentially duplicate names

    Returns:
        List of unique names
    """
    unique_names = []
    name_counts = {}

    for name in names:
        if name in name_counts:
            name_counts[name] += 1
            unique_names.append(f"{name}_{name_counts[name]}")
        else:
            name_counts[name] = 0
            unique_names.append(name)

    return unique_names


def _find_non_baseline_mask(values: pd.Series, baseline_value: float) -> pd.Series:
    """
    Return a mask for valid (non-NaN) values that differ from baseline.

    Uses a small absolute tolerance to avoid floating-point edge cases.
    """
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid_values = numeric_values.notna()
    is_baseline = pd.Series(
        np.isclose(
            numeric_values.to_numpy(),
            baseline_value,
            rtol=0.0,
            atol=1e-6,
            equal_nan=False,
        ),
        index=numeric_values.index,
    )
    return valid_values & ~is_baseline


def find_sync_from_raw_eeg(
    filepath: Union[str, Path],
    ir_blaster_column: str = DEFAULT_IR_COLUMN,
    baseline_value: float = IR_BASELINE_VALUE,
) -> float:
    """
    Find IR blaster synchronization pulse in raw OpenBCI EEG file.

    Args:
        filepath: Path to raw OpenBCI .txt data file
        ir_blaster_column: Name of column containing IR data
        baseline_value: Baseline value of IR signal (off state)

    Returns:
        Timestamp in seconds of first sync pulse

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or no sync pulse found
    """
    filepath = _validate_file_exists(filepath)

    # Read header information
    sample_rate, header_lines, column_names = _read_eeg_header(filepath)

    # Read data
    try:
        data = pd.read_csv(
            filepath, skiprows=header_lines, names=column_names, low_memory=False
        )
    except Exception as e:
        raise ValueError(f"Failed to read EEG data: {e}") from e

    # Validate IR column exists
    if ir_blaster_column not in data.columns:
        raise ValueError(
            f"IR blaster column '{ir_blaster_column}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    # Find first non-baseline value
    ir_values = pd.to_numeric(data[ir_blaster_column], errors="coerce")
    sync_event_rows = data[_find_non_baseline_mask(ir_values, baseline_value)]

    if sync_event_rows.empty:
        raise ValueError("No sync pulse found in file")

    # Calculate timestamp
    first_pulse_index = sync_event_rows.index[0]
    sync_timestamp = first_pulse_index / sample_rate

    return sync_timestamp


def find_sync_from_csv(
    filepath: Union[str, Path], baseline_value: float = IR_BASELINE_VALUE
) -> float:
    """
    Find IR blaster synchronization pulse in cleaned CSV file.

    CSV files contain pre-cleaned IR blaster data with noise removed.
    This is FASTER and MORE RELIABLE than parsing raw .txt files.

    CSV format:
        Time (sec),Value
        0.0,257.0
        0.004,257.0
        ...
        81.344,1      # IR pulse (consecutive 1s)
        81.348,1

    Args:
        filepath: Path to cleaned CSV file
        baseline_value: Baseline value of IR signal (257.0 = off)

    Returns:
        Timestamp in seconds of first sync pulse

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or no sync pulse found
    """
    filepath = _validate_file_exists(filepath)

    try:
        # Read CSV file
        data = pd.read_csv(filepath)

        # Validate columns
        if "Time (sec)" not in data.columns or "Value" not in data.columns:
            raise ValueError(
                f"CSV must have 'Time (sec)' and 'Value' columns. "
                f"Found: {list(data.columns)}"
            )

        # Find first non-baseline value
        # Convert to numeric to handle both int and float
        values = pd.to_numeric(data["Value"], errors="coerce")
        non_baseline = _find_non_baseline_mask(values, baseline_value)

        if not non_baseline.any():
            raise ValueError(
                f"No IR pulse found (all valid values are {baseline_value})"
            )

        # Get first pulse timestamp
        first_pulse_idx = non_baseline.idxmax()
        sync_time = float(data.loc[first_pulse_idx, "Time (sec)"])

        return sync_time

    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}") from e


def find_sync_pulse(
    filepath: Union[str, Path], prefer_csv: bool = True
) -> Tuple[float, str]:
    """
    Find IR blaster sync pulse, auto-detecting file type.

    Tries to find matching CSV file (cleaner data) or falls back to TXT.

    Args:
        filepath: Path to EEG file (.txt or .csv)
        prefer_csv: If True, look for CSV file first

    Returns:
        Tuple of (sync_timestamp, file_used)

    Raises:
        FileNotFoundError: If no valid file found
        ValueError: If sync pulse not found
    """
    path = _validate_file_exists(filepath)
    suffix = path.suffix.lower()
    csv_errors = []
    txt_errors = []

    def _unique_paths(paths: List[Path]) -> List[Path]:
        unique = []
        seen = set()
        for p in paths:
            p_str = str(p)
            if p_str in seen:
                continue
            seen.add(p_str)
            unique.append(p)
        return unique

    def _derive_txt_candidates_from_csv(csv_path: Path) -> List[Path]:
        candidates = [csv_path.with_suffix(".txt")]
        stem = csv_path.stem

        # Remove common CSV suffix patterns to recover the raw TXT stem.
        for token in ["fixed_irBlaster", "irBlaster"]:
            token_idx = stem.find(token)
            if token_idx != -1:
                raw_stem = stem[:token_idx].rstrip(" _-")
                if raw_stem:
                    candidates.append(csv_path.with_name(f"{raw_stem}.txt"))
                    candidates.extend(sorted(csv_path.parent.glob(f"{raw_stem}*.txt")))
        return _unique_paths(candidates)

    # Try CSV first when requested.
    csv_candidates: List[Path] = []
    if prefer_csv:
        if suffix == ".csv":
            csv_candidates = [path]
        elif suffix in [".txt", ""]:
            base_name = path.stem
            csv_candidates = sorted(path.parent.glob(f"{base_name}*irBlaster.csv"))

        for csv_path in _unique_paths(csv_candidates):
            if not csv_path.exists():
                continue
            try:
                sync_time = find_sync_from_csv(csv_path)
                return sync_time, f"CSV: {csv_path.name}"
            except (FileNotFoundError, ValueError) as e:
                csv_errors.append(f"{csv_path.name}: {e}")
                print(f"  ⚠ CSV read failed: {e}")
                print("  → Falling back to TXT file...")

    # Fall back to TXT candidates.
    txt_candidates: List[Path] = []
    if suffix in [".txt", ""]:
        txt_candidates = [path]
    elif suffix == ".csv":
        txt_candidates = _derive_txt_candidates_from_csv(path)
    else:
        raise FileNotFoundError(
            f"Unsupported EEG file extension '{path.suffix}'. Use .txt or .csv."
        )

    for txt_path in _unique_paths(txt_candidates):
        if not txt_path.exists():
            continue
        try:
            sync_time = find_sync_from_raw_eeg(txt_path)
            return sync_time, f"TXT: {txt_path.name}"
        except (FileNotFoundError, ValueError) as e:
            txt_errors.append(f"{txt_path.name}: {e}")

    # Report why sync extraction failed.
    error_lines = csv_errors + txt_errors
    if error_lines:
        joined = "\n  - ".join(error_lines)
        raise ValueError(f"Could not find sync pulse. Details:\n  - {joined}")

    raise FileNotFoundError(
        "No valid EEG file found for sync extraction. " f"Started from: {filepath}"
    )


def extract_eeg_segment(
    eeg_filepath: Union[str, Path],
    video_start: float,
    video_end: float,
    eeg_video_offset: float,
) -> pd.DataFrame:
    """
    Extract EEG data segment corresponding to a video time range.

    Args:
        eeg_filepath: Path to raw EEG .txt file
        video_start: Start time in video (seconds)
        video_end: End time in video (seconds)
        eeg_video_offset: Offset from sync_eeg_to_video result

    Returns:
        DataFrame with EEG data and time columns

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or time range invalid
    """
    filepath = _validate_file_exists(eeg_filepath)

    if video_end <= video_start:
        raise ValueError(
            "Invalid video time range: video_end must be greater than video_start."
        )

    # Convert video time to EEG time
    # video_time = eeg_time + offset -> eeg_time = video_time - offset
    eeg_start = video_start - eeg_video_offset
    eeg_end = video_end - eeg_video_offset

    if eeg_start < 0:
        raise ValueError(
            f"Calculated EEG start time is negative: {eeg_start:.2f}s. "
            "Check your offset and video times."
        )

    # Read header
    sample_rate, header_lines, column_names = _read_eeg_header(filepath)

    # Read data
    data = pd.read_csv(
        filepath, skiprows=header_lines, names=column_names, low_memory=False
    )

    # Calculate sample indices
    start_sample = int(eeg_start * sample_rate)
    end_sample = int(eeg_end * sample_rate)

    if start_sample >= len(data):
        raise ValueError(f"Start sample {start_sample} exceeds data length {len(data)}")

    # Extract segment
    end_sample = min(end_sample, len(data))
    segment = data.iloc[start_sample:end_sample].copy()

    # Add time columns
    segment["Time_EEG"] = np.arange(len(segment)) / sample_rate + eeg_start
    segment["Time_Video"] = segment["Time_EEG"] + eeg_video_offset

    return segment
