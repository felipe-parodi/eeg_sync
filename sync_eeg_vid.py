"""
EEG-Video Synchronization Tool

Synchronizes OpenBCI EEG recordings with video files using IR blaster pulses
and visual/audio cues. Optimized for long recordings (2+ hours).

Features:
- Frame-perfect video synchronization
- Multi-camera support
- Fast seeking for long videos
- Cross-platform compatibility
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import sys

import pandas as pd
import cv2
import numpy as np

# Note: 10s startup is unavoidable - pandas/cv2/numpy are heavy libraries
# The tradeoff is worth it for the functionality they provide

# ============================================================================
# Constants
# ============================================================================

# EEG data constants
DEFAULT_IR_COLUMN = "Analog Channel 0"
IR_BASELINE_VALUE = 257
HEADER_COMMENT_CHAR = "%"
DEFAULT_SAMPLE_RATE = 250  # Hz
NUM_EEG_CHANNELS = 8

# Video viewer constants
# Linux/Windows arrow keys
ARROW_KEY_LEFT = 81
ARROW_KEY_RIGHT = 83
ARROW_KEY_UP = 82
ARROW_KEY_DOWN = 84

# macOS arrow keys (when masked with 0xFF)
# macOS returns 63234, 63235, 63232, 63233 which become 2, 3, 0, 1
ARROW_KEY_LEFT_MAC = 2
ARROW_KEY_RIGHT_MAC = 3
ARROW_KEY_UP_MAC = 0
ARROW_KEY_DOWN_MAC = 1

KEY_ESC = 27
KEY_ENTER = 13
KEY_NO_PRESS = 255

# Navigation step sizes (in seconds)
STEP_SINGLE_FRAME = 1.0  # Will be divided by FPS
STEP_ONE_SECOND = 1.0
STEP_TEN_SECONDS = 10.0
STEP_ONE_MINUTE = 60.0

# Display settings
MAX_DISPLAY_WIDTH = 1920
KEYBOARD_POLL_MS = 1


# ============================================================================
# Helper Functions
# ============================================================================


def _validate_file_exists(filepath: Union[str, Path]) -> Path:
    """Validate that a file exists and return Path object."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not path.is_file():
        raise ValueError(f"Not a file: {filepath}")
    return path


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


def parse_timestamp(timestamp: Union[str, float, int]) -> float:
    """
    Parse human-readable timestamp to seconds.

    Accepts formats:
    - "1:23" -> 83.0 seconds (1 min 23 sec)
    - "1:23.5" -> 83.5 seconds
    - "83" or 83 -> 83.0 seconds

    Args:
        timestamp: Timestamp in various formats

    Returns:
        Timestamp in seconds

    Raises:
        ValueError: If timestamp format is invalid
    """
    if isinstance(timestamp, (int, float)):
        return float(timestamp)

    timestamp = str(timestamp).strip()

    if ":" in timestamp:
        try:
            parts = timestamp.split(":")
            if len(parts) != 2:
                raise ValueError("Timestamp must be in format 'M:SS' or 'M:SS.ms'")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid timestamp format: {timestamp}") from e
    else:
        try:
            return float(timestamp)
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format: {timestamp}") from e


def format_time(seconds: float) -> str:
    """
    Format seconds as M:SS.mmm string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "1:23.456"
    """
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


def ask_yes_no(prompt: str) -> bool:
    """
    Ask a yes/no question with validation.

    Args:
        prompt: Question to ask (without y/n suffix)

    Returns:
        True if yes, False if no
    """
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print(f"  ⚠ Invalid input '{response}'. Please enter 'y' or 'n'.")


def ask_file_path(prompt: str, must_exist: bool = True) -> str:
    """
    Ask for a file path with validation.

    Args:
        prompt: Question to ask
        must_exist: If True, file must exist

    Returns:
        Valid file path as string
    """
    while True:
        response = input(f"{prompt}: ").strip()

        if not response:
            print("  ⚠ Please enter a file path.")
            continue

        path = Path(response)

        if must_exist:
            if not path.exists():
                print(f"  ⚠ File not found: {response}")
                print(f"    Please check the path and try again.")
                continue
            if not path.is_file():
                print(f"  ⚠ Path is not a file: {response}")
                continue

        return str(path)


# ============================================================================
# Core EEG Synchronization Functions
# ============================================================================


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
    sync_event_rows = data[ir_values != baseline_value]

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
        non_baseline = values != baseline_value

        if not non_baseline.any():
            raise ValueError(f"No IR pulse found (all values are {baseline_value})")

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
    path = Path(filepath)

    # Try CSV first if preferred
    if prefer_csv:
        # Look for CSV file
        if path.suffix.lower() == ".csv":
            csv_path = path
        else:
            # Try to find matching CSV file
            # Pattern: TEp_OpenBCI-RAW-*.txt → TEp_OpenBCI-RAW-*fixed_irBlaster.csv
            base_name = path.stem  # e.g., "TEp_OpenBCI-RAW-2025-11-14_10-40-38"
            csv_pattern = f"{base_name}*irBlaster.csv"
            csv_files = list(path.parent.glob(csv_pattern))

            csv_path = csv_files[0] if csv_files else None

        if csv_path and csv_path.exists():
            try:
                sync_time = find_sync_from_csv(csv_path)
                return sync_time, f"CSV: {csv_path.name}"
            except (FileNotFoundError, ValueError) as e:
                print(f"  ⚠ CSV read failed: {e}")
                print(f"  → Falling back to TXT file...")

    # Fall back to TXT file
    if path.suffix.lower() in [".txt", ""]:
        txt_path = path
    else:
        raise FileNotFoundError(f"No valid EEG file found. Tried: {filepath}")

    sync_time = find_sync_from_raw_eeg(txt_path)
    return sync_time, f"TXT: {txt_path.name}"


# ============================================================================
# Video Frame Viewer
# ============================================================================


class VideoFrameViewer:
    """
    ULTRA-FAST interactive video frame viewer for precise synchronization.

    Optimized for long videos (2+ hours) with efficient frame seeking
    and responsive controls.
    """

    def __init__(
        self, video_path: Union[str, Path], max_display_width: int = MAX_DISPLAY_WIDTH
    ):
        """
        Initialize video viewer.

        Args:
            video_path: Path to video file
            max_display_width: Max width for display (auto-downscale if larger)

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video can't be opened or has invalid properties
        """
        self.video_path = _validate_file_exists(video_path)
        self.max_display_width = max_display_width

        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get and validate video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.fps <= 0:
            self.cap.release()
            raise ValueError(f"Invalid FPS: {self.fps}")
        if self.total_frames <= 0:
            self.cap.release()
            raise ValueError(f"Invalid frame count: {self.total_frames}")

        self.duration = self.total_frames / self.fps

        # Calculate display scaling
        self.scale = 1.0
        if self.width > max_display_width:
            self.scale = max_display_width / self.width

        # Frame cache for instant navigation
        self.frame_cache: Dict[int, np.ndarray] = {}

    def __del__(self):
        """Ensure video capture is released."""
        if hasattr(self, "cap"):
            self.cap.release()
            cv2.destroyAllWindows()

    def jump_to_timestamp(self, timestamp: Union[str, float]) -> None:
        """
        Jump to specific timestamp in video.

        Args:
            timestamp: Time in seconds or "M:SS" format
        """
        if isinstance(timestamp, str):
            time_sec = parse_timestamp(timestamp)
        else:
            time_sec = float(timestamp)

        frame_num = int(time_sec * self.fps)
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    def _read_frame(self, frame_num: int) -> Optional[np.ndarray]:
        """
        Read frame from video with caching.

        Args:
            frame_num: Frame number to read

        Returns:
            Frame as numpy array, or None if failed
        """
        # Check cache first
        if frame_num in self.frame_cache:
            return self.frame_cache[frame_num].copy()

        # Seek to frame BEFORE reading (cap.read() advances position!)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            # Cache current frame
            self.frame_cache = {frame_num: frame.copy()}
            return frame
        return None

    def _create_display_frame(
        self,
        frame: np.ndarray,
        current_frame: int,
        current_time: float,
        paused: bool = True,
    ) -> np.ndarray:
        """
        Create display frame with overlay.

        Args:
            frame: Original frame
            current_frame: Current frame number
            current_time: Current time in seconds
            paused: If True, show "PAUSED" indicator

        Returns:
            Frame with timestamp overlay
        """
        # Scale if needed
        if self.scale < 1.0:
            new_width = int(self.width * self.scale)
            new_height = int(self.height * self.scale)
            display_frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )
        else:
            display_frame = frame.copy()

        # Format timestamp
        time_str = format_time(current_time)
        frame_str = f"Frame {current_frame}/{self.total_frames}"

        # Calculate overlay scaling
        overlay_scale = self.scale if self.scale < 1.0 else 1.0
        font_scale = 1.5 * overlay_scale
        thickness = max(1, int(2 * overlay_scale))

        # Draw background rectangle
        cv2.rectangle(
            display_frame,
            (10, 10),
            (int(550 * overlay_scale), int(130 * overlay_scale)),
            (0, 0, 0),
            -1,
        )

        # Draw timestamp
        cv2.putText(
            display_frame,
            time_str,
            (20, int(50 * overlay_scale)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
        )

        # Draw frame number
        cv2.putText(
            display_frame,
            frame_str,
            (20, int(85 * overlay_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.53,
            (255, 255, 255),
            max(1, thickness - 1),
        )

        # Draw PAUSED indicator
        if paused:
            cv2.putText(
                display_frame,
                "[PAUSED - Use arrows to navigate]",
                (20, int(115 * overlay_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.5,
                (0, 255, 255),  # Yellow
                max(1, thickness - 1),
            )

        return display_frame

    def find_frame(
        self,
        window_name: str = "Frame Finder",
        start_timestamp: Optional[Union[str, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Interactive frame finder with keyboard navigation.

        Controls:
        - LEFT/RIGHT arrows: ±1 frame
        - UP/DOWN arrows: ±1 second
        - , / . keys: ±10 seconds
        - [ / ] keys: ±1 minute
        - SPACE: Mark frame and exit
        - Q/ESC: Quit without marking

        Args:
            window_name: Name for OpenCV window
            start_timestamp: Optional starting timestamp

        Returns:
            Dict with frame info if marked, None if cancelled
        """
        # Jump to start position if provided
        if start_timestamp is not None:
            self.jump_to_timestamp(start_timestamp)

        # Print info
        print(f"\n{'='*60}")
        print(f"OPENING VIDEO WINDOW - LOOK FOR IT ON YOUR SCREEN!")
        print(f"{'='*60}")
        print(f"\nVideo: {self.video_path.name}")
        print(
            f"Duration: {format_time(self.duration)} | "
            f"FPS: {self.fps:.1f} | "
            f"Frames: {self.total_frames}"
        )

        if self.scale < 1.0:
            print(
                f"Auto-downscaling for speed: "
                f"{self.width}px → {self.max_display_width}px"
            )

        if start_timestamp is not None:
            start_time = parse_timestamp(start_timestamp)
            print(f"→ Jumped to {format_time(start_time)}")

        print("\nControls:")
        print("  A/D or ← →:         ±1 frame")
        print("  W/S or ↑ ↓:         ±1 second")
        print("  , / . keys:         ±10 seconds")
        print("  [ / ] keys:         ±1 minute")
        print("  SPACE:              Mark frame and EXIT")
        print("  Q/ESC:              Quit")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Initial display - get starting frame position
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        needs_redraw = True

        # Flag to set window to topmost after first frame is shown
        first_display = True

        try:
            while True:
                # Only redraw if needed (key was pressed)
                if needs_redraw:
                    # DON'T call cap.get() here - it returns position AFTER last read!
                    # Use the manually tracked current_frame instead
                    current_time = current_frame / self.fps

                    # Read frame
                    frame = self._read_frame(current_frame)
                    if frame is None:
                        # Hit end of video, go back one frame
                        current_frame = max(0, current_frame - 1)
                        continue

                    # Create display with PAUSED indicator
                    display_frame = self._create_display_frame(
                        frame, current_frame, current_time, paused=True
                    )
                    cv2.imshow(window_name, display_frame)

                    # On first display, try to bring window to front
                    if first_display:
                        try:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                            # Then turn off topmost so user can move it if needed
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
                        except:
                            pass  # Ignore if property not supported
                        first_display = False
                        print(
                            f"\n>>> VIDEO WINDOW OPENED: '{window_name}' - Check your screen! <<<"
                        )

                    needs_redraw = False

                # Wait for key press (PAUSED - no continuous redraw!)
                key = cv2.waitKey(0) & 0xFF

                # Process key
                result = self._handle_key(key, current_frame, current_time)
                if result is not None:
                    return result  # User marked frame or quit

                # Check if navigation key was pressed
                new_frame = self._calculate_new_frame(key, current_frame)
                if new_frame != current_frame:
                    current_frame = new_frame
                    needs_redraw = True

        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)  # Process window events on macOS
            cv2.destroyAllWindows()

    def _handle_key(
        self, key: int, current_frame: int, current_time: float
    ) -> Optional[Dict[str, Any]]:
        """
        Handle keyboard input.

        Args:
            key: Key code
            current_frame: Current frame number
            current_time: Current time in seconds

        Returns:
            Result dict if user marked frame or cancelled, None to continue
        """
        # Debug: Print key code and action (use spaces to clear previous line)
        key_name = self._get_key_name(key)
        print(f"[DEBUG] Key: {key} ({key_name})                    ", end="\r")

        # Quit
        if key == ord("q") or key == KEY_ESC:
            print("\n✗ Cancelled")
            return {}  # Empty dict signals cancellation

        # Mark frame
        if key == ord(" "):
            result = {
                "frame_number": current_frame,
                "timestamp": current_time,
                "time_formatted": format_time(current_time),
                "marked": True,
            }
            print(
                f"\n✓ Marked: Frame {current_frame} at "
                f"{format_time(current_time)} ({current_time:.4f}s)"
            )
            return result

        # No action taken (navigation is handled separately)
        return None

    def _get_key_name(self, key: int) -> str:
        """Get human-readable name for key code."""
        key_map = {
            ARROW_KEY_LEFT: "LEFT",
            ARROW_KEY_RIGHT: "RIGHT",
            ARROW_KEY_UP: "UP",
            ARROW_KEY_DOWN: "DOWN",
            ARROW_KEY_LEFT_MAC: "LEFT(mac)",
            ARROW_KEY_RIGHT_MAC: "RIGHT(mac)",
            ARROW_KEY_UP_MAC: "UP(mac)",
            ARROW_KEY_DOWN_MAC: "DOWN(mac)",
            ord("a"): "A",
            ord("d"): "D",
            ord("w"): "W",
            ord("s"): "S",
            ord("q"): "Q",
            ord(" "): "SPACE",
            KEY_ESC: "ESC",
            KEY_ENTER: "ENTER",
        }
        return key_map.get(key, f"?")

    def _calculate_new_frame(
        self, key: int, current_frame: int, allow_wasd: bool = True
    ) -> int:
        """
        Calculate new frame based on key press.

        Args:
            key: Key code
            current_frame: Current frame number
            allow_wasd: If True, allow W/A/S/D keys for navigation (default True)
                        If False, only arrow keys work (for multi-clap mode)

        Returns:
            New frame number
        """
        # Single frame navigation (arrow keys + A/D fallback)
        # Support both Linux (81/83) and macOS (2/3) arrow codes
        if key in [ARROW_KEY_RIGHT, ARROW_KEY_RIGHT_MAC]:
            return min(current_frame + 1, self.total_frames - 1)
        elif key == ord("d") and allow_wasd:
            return min(current_frame + 1, self.total_frames - 1)
        elif key in [ARROW_KEY_LEFT, ARROW_KEY_LEFT_MAC]:
            return max(current_frame - 1, 0)
        elif key == ord("a") and allow_wasd:
            return max(current_frame - 1, 0)

        # 1 second navigation (arrow keys + W/S fallback)
        # Support both Linux (82/84) and macOS (0/1) arrow codes
        elif key in [ARROW_KEY_UP, ARROW_KEY_UP_MAC]:
            return min(
                current_frame + int(self.fps * STEP_ONE_SECOND), self.total_frames - 1
            )
        elif key == ord("w") and allow_wasd:
            return min(
                current_frame + int(self.fps * STEP_ONE_SECOND), self.total_frames - 1
            )
        elif key in [ARROW_KEY_DOWN, ARROW_KEY_DOWN_MAC]:
            return max(current_frame - int(self.fps * STEP_ONE_SECOND), 0)
        elif key == ord("s") and allow_wasd:
            return max(current_frame - int(self.fps * STEP_ONE_SECOND), 0)

        # 10 second navigation
        elif key == ord(",") or key == ord("<"):
            return max(current_frame - int(self.fps * STEP_TEN_SECONDS), 0)
        elif key == ord(".") or key == ord(">"):
            return min(
                current_frame + int(self.fps * STEP_TEN_SECONDS), self.total_frames - 1
            )

        # 1 minute navigation
        elif key == ord("[") or key == ord("{"):
            return max(current_frame - int(self.fps * STEP_ONE_MINUTE), 0)
        elif key == ord("]") or key == ord("}"):
            return min(
                current_frame + int(self.fps * STEP_ONE_MINUTE), self.total_frames - 1
            )

        return current_frame

    def find_multiple_frames(
        self,
        window_name: str = "Mark Multiple Claps",
        start_timestamp: Optional[Union[str, float]] = None,
        expected_count: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Interactive multi-frame marker for clap synchronization.

        Controls:
        - C: Mark a clap at current frame
        - U: Undo last marked clap
        - ENTER/SPACE: Save and exit
        - A/D or ← →: ±1 frame
        - W/S or ↑ ↓: ±1 second
        - , / . keys: ±10 seconds
        - [ / ] keys: ±1 minute
        - Q/ESC: Quit without saving

        Args:
            window_name: Name for OpenCV window
            start_timestamp: Optional starting timestamp
            expected_count: Optional expected number of claps to mark

        Returns:
            List of marked frame dicts if saved, None if cancelled
        """
        # Jump to start position if provided
        if start_timestamp is not None:
            self.jump_to_timestamp(start_timestamp)

        # Print info
        print(f"\n{'='*60}")
        print(f"MULTI-CLAP MARKING MODE")
        print(f"{'='*60}")
        print(f"\nVideo: {self.video_path.name}")
        print(
            f"Duration: {format_time(self.duration)} | "
            f"FPS: {self.fps:.1f} | "
            f"Frames: {self.total_frames}"
        )

        if expected_count:
            print(f"\n⚠ Please mark exactly {expected_count} claps")

        if start_timestamp is not None:
            start_time = parse_timestamp(start_timestamp)
            print(f"→ Jumped to {format_time(start_time)}")

        print("\nControls:")
        print("  C:                  Mark clap at current frame")
        print("  U:                  Undo last marked clap")
        print("  ENTER/SPACE:        Save and EXIT")
        print("  ← →:                ±1 frame")
        print("  ↑ ↓:                ±1 second")
        print("  , / . keys:         ±10 seconds")
        print("  [ / ] keys:         ±1 minute")
        print("  Q/ESC:              Quit without saving")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Track marked claps
        marked_claps: List[Dict[str, Any]] = []

        # Initial display - get starting frame position
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        needs_redraw = True

        # Flag to set window to topmost after first frame is shown
        first_display = True

        try:
            while True:
                # Only redraw if needed
                if needs_redraw:
                    # DON'T call cap.get() here - use manually tracked current_frame
                    current_time = current_frame / self.fps

                    # Read frame
                    frame = self._read_frame(current_frame)
                    if frame is None:
                        current_frame = max(0, current_frame - 1)
                        continue

                    # Create display with marked claps overlay
                    display_frame = self._create_multi_clap_display(
                        frame, current_frame, current_time, marked_claps, expected_count
                    )
                    cv2.imshow(window_name, display_frame)

                    # On first display, try to bring window to front
                    if first_display:
                        try:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                            # Then turn off topmost so user can move it if needed
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
                        except:
                            pass  # Ignore if property not supported
                        first_display = False
                        print(
                            f"\n>>> VIDEO WINDOW OPENED: '{window_name}' - Check your screen! <<<"
                        )

                    needs_redraw = False

                # Wait for key press
                key = cv2.waitKey(0) & 0xFF

                # Debug key code
                key_name = self._get_key_name(key)
                print(f"[DEBUG] Key: {key} ({key_name})                    ", end="\r")

                # Handle clap marking keys
                if key == ord("c"):
                    # Mark clap and update display immediately
                    clap_info = {
                        "frame_number": current_frame,
                        "timestamp": current_time,
                        "time_formatted": format_time(current_time),
                    }
                    marked_claps.append(clap_info)
                    print(
                        f"\n✓ Marked clap #{len(marked_claps)} at "
                        f"{format_time(current_time)} (frame {current_frame})"
                    )
                    # Redraw to update clap counter display (Bug fix #2)
                    needs_redraw = True

                elif key == ord("u"):
                    # Undo last clap
                    if marked_claps:
                        removed = marked_claps.pop()
                        print(
                            f"\n↶ Undid clap at "
                            f"{removed['time_formatted']} "
                            f"(frame {removed['frame_number']})"
                        )
                        # Redraw to update clap count display
                        needs_redraw = True
                    else:
                        print("\n⚠ No claps to undo", end="\r")

                # Save and exit
                elif key == ord(" ") or key == KEY_ENTER:  # SPACE or ENTER
                    if not marked_claps:
                        print(
                            "\n⚠ No claps marked! Press C to mark claps, or Q to quit"
                        )
                        continue

                    if expected_count and len(marked_claps) != expected_count:
                        print(
                            f"\n⚠ Expected {expected_count} claps, "
                            f"but you marked {len(marked_claps)}. "
                            f"Continue anyway? (y/n): ",
                            end="",
                        )
                        # This is tricky in the video window - just accept it
                        pass

                    print(f"\n✓ Saved {len(marked_claps)} claps!")
                    return marked_claps

                # Quit
                elif key == ord("q") or key == KEY_ESC:  # Q or ESC
                    print("\n✗ Cancelled (no claps saved)")
                    return None

                # Navigation (disable W/A/S/D to avoid conflicts with C key)
                else:
                    new_frame = self._calculate_new_frame(
                        key, current_frame, allow_wasd=False
                    )
                    if new_frame != current_frame:
                        current_frame = new_frame
                        needs_redraw = True

        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)  # Process window events on macOS
            cv2.destroyAllWindows()

    def _create_multi_clap_display(
        self,
        frame: np.ndarray,
        current_frame: int,
        current_time: float,
        marked_claps: List[Dict[str, Any]],
        expected_count: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create display frame with multi-clap marking overlay.

        Args:
            frame: Original frame
            current_frame: Current frame number
            current_time: Current time in seconds
            marked_claps: List of marked clap dictionaries
            expected_count: Expected number of claps (for progress display)

        Returns:
            Frame with overlay
        """
        # Scale if needed
        if self.scale < 1.0:
            new_width = int(self.width * self.scale)
            new_height = int(self.height * self.scale)
            display_frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )
        else:
            display_frame = frame.copy()

        # Calculate overlay scaling
        overlay_scale = self.scale if self.scale < 1.0 else 1.0
        font_scale = 1.5 * overlay_scale
        thickness = max(1, int(2 * overlay_scale))

        # Format timestamp
        time_str = format_time(current_time)
        frame_str = f"Frame {current_frame}/{self.total_frames}"

        # Count display
        if expected_count:
            count_str = f"Marked: {len(marked_claps)}/{expected_count} claps"
        else:
            count_str = f"Marked: {len(marked_claps)} claps"

        # Draw background rectangle
        bg_height = 180 if marked_claps else 130
        cv2.rectangle(
            display_frame,
            (10, 10),
            (int(600 * overlay_scale), int(bg_height * overlay_scale)),
            (0, 0, 0),
            -1,
        )

        # Draw timestamp
        cv2.putText(
            display_frame,
            time_str,
            (20, int(50 * overlay_scale)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
        )

        # Draw frame number
        cv2.putText(
            display_frame,
            frame_str,
            (20, int(85 * overlay_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.53,
            (255, 255, 255),
            max(1, thickness - 1),
        )

        # Draw clap count
        color = (
            (0, 255, 0)
            if (not expected_count or len(marked_claps) == expected_count)
            else (0, 165, 255)
        )
        cv2.putText(
            display_frame,
            count_str,
            (20, int(115 * overlay_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.5,
            color,
            max(1, thickness - 1),
        )

        # Draw marked claps list
        if marked_claps:
            y_pos = 145
            claps_str = "Claps: " + ", ".join(
                f"{c['time_formatted']}" for c in marked_claps
            )
            cv2.putText(
                display_frame,
                claps_str,
                (20, int(y_pos * overlay_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.45,
                (100, 200, 255),
                max(1, thickness - 1),
            )

        # Draw instructions reminder
        cv2.putText(
            display_frame,
            "[C: mark | U: undo | ENTER: save]",
            (20, int((bg_height - 10) * overlay_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.45,
            (200, 200, 200),
            max(1, thickness - 1),
        )

        return display_frame


# ============================================================================
# High-Level Synchronization Functions
# ============================================================================


def sync_two_eeg_files(
    eeg_filepath_a: Union[str, Path], eeg_filepath_b: Union[str, Path]
) -> Optional[Dict[str, Any]]:
    """
    Synchronize two EEG recordings using IR blaster pulses.

    Both EEG files should have IR blaster pulses at approximately the same
    real-world time. This calculates the offset between them.

    Args:
        eeg_filepath_a: Path to first EEG .txt file (reference)
        eeg_filepath_b: Path to second EEG .txt file

    Returns:
        Dict with synchronization results, or None if error
    """
    print("=" * 60)
    print("STEP 1: Synchronizing Two EEG Recordings")
    print("=" * 60)

    # Find IR pulse in EEG A (prefers clean CSV files)
    print("\n[1/2] Analyzing EEG file A for IR blaster pulse...")
    try:
        eeg_sync_time_a, source_a = find_sync_pulse(eeg_filepath_a)
        print(f"  Using: {source_a}")
        print(
            f"✓ IR blaster pulse found at {eeg_sync_time_a:.4f} seconds "
            "in EEG file A"
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error in EEG file A: {e}")
        return None

    # Find IR pulse in EEG B (prefers clean CSV files)
    print("\n[2/2] Analyzing EEG file B for IR blaster pulse...")
    try:
        eeg_sync_time_b, source_b = find_sync_pulse(eeg_filepath_b)
        print(f"  Using: {source_b}")
        print(
            f"✓ IR blaster pulse found at {eeg_sync_time_b:.4f} seconds "
            "in EEG file B"
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error in EEG file B: {e}")
        return None

    # Calculate offset
    offset = eeg_sync_time_a - eeg_sync_time_b

    result = {
        "eeg_file_a": str(eeg_filepath_a),
        "eeg_file_b": str(eeg_filepath_b),
        "eeg_sync_time_a": eeg_sync_time_a,
        "eeg_sync_time_b": eeg_sync_time_b,
        "offset": offset,
        "note": (
            f"To convert EEG B time to EEG A time: " f"time_a = time_b + {offset:.4f}"
        ),
    }

    print("\n✓ EEG synchronization complete!")
    print(f"  EEG A IR pulse: {eeg_sync_time_a:.4f} seconds")
    print(f"  EEG B IR pulse: {eeg_sync_time_b:.4f} seconds")
    print(f"  Time offset:    {offset:.4f} seconds")

    return result


def sync_eeg_to_video(
    eeg_filepath: Union[str, Path],
    video_path: Union[str, Path],
    approx_video_time: Optional[Union[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Synchronize EEG data to video using IR blaster pulse.

    Workflow:
    1. Find IR blaster pulse in EEG data
    2. User finds exact frame of red light in video
    3. Calculate time offset

    Args:
        eeg_filepath: Path to raw EEG .txt file
        video_path: Path to video file
        approx_video_time: Approximate timestamp of red light in video

    Returns:
        Dict with synchronization results, or None if cancelled

    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If file formats are invalid
    """
    print("=" * 60)
    print("STEP 1: Synchronizing EEG to Video")
    print("=" * 60)

    # Find IR pulse in EEG (prefers clean CSV files)
    print("\n[1/3] Analyzing EEG data for IR blaster pulse...")
    try:
        eeg_sync_time, source = find_sync_pulse(eeg_filepath)
        print(f"  Using: {source}")
        print(f"✓ IR blaster pulse found at {eeg_sync_time:.4f} seconds " "in EEG data")
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error: {e}")
        return None

    # Find red light in video
    print("\n[2/3] Finding exact frame of red light in video...")
    if approx_video_time:
        print(f"Starting at approximate time: {approx_video_time}")

    try:
        viewer = VideoFrameViewer(video_path)
        video_result = viewer.find_frame(
            window_name="Find RED LIGHT sync", start_timestamp=approx_video_time
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error: {e}")
        return None

    if not video_result or not video_result.get("marked"):
        print("✗ Synchronization cancelled")
        return None

    video_sync_time = video_result["timestamp"]
    print(f"✓ Red light found at {video_sync_time:.4f} seconds in video")

    # Calculate offset
    print("\n[3/3] Calculating synchronization offset...")
    offset = video_sync_time - eeg_sync_time

    result = {
        "eeg_file": str(eeg_filepath),
        "video_file": str(video_path),
        "eeg_sync_time": eeg_sync_time,
        "video_sync_time": video_sync_time,
        "video_frame": video_result["frame_number"],
        "offset": offset,
        "note": (
            f"To convert EEG time to video time: "
            f"video_time = eeg_time + {offset:.4f}"
        ),
    }

    print("\n✓ Synchronization complete!")
    print(f"  EEG IR pulse:    {eeg_sync_time:.4f} seconds")
    print(
        f"  Video red light: {video_sync_time:.4f} seconds "
        f"(frame {video_result['frame_number']})"
    )
    print(f"  Time offset:     {offset:.4f} seconds")

    return result


def sync_videos(
    video_a_path: Union[str, Path],
    video_b_path: Union[str, Path],
    approx_time_a: Optional[Union[str, float]] = None,
    approx_time_b: Optional[Union[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Synchronize two videos using multiple claps/sync events.

    Workflow:
    1. Mark claps in Video A (C to mark, U to undo, ENTER to save)
    2. Mark same number of claps in Video B
    3. Calculate offset using average of all clap pairs

    Args:
        video_a_path: Path to first video (reference)
        video_b_path: Path to second video
        approx_time_a: Approximate timestamp of claps in video A
        approx_time_b: Approximate timestamp of claps in video B

    Returns:
        Dict with synchronization results, or None if cancelled
    """
    print("\n" + "=" * 60)
    print("STEP 2: Synchronizing Video B to Video A (Multi-Clap)")
    print("=" * 60)

    # Mark claps in Video A
    print("\n[1/3] Marking claps in Video A...")
    if approx_time_a:
        print(f"Starting at approximate time: {approx_time_a}")

    try:
        viewer_a = VideoFrameViewer(video_a_path)
        claps_a = viewer_a.find_multiple_frames(
            window_name="Mark CLAPS in Video A", start_timestamp=approx_time_a
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error: {e}")
        return None

    if not claps_a:
        print("✗ Synchronization cancelled (no claps marked in Video A)")
        return None

    num_claps = len(claps_a)
    print(f"\n✓ Marked {num_claps} clap(s) in Video A:")
    for i, clap in enumerate(claps_a, 1):
        print(f"  Clap {i}: {clap['time_formatted']} (frame {clap['frame_number']})")

    # Mark claps in Video B (with retry if counts don't match)
    claps_b = None
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        print(f"\n[2/3] Marking {num_claps} clap(s) in Video B...")
        print(f"⚠ Please mark the SAME {num_claps} clap(s) in Video B")
        if approx_time_b:
            print(f"Starting at approximate time: {approx_time_b}")

        try:
            viewer_b = VideoFrameViewer(video_b_path)
            claps_b = viewer_b.find_multiple_frames(
                window_name="Mark CLAPS in Video B",
                start_timestamp=approx_time_b,
                expected_count=num_claps,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ Error: {e}")
            return None

        if not claps_b:
            print("✗ Synchronization cancelled (no claps marked in Video B)")
            return None

        # Check if counts match
        if len(claps_b) == num_claps:
            break  # Success!
        else:
            print(
                f"\n⚠ Mismatch: Video A has {num_claps} claps, but you marked {len(claps_b)} in Video B"
            )
            if ask_yes_no("  Do you want to re-mark the claps in Video B?"):
                retry_count += 1
                continue
            else:
                print("  Proceeding with mismatched clap counts...")
                break

    print(f"\n✓ Marked {len(claps_b)} clap(s) in Video B:")
    for i, clap in enumerate(claps_b, 1):
        print(f"  Clap {i}: {clap['time_formatted']} (frame {clap['frame_number']})")

    # Calculate offset(s)
    print(f"\n[3/3] Calculating synchronization offset...")

    if len(claps_b) != num_claps:
        print(
            f"⚠ Warning: Different number of claps "
            f"(A: {num_claps}, B: {len(claps_b)})"
        )

    # Calculate offset for each clap pair
    offsets = []
    for i in range(min(num_claps, len(claps_b))):
        offset_i = claps_a[i]["timestamp"] - claps_b[i]["timestamp"]
        offsets.append(offset_i)
        print(
            f"  Clap {i+1} offset: {offset_i:.4f}s "
            f"(A: {claps_a[i]['time_formatted']}, "
            f"B: {claps_b[i]['time_formatted']})"
        )

    # Use average offset
    avg_offset = sum(offsets) / len(offsets)

    # Calculate offset variability
    if len(offsets) > 1:
        offset_std = np.std(offsets)
        print(f"\n  Average offset: {avg_offset:.4f}s")
        print(f"  Std deviation:  {offset_std:.4f}s")
        if offset_std > 0.1:
            print(f"  ⚠ Warning: High variability in offsets (>{0.1}s)")
    else:
        print(f"\n  Offset: {avg_offset:.4f}s")

    result = {
        "video_a": str(video_a_path),
        "video_b": str(video_b_path),
        "claps_a": claps_a,
        "claps_b": claps_b,
        "offsets": offsets,
        "offset": avg_offset,
        "offset_std": np.std(offsets) if len(offsets) > 1 else 0.0,
        "note": (
            f"To convert Video B time to Video A time: "
            f"time_a = time_b + {avg_offset:.4f}"
        ),
    }

    print("\n✓ Video synchronization complete!")
    print(f"  Final offset: {avg_offset:.4f} seconds")

    return result


# ============================================================================
# Plotting & Analysis Functions
# ============================================================================


def plot_sync_timeline(
    eeg_sync_result: Dict[str, Any],
    video_sync_result: Optional[Dict[str, Any]] = None,
    eeg_eeg_sync_result: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Visualize synchronization timeline between EEG(s) and video(s).

    Shows all data streams:
    - Dual EEG: EEG A, EEG B, Video A, Video B (4 streams)
    - Single EEG: EEG, Video A, Video B (3 streams)

    Args:
        eeg_sync_result: Result from sync_eeg_to_video()
        video_sync_result: Optional result from sync_videos()
        eeg_eeg_sync_result: Optional result from sync_two_eeg_files()
        figsize: Figure size tuple

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Determine layout based on available data
    has_dual_eeg = eeg_eeg_sync_result is not None
    has_video_b = video_sync_result is not None

    # Y-axis positions (bottom to top)
    y_pos = 0
    yticks = []
    yticklabels = []

    # Draw Video B (if present) - bottom track
    if has_video_b:
        if "claps_b" in video_sync_result:
            # Multi-clap sync
            claps_b = video_sync_result["claps_b"]
            max_time_b = max(c["timestamp"] for c in claps_b)
            ax.barh(
                y_pos,
                max_time_b,
                left=0,
                height=0.4,
                color="#9b59b6",
                alpha=0.7,
                label="Video B",
            )
            for i, clap in enumerate(claps_b):
                ax.plot(
                    clap["timestamp"],
                    y_pos,
                    "g^",
                    markersize=8,
                    label=f"Clap {i+1} (Video B)" if i == 0 else "",
                )
        yticks.append(y_pos)
        yticklabels.append("Video B")
        y_pos += 1

    # Draw Video A
    video_a_time = eeg_sync_result["video_sync_time"]
    ax.barh(
        y_pos,
        video_a_time,
        left=0,
        height=0.4,
        color="#e74c3c",
        alpha=0.7,
        label="Video A",
    )
    ax.plot(video_a_time, y_pos, "r*", markersize=12, label="Red Light (Video A)")

    # Add claps if present
    if has_video_b and "claps_a" in video_sync_result:
        for i, clap in enumerate(video_sync_result["claps_a"]):
            ax.plot(
                clap["timestamp"],
                y_pos,
                "g^",
                markersize=8,
                label=f"Clap {i+1} (Video A)" if i == 0 else "",
            )

    yticks.append(y_pos)
    yticklabels.append("Video A")
    y_pos += 1

    # Draw EEG B (if dual EEG) - second from top
    if has_dual_eeg:
        eeg_b_time = eeg_eeg_sync_result["eeg_sync_time_b"]
        ax.barh(
            y_pos,
            eeg_b_time,
            left=0,
            height=0.4,
            color="#2ecc71",
            alpha=0.7,
            label="EEG B",
        )
        ax.plot(eeg_b_time, y_pos, "ro", markersize=8, label="IR Pulse (EEG B)")
        yticks.append(y_pos)
        yticklabels.append("EEG B")
        y_pos += 1

    # Draw EEG A - top track
    eeg_a_time = eeg_sync_result["eeg_sync_time"]
    ax.barh(
        y_pos,
        eeg_a_time,
        left=0,
        height=0.4,
        color="#3498db",
        alpha=0.7,
        label="EEG A" if has_dual_eeg else "EEG",
    )
    ax.plot(
        eeg_a_time,
        y_pos,
        "ro",
        markersize=8,
        label="IR Pulse (EEG A)" if has_dual_eeg else "IR Pulse",
    )
    yticks.append(y_pos)
    yticklabels.append("EEG A" if has_dual_eeg else "EEG")

    # Draw sync arrows and labels
    # EEG A/B sync (if dual EEG)
    if has_dual_eeg:
        eeg_b_y = yticks[-2]  # EEG B position
        eeg_a_y = yticks[-1]  # EEG A position
        eeg_b_time = eeg_eeg_sync_result["eeg_sync_time_b"]
        eeg_a_time = eeg_eeg_sync_result["eeg_sync_time_a"]
        eeg_offset = eeg_eeg_sync_result["offset"]

        ax.annotate(
            "",
            xy=(eeg_a_time, eeg_a_y - 0.25),
            xytext=(eeg_b_time, eeg_b_y + 0.25),
            arrowprops=dict(arrowstyle="<->", color="purple", lw=2),
        )
        ax.text(
            (eeg_a_time + eeg_b_time) / 2,
            (eeg_a_y + eeg_b_y) / 2,
            f"Δ={eeg_offset:.3f}s",
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    # EEG → Video A sync
    video_a_y = yticks[1 if has_video_b else 0]  # Video A position
    eeg_ref_y = yticks[-1]  # Top EEG position
    eeg_to_vid_offset = eeg_sync_result["offset"]

    ax.annotate(
        "",
        xy=(video_a_time, video_a_y + 0.25),
        xytext=(eeg_a_time, eeg_ref_y - 0.25),
        arrowprops=dict(arrowstyle="<->", color="green", lw=2),
    )
    ax.text(
        (eeg_a_time + video_a_time) / 2,
        (video_a_y + eeg_ref_y) / 2,
        f"Δ={eeg_to_vid_offset:.3f}s",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Video A → Video B sync (if present)
    if has_video_b:
        video_b_y = yticks[0]
        v_offset = video_sync_result["offset"]

        # Use first clap times for arrow
        if "claps_a" in video_sync_result and "claps_b" in video_sync_result:
            sync_time_a = video_sync_result["claps_a"][0]["timestamp"]
            sync_time_b = video_sync_result["claps_b"][0]["timestamp"]
        else:
            sync_time_a = video_sync_result.get("sync_time_a", video_a_time)
            sync_time_b = video_sync_result.get("sync_time_b", 0)

        ax.annotate(
            "",
            xy=(sync_time_a, video_a_y - 0.25),
            xytext=(sync_time_b, video_b_y + 0.25),
            arrowprops=dict(arrowstyle="<->", color="orange", lw=2),
        )

        # Show offset with std if multiple claps
        if "offset_std" in video_sync_result and video_sync_result["offset_std"] > 0:
            offset_text = (
                f"Δ={v_offset:.3f}s\n±{video_sync_result['offset_std']*1000:.0f}ms"
            )
        else:
            offset_text = f"Δ={v_offset:.3f}s"

        ax.text(
            (sync_time_a + sync_time_b) / 2,
            (video_a_y + video_b_y) / 2,
            offset_text,
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    # Styling
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time (seconds)", fontsize=12)

    title_parts = []
    if has_dual_eeg:
        title_parts.append("Dual EEG")
    if has_video_b:
        title_parts.append("Dual Video")
    title = (
        " + ".join(title_parts) + " Synchronization"
        if title_parts
        else "EEG-Video Synchronization"
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    return fig


def plot_eeg_data(
    eeg_filepath: Union[str, Path],
    start_time: float = 0,
    duration: float = 10,
    channels: Optional[List[int]] = None,
    sync_time: Optional[float] = None,
    show_ir: bool = True,
):
    """
    Plot EEG channels from raw data file.

    Args:
        eeg_filepath: Path to raw EEG .txt file
        start_time: Start time in seconds
        duration: Duration to plot in seconds
        channels: List of channel indices to plot (None = all)
        sync_time: If provided, mark the sync point with vertical line
        show_ir: Whether to show IR blaster channel

    Returns:
        Matplotlib figure

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    import matplotlib.pyplot as plt

    filepath = _validate_file_exists(eeg_filepath)

    # Read header
    sample_rate, header_lines, column_names = _read_eeg_header(filepath)

    # Read data
    data = pd.read_csv(
        filepath, skiprows=header_lines, names=column_names, low_memory=False
    )

    # Calculate sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)

    # Extract segment
    segment = data.iloc[start_sample:end_sample]
    time_axis = np.arange(len(segment)) / sample_rate + start_time

    # Determine channels
    if channels is None:
        channels = list(range(NUM_EEG_CHANNELS))
    eeg_channels = [f"EXG Channel {i}" for i in channels]

    # Create subplots
    n_plots = len(eeg_channels) + (1 if show_ir else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    # Plot EEG channels
    for i, ch_name in enumerate(eeg_channels):
        if ch_name in segment.columns:
            axes[i].plot(time_axis, segment[ch_name], linewidth=0.5)
            axes[i].set_ylabel(ch_name.replace("EXG Channel ", "Ch "), fontsize=10)
            axes[i].grid(alpha=0.3)

            # Mark sync point
            if (
                sync_time is not None
                and start_time <= sync_time <= start_time + duration
            ):
                axes[i].axvline(
                    sync_time,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Sync Point",
                )
                if i == 0:
                    axes[i].legend(loc="upper right")

    # Plot IR channel
    if show_ir and DEFAULT_IR_COLUMN in segment.columns:
        axes[-1].plot(
            time_axis, segment[DEFAULT_IR_COLUMN], linewidth=1, color="orange"
        )
        axes[-1].set_ylabel("IR Blaster", fontsize=10)
        axes[-1].grid(alpha=0.3)

        if sync_time is not None and start_time <= sync_time <= start_time + duration:
            axes[-1].axvline(sync_time, color="red", linestyle="--", alpha=0.7)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    fig.suptitle(
        f"EEG Data: {start_time}s to {start_time + duration}s",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


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


# ============================================================================
# Main CLI Interface
# ============================================================================


def collect_all_files() -> Dict[str, Optional[str]]:
    """
    Collect all file paths from user upfront.

    Returns:
        Dict with keys: eeg_file_a, eeg_file_b, video_a, video_b
    """
    print("\n" + "=" * 60)
    print("FILE SELECTION")
    print("=" * 60)
    print("\nPlease provide paths to your files.")
    print("Required: At least 1 EEG file AND 1 Video file")
    print("Optional: Second EEG and/or second Video")
    print("\nYou can:")
    print("  - Drag and drop files into the terminal")
    print("  - Type/paste the full file path")
    print("  - Press ENTER to skip optional files")

    files = {}

    # EEG File A
    print("\n" + "-" * 60)
    files["eeg_file_a"] = ask_file_path("EEG File A (.csv) - REQUIRED", must_exist=True)

    # EEG File B (optional)
    print("\n" + "-" * 60)
    response = input("EEG File B (.csv) - OPTIONAL (press ENTER to skip): ").strip()
    if response:
        try:
            files["eeg_file_b"] = str(_validate_file_exists(response))
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠ Warning: {e}")
            print("  → Skipping EEG File B")
            files["eeg_file_b"] = None
    else:
        files["eeg_file_b"] = None

    # Video A
    print("\n" + "-" * 60)
    files["video_a"] = ask_file_path(
        "Video A (.mp4/.mov/.avi) - REQUIRED (with red light sync)", must_exist=True
    )

    # Video B (optional)
    print("\n" + "-" * 60)
    response = input(
        "Video B (.mp4/.mov/.avi) - OPTIONAL (press ENTER to skip): "
    ).strip()
    if response:
        try:
            files["video_b"] = str(_validate_file_exists(response))
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠ Warning: {e}")
            print("  → Skipping Video B")
            files["video_b"] = None
    else:
        files["video_b"] = None

    return files


def validate_file_paths(files: Dict[str, Optional[str]]) -> Tuple[bool, List[str]]:
    """
    Validate collected file paths.

    Args:
        files: Dict with file paths

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check minimum requirements
    if not files.get("eeg_file_a"):
        errors.append("❌ At least one EEG file is required")
    if not files.get("video_a"):
        errors.append("❌ At least one Video file is required")

    # Check for duplicates
    provided_files = {k: v for k, v in files.items() if v}
    file_paths = list(provided_files.values())
    file_labels = list(provided_files.keys())

    for i in range(len(file_paths)):
        for j in range(i + 1, len(file_paths)):
            if file_paths[i] == file_paths[j]:
                errors.append(
                    f"❌ Duplicate file: {file_labels[i]} and {file_labels[j]} "
                    f"are using the same file"
                )

    # Validate extensions
    eeg_exts = [".csv"]
    video_exts = [".mp4", ".mov", ".avi"]

    for key, filepath in files.items():
        if filepath:
            path = Path(filepath)
            if "eeg" in key.lower():
                if path.suffix.lower() not in eeg_exts:
                    errors.append(f"⚠️ {key}: Expected .csv file, got {path.suffix}")
            elif "video" in key.lower():
                if path.suffix.lower() not in video_exts:
                    errors.append(
                        f"⚠️ {key}: Expected video file (.mp4/.mov/.avi), got {path.suffix}"
                    )

    return len(errors) == 0, errors


def determine_workflow(files: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Determine which sync steps will run based on provided files.

    Args:
        files: Dict with file paths

    Returns:
        Dict with workflow info
    """
    workflow = {
        "sync_eeg_to_eeg": False,
        "sync_eeg_to_video": False,
        "sync_video_to_video": False,
        "steps": [],
    }

    # Check what syncs are possible
    if files.get("eeg_file_a") and files.get("eeg_file_b"):
        workflow["sync_eeg_to_eeg"] = True
        workflow["steps"].append("✓ Step 1: Sync EEG A ↔ EEG B")

    if (files.get("eeg_file_a") or files.get("eeg_file_b")) and files.get("video_a"):
        workflow["sync_eeg_to_video"] = True
        eeg_label = "EEG A" if files.get("eeg_file_a") else "EEG B"
        workflow["steps"].append(f"✓ Step 2: Sync {eeg_label} → Video A")

    if files.get("video_a") and files.get("video_b"):
        workflow["sync_video_to_video"] = True
        workflow["steps"].append("✓ Step 3: Sync Video A ↔ Video B")

    return workflow


def ask_red_light_timestamp() -> Optional[str]:
    """
    Ask for red light timestamp hint for EEG→Video sync.

    Returns:
        Timestamp hint or None
    """
    print("\n" + "=" * 60)
    print("RED LIGHT TIMESTAMP HINT")
    print("=" * 60)
    print("\nTo help locate the red light sync point in the video:")
    print("Format: 'M:SS' (e.g., '1:23') or seconds (e.g., '83')")
    print("Press ENTER to start at 0:00")

    response = input("\nApproximate time of RED LIGHT in Video A: ").strip()
    return response if response else None


def ask_clap_timestamps() -> Tuple[Optional[str], Optional[str]]:
    """
    Ask for clap timestamp hints for Video A↔Video B sync.

    Returns:
        Tuple of (video_a_hint, video_b_hint)
    """
    print("\n" + "=" * 60)
    print("CLAP TIMESTAMP HINTS")
    print("=" * 60)
    print("\nTo help locate the clap sync points in both videos:")
    print("Format: 'M:SS' (e.g., '2:30') or seconds (e.g., '150')")
    print("Press ENTER to start at 0:00 for any field.")

    print("\n" + "-" * 60)
    response_a = input("Approximate time of CLAPS in Video A: ").strip()
    approx_a = response_a if response_a else None

    print("\n" + "-" * 60)
    response_b = input("Approximate time of CLAPS in Video B: ").strip()
    approx_b = response_b if response_b else None

    return approx_a, approx_b


def main():
    """
    Main interactive workflow for EEG-video synchronization.

    Workflow:
    1. Collect all file paths upfront
    2. Validate files
    3. Show workflow preview
    4. Collect timestamp hints
    5. Run synchronization steps
    6. Save results
    """
    import json

    print("\n" + "=" * 60)
    print("EEG-VIDEO SYNCHRONIZATION TOOL")
    print("=" * 60)

    # Step 1: Collect all files
    files = collect_all_files()

    # Step 2: Validate files
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    is_valid, errors = validate_file_paths(files)

    if not is_valid:
        print("\n⚠️ Validation errors found:")
        for error in errors:
            print(f"  {error}")
        print("\nPlease fix the errors and try again.")
        return

    print("✓ All files validated successfully!")

    # Step 3: Determine workflow
    workflow = determine_workflow(files)

    print("\n" + "=" * 60)
    print("WORKFLOW PREVIEW")
    print("=" * 60)
    print("\nBased on your files, the following sync steps will run:\n")
    for step in workflow["steps"]:
        print(f"  {step}")

    # Confirm to proceed
    if not ask_yes_no("\nProceed with synchronization?"):
        print("Cancelled.")
        return

    # Step 4: Run synchronization workflow
    print("\n" + "=" * 60)
    print("SYNCHRONIZATION")
    print("=" * 60)

    eeg_eeg_sync = None
    eeg_sync = None
    video_sync = None

    # Determine which EEG file to use as reference
    eeg_ref_file = files.get("eeg_file_a") or files.get("eeg_file_b")

    # Step 4a: Sync EEG A ↔ EEG B (no timestamp hint needed - uses IR pulse)
    if workflow["sync_eeg_to_eeg"]:
        try:
            eeg_eeg_sync = sync_two_eeg_files(files["eeg_file_a"], files["eeg_file_b"])
        except (FileNotFoundError, ValueError) as e:
            print(f"\n✗ Error syncing EEG files: {e}")
            print("\nExiting...")
            return

        if eeg_eeg_sync is None:
            print("\nEEG sync cancelled. Exiting...")
            return

    # Step 4b: Sync EEG → Video A (ask for red light timestamp just-in-time)
    if workflow["sync_eeg_to_video"]:
        # Ask for timestamp hint NOW (right before opening video)
        approx_red = ask_red_light_timestamp()

        try:
            eeg_sync = sync_eeg_to_video(eeg_ref_file, files["video_a"], approx_red)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n✗ Error syncing EEG to video: {e}")
            print("\nExiting...")
            return

        if eeg_sync is None:
            print("\nEEG-Video sync cancelled. Exiting...")
            return

    # Step 4c: Sync Video A ↔ Video B (ask for clap timestamps just-in-time)
    if workflow["sync_video_to_video"]:
        # Ask for timestamp hints NOW (right before opening videos)
        approx_claps_a, approx_claps_b = ask_clap_timestamps()

        try:
            video_sync = sync_videos(
                files["video_a"], files["video_b"], approx_claps_a, approx_claps_b
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"\n✗ Error syncing videos: {e}")
            print("Continuing without video sync...")

    # Print summary
    print("\n" + "=" * 60)
    print("SYNCHRONIZATION SUMMARY")
    print("=" * 60)

    if eeg_eeg_sync:
        print(f"\nEEG File A: {eeg_eeg_sync['eeg_file_a']}")
        print(f"EEG File B: {eeg_eeg_sync['eeg_file_b']}")
        print("\nEEG A ↔ EEG B:")
        print(f"  Offset: {eeg_eeg_sync['offset']:.4f} seconds")
        print(f"  {eeg_eeg_sync['note']}")
    else:
        print(f"\nEEG File: {eeg_ref_file}")

    if eeg_sync:
        print(f"Video A:  {files['video_a']}")
        print("\nEEG ↔ Video A:")
        print(f"  Offset: {eeg_sync['offset']:.4f} seconds")
        print(f"  {eeg_sync['note']}")

    if video_sync:
        print(f"\nVideo B:  {video_sync['video_b']}")
        print("\nVideo A ↔ Video B:")
        print(f"  Offset: {video_sync['offset']:.4f} seconds")
        print(f"  {video_sync['note']}")

    # Save results
    save = ask_yes_no("\nSave synchronization results to file?")
    if save:
        output_file = "sync_results.json"
        results = {
            "eeg_a_to_eeg_b": eeg_eeg_sync,
            "eeg_to_video_a": eeg_sync,
            "video_a_to_video_b": video_sync,
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {output_file}")

    # Offer to generate plots
    if eeg_sync:  # Only offer if we have EEG-video sync
        plot = ask_yes_no("\nGenerate synchronization timeline plot?")
        if plot:
            try:
                fig = plot_sync_timeline(eeg_sync, video_sync, eeg_eeg_sync)
                plot_file = "sync_timeline.png"
                import matplotlib.pyplot as plt

                plt.savefig(plot_file, dpi=150, bbox_inches="tight")
                print(f"✓ Plot saved to {plot_file}")
                plt.close()
            except Exception as e:
                print(f"⚠ Could not generate plot: {e}")

    # Show how to use the data
    print("\n" + "=" * 60)
    print("HOW TO USE THIS DATA")
    print("=" * 60)
    print(
        """
To use the synchronization offsets in your analysis:

1. Load the sync results:
   import json
   with open('sync_results.json') as f:
       sync = json.load(f)

2. Convert timestamps:
   # EEG B time → EEG A time
   eeg_a_time = eeg_b_time + sync['eeg_a_to_eeg_b']['offset']

   # EEG time → Video time
   video_time = eeg_time + sync['eeg_to_video_a']['offset']

3. Extract aligned data:
   from sync_eeg_vid import extract_eeg_segment

   # Get EEG data for video segment 1:30 to 2:00
   eeg_data = extract_eeg_segment(
       eeg_filepath='path/to/eeg.csv',
       video_start=90,   # 1:30 in seconds
       video_end=120,    # 2:00 in seconds
       eeg_video_offset=sync['eeg_to_video_a']['offset']
   )
   # Returns DataFrame with Time_EEG and Time_Video columns

See README.md for more examples!
"""
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
