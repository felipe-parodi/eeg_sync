"""Interactive video frame viewer for precise synchronization."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .util import _require_cv2, _validate_file_exists, format_time, parse_timestamp

try:
    import cv2
except ImportError:
    cv2 = None

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
DEBUG_KEY_EVENTS = False


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
        _require_cv2()
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
        if cv2 is None:
            return
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
        print("OPENING VIDEO WINDOW - LOOK FOR IT ON YOUR SCREEN!")
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
                        except Exception:
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
        if DEBUG_KEY_EVENTS:
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
        return key_map.get(key, "?")

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
        print("MULTI-CLAP MARKING MODE")
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
                        except Exception:
                            pass  # Ignore if property not supported
                        first_display = False
                        print(
                            f"\n>>> VIDEO WINDOW OPENED: '{window_name}' - Check your screen! <<<"
                        )

                    needs_redraw = False

                # Wait for key press
                key = cv2.waitKey(0) & 0xFF

                if DEBUG_KEY_EVENTS:
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
                            "Saving with mismatched counts.",
                        )

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
