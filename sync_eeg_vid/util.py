"""Utility helpers shared across the sync_eeg_vid package."""

from pathlib import Path
from typing import Union

try:
    import cv2
except ImportError:
    cv2 = None


def _validate_file_exists(filepath: Union[str, Path]) -> Path:
    """Validate that a file exists and return Path object."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not path.is_file():
        raise ValueError(f"Not a file: {filepath}")
    return path


def _require_cv2() -> None:
    """Require OpenCV for interactive video operations."""
    if cv2 is None:
        raise ImportError(
            "OpenCV is required for video synchronization features. "
            "Install with: pip install opencv-python"
        )


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


def _strip_quotes(filepath: str) -> str:
    """
    Strip surrounding quotes from file path and validate.

    REQUIRES quotes around paths with spaces. This prevents common errors
    from unquoted paths being interpreted incorrectly.

    Args:
        filepath: Raw file path string

    Returns:
        File path with quotes removed

    Raises:
        ValueError: If path contains spaces but is not quoted
    """
    original = filepath
    filepath = filepath.strip()

    # Check if path has quotes
    has_quotes = False
    if len(filepath) >= 2:
        if (filepath[0] == '"' and filepath[-1] == '"') or (
            filepath[0] == "'" and filepath[-1] == "'"
        ):
            has_quotes = True
            filepath = filepath[1:-1]

    # Validate: if path has spaces, it MUST have been quoted
    if " " in filepath and not has_quotes:
        raise ValueError(
            f"Path contains spaces and must be enclosed in quotes.\n"
            f"  You entered: {original}\n"
            f'  Please use:  "{filepath}"'
        )

    return filepath


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
        print(f"\n{prompt}")
        print("(Paste path then press ENTER, or type manually)")
        print("→ ", end="", flush=True)

        response = input().strip()

        if not response:
            print("  ⚠ Please enter a file path.")
            continue

        # Clean up pasted text that might include the prompt
        # This happens when terminals include prompt text in paste
        if response.startswith(prompt):
            response = response[len(prompt) :].strip()
            # Also remove leading colon if present
            if response.startswith(":"):
                response = response[1:].strip()

        # Strip quotes and validate (requires quotes for paths with spaces)
        try:
            response = _strip_quotes(response)
        except ValueError as e:
            print(f"  ⚠ {e}")
            continue

        path = Path(response)

        if must_exist:
            if not path.exists():
                print(f"  ⚠ File not found: {response}")
                print("    Please check the path and try again.")
                print("    TIP: If path has spaces, wrap it in quotes")
                continue
            if not path.is_file():
                print(f"  ⚠ Path is not a file: {response}")
                continue

        return str(path)
