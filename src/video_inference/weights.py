"""Where model weights live.

All ``*.pt`` weights sit in ``<repo>/models/`` (gitignored). Code may refer to
a weights file by its bare name (``"yolo11m-pose.pt"``); :func:`resolve_weights`
turns that into ``models/<name>``, so Ultralytics finds it there, or downloads
a stock model there instead of into whatever folder a command runs from.
"""

from __future__ import annotations

from pathlib import Path

from video_inference.paths import MODELS_DIR  # noqa: E402,F401


def resolve_weights(path: str | Path) -> str:
    """Return where a weights file should be loaded from.

    Args:
        path: A bare file name (``yolo11l-pose.pt``) or a path.

    Returns:
        The path unchanged when it has a folder or exists in the current
        folder; otherwise ``models/<name>``.
    """
    candidate = Path(path)
    if candidate.parent != Path(".") or candidate.exists():
        return str(path)
    return str(MODELS_DIR / candidate.name)
