"""Device/runtime helpers for SAM-3D video inference."""

from __future__ import annotations

from typing import Callable, Literal

Device = Literal["cpu", "cuda", "mps"]
DevicePreference = Literal["auto", "cpu", "cuda", "mps"]
InferenceMode = Literal["auto", "full", "body"]


def resolve_device(
    preference: DevicePreference = "auto",
    cuda_available_fn: Callable[[], bool] | None = None,
    mps_available_fn: Callable[[], bool] | None = None,
) -> Device:
    """
    Resolve execution device from a user preference.

    Args:
        preference: Requested device mode.
        cuda_available_fn: Optional detector for CUDA availability.
        mps_available_fn: Optional detector for MPS (Apple Silicon) availability.

    Returns:
        "cuda", "mps", or "cpu".
    """
    if preference not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(
            f"Invalid device preference '{preference}'. "
            "Expected one of: auto, cpu, cuda, mps."
        )

    if cuda_available_fn is None or mps_available_fn is None:
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]

    if cuda_available_fn is None:
        cuda_available = bool(torch.cuda.is_available()) if torch else False
    else:
        cuda_available = bool(cuda_available_fn())

    if mps_available_fn is None:
        mps_available = (
            bool(torch.backends.mps.is_available()) if torch else False
        )
    else:
        mps_available = bool(mps_available_fn())

    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return "cuda"
    if preference == "mps":
        if not mps_available:
            raise RuntimeError("MPS requested but no MPS device is available.")
        return "mps"

    # auto: prefer cuda > mps > cpu
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def resolve_inference_mode(
    device: Device,
    requested_mode: InferenceMode = "auto",
) -> str:
    """
    Resolve model inference mode based on device constraints.

    On CPU, default to `body` mode for compatibility and performance.
    """
    if requested_mode not in {"auto", "full", "body"}:
        raise ValueError(
            f"Invalid inference mode '{requested_mode}'. "
            "Expected one of: auto, full, body."
        )

    if requested_mode == "auto":
        return "full" if device == "cuda" else "body"

    if requested_mode == "full" and device != "cuda":
        # Upstream has CUDA-hardcoded paths in full mode.
        return "body"

    return requested_mode
