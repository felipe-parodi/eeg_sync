import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_inference.device import resolve_device, resolve_inference_mode  # noqa: E402


def test_resolve_device_auto_prefers_cuda_when_available():
    device = resolve_device("auto", cuda_available_fn=lambda: True)
    assert device == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_when_cuda_unavailable():
    device = resolve_device("auto", cuda_available_fn=lambda: False)
    assert device == "cpu"


def test_resolve_device_cuda_raises_when_unavailable():
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda", cuda_available_fn=lambda: False)


def test_resolve_inference_mode_auto_uses_body_on_cpu():
    assert resolve_inference_mode("cpu", "auto") == "body"


def test_resolve_inference_mode_full_downgrades_on_cpu():
    assert resolve_inference_mode("cpu", "full") == "body"


def test_resolve_inference_mode_auto_uses_full_on_cuda():
    assert resolve_inference_mode("cuda", "auto") == "full"
