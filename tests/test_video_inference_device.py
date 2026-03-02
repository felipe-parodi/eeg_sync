import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_inference.device import resolve_device, resolve_inference_mode  # noqa: E402
from video_inference.sam3d_runner import _patch_estimator_device_transfers  # noqa: E402


def test_resolve_device_auto_prefers_cuda_when_available():
    device = resolve_device(
        "auto",
        cuda_available_fn=lambda: True,
        mps_available_fn=lambda: True,
    )
    assert device == "cuda"


def test_resolve_device_auto_falls_back_to_mps_when_cuda_unavailable():
    device = resolve_device(
        "auto",
        cuda_available_fn=lambda: False,
        mps_available_fn=lambda: True,
    )
    assert device == "mps"


def test_resolve_device_auto_falls_back_to_cpu_when_no_accelerator():
    device = resolve_device(
        "auto",
        cuda_available_fn=lambda: False,
        mps_available_fn=lambda: False,
    )
    assert device == "cpu"


def test_resolve_device_cuda_raises_when_unavailable():
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device(
            "cuda",
            cuda_available_fn=lambda: False,
            mps_available_fn=lambda: False,
        )


def test_resolve_device_mps_raises_when_unavailable():
    with pytest.raises(RuntimeError, match="MPS requested"):
        resolve_device(
            "mps",
            cuda_available_fn=lambda: False,
            mps_available_fn=lambda: False,
        )


def test_resolve_inference_mode_auto_uses_body_on_cpu():
    assert resolve_inference_mode("cpu", "auto") == "body"


def test_resolve_inference_mode_full_downgrades_on_cpu():
    assert resolve_inference_mode("cpu", "full") == "body"


def test_resolve_inference_mode_auto_uses_full_on_cuda():
    assert resolve_inference_mode("cuda", "auto") == "full"


def test_patch_estimator_device_transfers_remaps_cuda_to_runtime_device():
    class FakeEstimatorModule:
        def __init__(self):
            self.calls = []

            def recursive_to(batch, target_device):
                self.calls.append(target_device)
                return {"batch": batch, "device": target_device}

            self.recursive_to = recursive_to

    fake_module = FakeEstimatorModule()
    _patch_estimator_device_transfers("cpu", estimator_module=fake_module)

    result = fake_module.recursive_to(batch={"x": 1}, target_device="cuda")
    assert result["device"] == "cpu"
    assert fake_module.calls == ["cpu"]


def test_patch_estimator_device_transfers_keeps_non_cuda_targets():
    class FakeEstimatorModule:
        def __init__(self):
            self.calls = []

            def recursive_to(batch, target_device):
                self.calls.append(target_device)
                return {"batch": batch, "device": target_device}

            self.recursive_to = recursive_to

    fake_module = FakeEstimatorModule()
    _patch_estimator_device_transfers("cpu", estimator_module=fake_module)

    result = fake_module.recursive_to(batch={"x": 1}, target_device="meta")
    assert result["device"] == "meta"
    assert fake_module.calls == ["meta"]
