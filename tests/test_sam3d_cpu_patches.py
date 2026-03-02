"""Regression tests for SAM-3D CPU device patches.

These tests verify the monkey-patch pipeline that remaps upstream CUDA-hardcoded
calls to the runtime-resolved device. They do NOT require the SAM-3D submodule
or torch to be installed.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_inference.device import resolve_device, resolve_inference_mode
from video_inference.sam3d_runner import (
    _ensure_submodule_importable,
    _patch_cpu_device_paths,
    _patch_estimator_device_transfers,
)


def test_ensure_submodule_importable_raises_with_actionable_message(
    tmp_path: Path,
) -> None:
    """Verify missing submodule produces a clear, actionable error."""
    fake_root = tmp_path / "nonexistent"
    with patch("video_inference.sam3d_runner.SUBMODULE_ROOT", fake_root):
        with pytest.raises(FileNotFoundError, match="git submodule update"):
            _ensure_submodule_importable()


def test_patch_cpu_device_paths_replaces_method_on_model() -> None:
    """Verify _patch_cpu_device_paths binds a new method to the model."""

    class FakeModel:
        def get_ray_condition(self, batch):
            raise RuntimeError("Original CUDA-hardcoded method called")

    model = FakeModel()
    _patch_cpu_device_paths(model)

    # The method should have been replaced (bound method, not the original)
    assert "cpu_safe" in model.get_ray_condition.__func__.__name__


def test_full_cpu_patch_pipeline_is_coherent() -> None:
    """End-to-end: resolve_device -> resolve_inference_mode -> patches remap cuda."""
    device = resolve_device(
        "auto", cuda_available_fn=lambda: False, mps_available_fn=lambda: False
    )
    assert device == "cpu"

    mode = resolve_inference_mode(device, "auto")
    assert mode == "body"  # CPU forces body mode

    class FakeEstimatorModule:
        @staticmethod
        def recursive_to(batch, target_device):
            return target_device

    fake_mod = FakeEstimatorModule()
    _patch_estimator_device_transfers(device, estimator_module=fake_mod)

    # After patching, "cuda" targets should be remapped to "cpu"
    assert fake_mod.recursive_to(batch={}, target_device="cuda") == "cpu"
    # Non-cuda targets pass through unchanged
    assert fake_mod.recursive_to(batch={}, target_device="meta") == "meta"
