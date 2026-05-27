"""Tests for the per-frame tracker adapters in video_inference/trackers.py.

External tracker libraries are heavy (supervision pulls in opencv; boxmot
pulls in torch). These tests stub the underlying implementations and verify
only the adapter contract: each adapter returns one tracker id per input
detection, in input order, with None for unmatched detections.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _fake_supervision_module():
    """Build a minimal stand-in for the supervision module."""

    class _Detections:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _ByteTrack:
        last_kwargs: dict = {}

        def __init__(self, **kwargs):
            _ByteTrack.last_kwargs = kwargs
            self._next = 0

        def update_with_detections(self, detections):
            n = len(detections.kwargs["xyxy"])
            ids = np.arange(self._next, self._next + n, dtype=int)
            self._next += n
            return types.SimpleNamespace(tracker_id=ids)

    return types.SimpleNamespace(Detections=_Detections, ByteTrack=_ByteTrack)


def test_supervision_adapter_returns_one_id_per_detection(monkeypatch):
    monkeypatch.setitem(sys.modules, "supervision", _fake_supervision_module())
    from video_inference.trackers import SupervisionByteTrackAdapter

    adapter = SupervisionByteTrackAdapter(lost_track_buffer=60, frame_rate=8.0)

    xyxy = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    conf = np.array([0.9, 0.8], dtype=float)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ids = adapter.assign(xyxy, conf, img)

    assert ids == [0, 1]


def test_supervision_adapter_handles_empty_input(monkeypatch):
    monkeypatch.setitem(sys.modules, "supervision", _fake_supervision_module())
    from video_inference.trackers import SupervisionByteTrackAdapter

    adapter = SupervisionByteTrackAdapter(lost_track_buffer=60, frame_rate=8.0)
    out = adapter.assign(
        np.zeros((0, 4), dtype=float),
        np.zeros((0,), dtype=float),
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    assert out == []


def test_supervision_adapter_forwards_frame_rate(monkeypatch):
    fake = _fake_supervision_module()
    monkeypatch.setitem(sys.modules, "supervision", fake)
    from video_inference.trackers import SupervisionByteTrackAdapter

    SupervisionByteTrackAdapter(lost_track_buffer=42, frame_rate=8.0)

    assert fake.ByteTrack.last_kwargs == {
        "lost_track_buffer": 42,
        "frame_rate": 8.0,
    }


def _install_fake_boxmot(monkeypatch, *, det_inds, track_ids):
    """Install a fake boxmot module that returns the given assignments."""

    class _TrackResults(np.ndarray):
        @property
        def det_ind(self):
            return np.asarray(self[:, 7], dtype=int)

        @property
        def id(self):
            return np.asarray(self[:, 4], dtype=int)

        def __new__(cls, rows):
            arr = np.asarray(rows, dtype=np.float32).reshape(-1, 8)
            return arr.view(cls)

    class _BotSort:
        last_kwargs: dict = {}

        def __init__(self, **kwargs):
            _BotSort.last_kwargs = kwargs

        def update(self, dets, img):
            rows = []
            for di, tid in zip(det_inds, track_ids):
                # columns: x1, y1, x2, y2, id, conf, cls, det_ind
                rows.append([0, 0, 1, 1, tid, 0.9, 0, di])
            return _TrackResults(rows)

    class _ReID:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    botsort_mod = types.ModuleType("boxmot.trackers.botsort.botsort")
    botsort_mod.BotSort = _BotSort
    reid_mod = types.ModuleType("boxmot.reid.core.reid")
    reid_mod.ReID = _ReID
    monkeypatch.setitem(sys.modules, "boxmot.trackers.botsort.botsort", botsort_mod)
    monkeypatch.setitem(sys.modules, "boxmot.reid.core.reid", reid_mod)
    return _BotSort


def test_boxmot_adapter_maps_det_ind_back_to_input_order(monkeypatch):
    # boxmot returns outputs in tracker's internal order, with det_ind giving
    # the original detection index. Adapter must respect this.
    _install_fake_boxmot(monkeypatch, det_inds=[2, 0, 1], track_ids=[10, 11, 12])
    from video_inference.trackers import BoxmotBotSortAdapter

    adapter = BoxmotBotSortAdapter(lost_track_buffer=16, frame_rate=8.0)
    xyxy = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=float)
    conf = np.array([0.9, 0.8, 0.7], dtype=float)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ids = adapter.assign(xyxy, conf, img)

    # det_ind=2 got id=10, det_ind=0 got id=11, det_ind=1 got id=12.
    # Output is per-input order, so:
    assert ids == [11, 12, 10]


def test_boxmot_adapter_unmatched_input_returns_none(monkeypatch):
    # 3 inputs, but boxmot only emitted 2 outputs (for det_ind 0 and 2).
    _install_fake_boxmot(monkeypatch, det_inds=[0, 2], track_ids=[5, 7])
    from video_inference.trackers import BoxmotBotSortAdapter

    adapter = BoxmotBotSortAdapter(lost_track_buffer=16, frame_rate=8.0)
    ids = adapter.assign(
        np.zeros((3, 4), dtype=float),
        np.array([0.9, 0.8, 0.7]),
        np.zeros((100, 100, 3), dtype=np.uint8),
    )

    assert ids == [5, None, 7]


def test_make_tracker_dispatches_to_bytetrack(monkeypatch):
    monkeypatch.setitem(sys.modules, "supervision", _fake_supervision_module())
    from video_inference.trackers import (
        SupervisionByteTrackAdapter,
        make_tracker,
    )

    t = make_tracker(backend="roboflow", name="bytetrack", frame_rate=8.0, device="cpu")
    assert isinstance(t, SupervisionByteTrackAdapter)


def test_make_tracker_dispatches_to_botsort(monkeypatch):
    _install_fake_boxmot(monkeypatch, det_inds=[], track_ids=[])
    from video_inference.trackers import BoxmotBotSortAdapter, make_tracker

    t = make_tracker(backend="boxmot", name="botsort", frame_rate=8.0, device="cpu")
    assert isinstance(t, BoxmotBotSortAdapter)
    # 2 s default buffer at 8 fps -> 16 frames
    assert t.lost_track_buffer == 16


def test_make_tracker_rejects_unknown_pair():
    from video_inference.trackers import make_tracker

    with pytest.raises(ValueError, match="Unsupported tracker backend"):
        make_tracker(backend="nonsense", name="bytetrack", frame_rate=8.0)
