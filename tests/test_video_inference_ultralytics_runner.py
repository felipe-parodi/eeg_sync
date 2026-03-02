import sys
import types
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_inference.ultralytics_runner import (  # noqa: E402
    UltraRunnerConfig,
    _assign_from_role_indices,
    _assign_output_track_slot,
    _resolve_role_indices_from_tracker_ids,
    run_ultralytics_pose_on_images,
)


def _fake_detections():
    return [
        {"bbox": np.asarray([10, 20, 60, 100], dtype=float), "confidence": 0.9},
        {"bbox": np.asarray([100, 30, 220, 260], dtype=float), "confidence": 0.95},
    ]


def test_resolve_role_indices_initializes_with_area_prior():
    detections = _fake_detections()
    role_map = {}

    parent_idx, child_idx = _resolve_role_indices_from_tracker_ids(
        detections=detections,
        tracker_ids=[11, 22],
        role_by_tracker_id=role_map,
    )

    assert parent_idx == 1
    assert child_idx == 0
    assert role_map[22] == 0
    assert role_map[11] == 1


def test_resolve_role_indices_uses_existing_tracker_role_map():
    detections = _fake_detections()
    role_map = {11: 1, 22: 0}

    parent_idx, child_idx = _resolve_role_indices_from_tracker_ids(
        detections=detections,
        tracker_ids=[11, 22],
        role_by_tracker_id=role_map,
    )

    assert parent_idx == 1
    assert child_idx == 0


def test_resolve_role_indices_handles_partial_known_ids():
    detections = _fake_detections()
    role_map = {99: 0}

    parent_idx, child_idx = _resolve_role_indices_from_tracker_ids(
        detections=detections,
        tracker_ids=[88, 99],
        role_by_tracker_id=role_map,
    )

    assert parent_idx == 1
    assert child_idx == 0
    assert role_map[99] == 0
    assert role_map[88] == 1


def test_assign_from_role_indices_preserves_detection_indices():
    detections = _fake_detections()
    assigned = _assign_from_role_indices(
        detections=detections,
        parent_idx=1,
        child_idx=0,
    )

    parent = assigned[0]
    child = assigned[1]
    assert parent.track_label == "parent"
    assert child.track_label == "child"
    assert parent.detection_index == 1
    assert child.detection_index == 0


def test_assign_output_track_slot_reuses_existing_slot_for_same_source():
    source_to_output = {100: 0}
    output_to_source = {0: 100}
    last_seen = {0: 4}
    active_slots = set()

    output_slot = _assign_output_track_slot(
        source_track_id=100,
        frame_idx=5,
        max_persons=4,
        source_to_output=source_to_output,
        output_to_source=output_to_source,
        last_seen_by_output=last_seen,
        active_output_slots=active_slots,
    )

    assert output_slot == 0
    assert source_to_output == {100: 0}
    assert output_to_source == {0: 100}
    assert last_seen[0] == 5
    assert active_slots == {0}


def test_assign_output_track_slot_recycles_oldest_inactive_slot():
    source_to_output = {10: 0, 11: 1}
    output_to_source = {0: 10, 1: 11}
    last_seen = {0: 20, 1: 5}
    active_slots = {0}

    output_slot = _assign_output_track_slot(
        source_track_id=12,
        frame_idx=21,
        max_persons=2,
        source_to_output=source_to_output,
        output_to_source=output_to_source,
        last_seen_by_output=last_seen,
        active_output_slots=active_slots,
    )

    assert output_slot == 1
    assert source_to_output == {10: 0, 12: 1}
    assert output_to_source == {0: 10, 1: 12}
    assert last_seen[1] == 21


class _FakeTensor:
    def __init__(self, data):
        self._data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensor([[10.0, 10.0, 50.0, 60.0]])
        self.conf = _FakeTensor([0.95])
        self.cls = _FakeTensor([0])


class _FakeKeypoints:
    def __init__(self):
        self.xy = _FakeTensor(np.zeros((1, 17, 2), dtype=float))
        self.conf = _FakeTensor(np.ones((1, 17), dtype=float))


class _FakeYOLO:
    def __init__(self, _model_path: str):
        pass

    def __call__(self, source, *_args, **_kwargs):
        # Support both single-image and batch calls.
        n = len(source) if isinstance(source, list) else 1
        return [
            types.SimpleNamespace(
                boxes=_FakeBoxes(),
                keypoints=_FakeKeypoints(),
            )
            for _ in range(n)
        ]


class _FakeDetections:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeByteTrack:
    def __init__(self):
        self._next_id = 0

    def update_with_detections(self, _detections):
        tracker_id = np.asarray([self._next_id], dtype=int)
        self._next_id += 1
        return types.SimpleNamespace(tracker_id=tracker_id)


def test_run_ultralytics_pose_recycles_slots_for_new_tracker_ids(tmp_path, monkeypatch):
    image_folder = tmp_path / "frames"
    image_folder.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(6):
        (image_folder / f"frame_{frame_idx:06d}.jpg").write_bytes(b"x")

    output_json = tmp_path / "inference_raw.json"

    fake_ultralytics = types.SimpleNamespace(YOLO=_FakeYOLO)
    fake_supervision = types.SimpleNamespace(
        Detections=_FakeDetections,
        ByteTrack=_FakeByteTrack,
    )
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setitem(sys.modules, "supervision", fake_supervision)

    cfg = UltraRunnerConfig(
        model_path="fake.pt",
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
        conf=0.1,
        iou=0.7,
        tracker_backend="roboflow",
        tracker_name="bytetrack",
        max_persons=2,
        keep_empty_frames=True,
    )
    payload = run_ultralytics_pose_on_images(cfg)

    assert len(payload["frames"]) == 6
    persons_per_frame = [len(frame["persons"]) for frame in payload["frames"]]
    assert persons_per_frame == [1, 1, 1, 1, 1, 1]
    for frame in payload["frames"]:
        assert frame["persons"][0]["track_id"] in {0, 1}
