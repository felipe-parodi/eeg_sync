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
                orig_img=np.zeros((480, 640, 3), dtype=np.uint8),
            )
            for _ in range(n)
        ]


class _FakeDetections:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeByteTrack:
    def __init__(self, **kwargs):
        self._next_id = 0
        self.kwargs = kwargs

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


def test_run_ultralytics_pose_forwards_frame_rate_to_bytetrack(tmp_path, monkeypatch):
    """ByteTrack's lost_track_buffer is interpreted via frame_rate/30. If
    we don't forward our actual fps, an 8 fps job gets 7.5 s of tolerance
    instead of 2 s, encouraging cross-person ID re-association during
    occlusion."""
    image_folder = tmp_path / "frames"
    image_folder.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(2):
        (image_folder / f"frame_{frame_idx:06d}.jpg").write_bytes(b"x")
    output_json = tmp_path / "inference_raw.json"

    seen_kwargs: list[dict] = []

    class _CapturingByteTrack(_FakeByteTrack):
        def __init__(self, **kwargs):
            seen_kwargs.append(kwargs)
            super().__init__(**kwargs)

    fake_ultralytics = types.SimpleNamespace(YOLO=_FakeYOLO)
    fake_supervision = types.SimpleNamespace(
        Detections=_FakeDetections,
        ByteTrack=_CapturingByteTrack,
    )
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setitem(sys.modules, "supervision", fake_supervision)

    cfg = UltraRunnerConfig(
        model_path="fake.pt",
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
        conf=0.1,
        tracker_backend="roboflow",
        tracker_name="bytetrack",
        max_persons=2,
        frame_rate=8.0,
    )
    run_ultralytics_pose_on_images(cfg)

    assert len(seen_kwargs) == 1
    assert seen_kwargs[0].get("frame_rate") == 8.0
    assert seen_kwargs[0].get("lost_track_buffer") == 60


def test_run_ultralytics_pose_accepts_boxmot_tracker_backend(tmp_path, monkeypatch):
    """Regression: the runner's upstream validation must list 'boxmot' as a
    valid tracker_backend so the portal default (boxmot+botsort) doesn't
    blow up before the BoT-SORT adapter ever runs."""
    image_folder = tmp_path / "frames"
    image_folder.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(2):
        (image_folder / f"frame_{frame_idx:06d}.jpg").write_bytes(b"x")
    output_json = tmp_path / "inference_raw.json"

    # Fake boxmot tracker that mimics the adapter contract: returns a
    # TrackResults-shaped (N,8) array with det_ind back-mapping.
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

    class _FakeBotSort:
        next_id = 100

        def __init__(self, **kwargs):
            pass

        def update(self, dets, img):
            rows = []
            for i in range(len(dets)):
                rows.append([0, 0, 1, 1, _FakeBotSort.next_id + i, 0.9, 0, i])
            _FakeBotSort.next_id += len(dets)
            return _TrackResults(rows)

    class _FakeReIDBackend:
        def warmup(self):
            pass

    class _FakeReID:
        # Real boxmot ReID exposes the embedding backend on ``.model``; the
        # tracker holds the backend, not the wrapper.
        def __init__(self, **kwargs):
            self.model = _FakeReIDBackend()

    botsort_mod = types.ModuleType("boxmot.trackers.botsort.botsort")
    botsort_mod.BotSort = _FakeBotSort
    reid_mod = types.ModuleType("boxmot.reid.core.reid")
    reid_mod.ReID = _FakeReID
    monkeypatch.setitem(sys.modules, "boxmot.trackers.botsort.botsort", botsort_mod)
    monkeypatch.setitem(sys.modules, "boxmot.reid.core.reid", reid_mod)
    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO)
    )

    cfg = UltraRunnerConfig(
        model_path="fake.pt",
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
        conf=0.1,
        tracker_backend="boxmot",
        tracker_name="botsort",
        max_persons=3,
        frame_rate=8.0,
    )
    payload = run_ultralytics_pose_on_images(cfg)

    assert len(payload["frames"]) == 2
    # Every frame got a person with a track id from the fake BoT-SORT.
    for frame in payload["frames"]:
        assert len(frame["persons"]) >= 1


def test_run_ultralytics_pose_rejects_unknown_tracker_backend(tmp_path, monkeypatch):
    """Validation error mentions all three valid options. ultralytics is
    stubbed because CI's dev extra doesn't install it, but the validation
    we're checking runs immediately after the import so a stub is enough."""
    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO)
    )
    cfg = UltraRunnerConfig(
        model_path="fake.pt",
        image_folder=str(tmp_path),
        output_json=str(tmp_path / "out.json"),
        tracker_backend="nonsense",
        tracker_name="bytetrack",
        max_persons=2,
    )
    import pytest

    with pytest.raises(ValueError, match="boxmot"):
        run_ultralytics_pose_on_images(cfg)


def test_run_ultralytics_pose_offline_backend_writes_cache_and_roles(
    tmp_path, monkeypatch
):
    """The offline backend caches detections next to the raw JSON and emits
    parent/child persons; the payload must be JSON-serializable."""
    import json

    from video_inference import detection_cache
    from video_inference.detection_cache import DetCache

    image_folder = tmp_path / "frames"
    image_folder.mkdir()
    n = 6
    for i in range(n):
        (image_folder / f"frame_{i + 1:06d}.jpg").write_bytes(b"x")
    (image_folder / "frame_index.csv").write_text(
        "frame_idx,timestamp_s,image_name\n"
        + "".join(f"{i},{100 + i / 6:.6f},frame_{i + 1:06d}.jpg\n" for i in range(n)),
        encoding="utf-8",
    )

    def fake_build(image_paths, frame_idx, timestamp_s, **kwargs):
        boxes, kps = [], []
        for _ in image_paths:
            for x, h in ((100.0, 300.0), (500.0, 140.0)):
                boxes.append([x, 100.0, x + h / 2, 100.0 + h])
                k = np.ones((17, 3), np.float32)
                k[:, 0] = x + np.linspace(0, h / 2, 17)
                k[:, 1] = 100.0 + np.linspace(0, h, 17)
                kps.append(k)
        emb = np.tile(np.eye(4, dtype=np.float32)[:2], (len(image_paths), 1))
        return DetCache(
            frame_idx=np.asarray(frame_idx),
            timestamp_s=np.asarray(timestamp_s),
            offsets=np.arange(0, 2 * len(image_paths) + 1, 2),
            boxes=np.asarray(boxes, np.float32),
            conf=np.full(len(boxes), 0.9, np.float32),
            keypoints=np.asarray(kps),
            embeddings=emb,
            image_size=(960, 540),
        )

    monkeypatch.setattr(detection_cache, "build_detection_cache", fake_build)
    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO)
    )
    output_json = tmp_path / "intermediate" / "inference_raw.json"
    cfg = UltraRunnerConfig(
        model_path="fake.pt",
        image_folder=str(image_folder),
        output_json=str(output_json),
        device="cpu",
        tracker_backend="offline",
        max_persons=2,
        frame_rate=6.0,
    )
    payload = run_ultralytics_pose_on_images(cfg)

    assert (output_json.parent / "detections.npz").exists()
    json.loads(output_json.read_text(encoding="utf-8"))
    assert len(payload["frames"]) == n
    for frame in payload["frames"]:
        labels = {p["track_label"]: p for p in frame["persons"]}
        assert set(labels) == {"parent", "child"}
        assert labels["parent"]["track_id"] == 0
        assert labels["parent"]["bbox_xyxy"][0] == 100.0  # the taller person
        assert len(labels["child"]["keypoints_3d"]) == 17
