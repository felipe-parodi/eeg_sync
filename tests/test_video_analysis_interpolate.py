import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from video_analysis.interpolate import (  # noqa: E402
    InterpolationConfig,
    interpolate_camera_outputs,
)


def _build_mock_camera_dir(camera_dir: Path) -> None:
    tracks = pd.DataFrame(
        [
            {
                "frame_idx": 0,
                "timestamp_s": 0.0,
                "track_id": 0,
                "track_label": "parent",
                "bbox_x1": 0.0,
                "bbox_y1": 10.0,
                "bbox_x2": 20.0,
                "bbox_y2": 30.0,
                "track_confidence": 0.8,
            },
            {
                "frame_idx": 2,
                "timestamp_s": 2.0,
                "track_id": 0,
                "track_label": "parent",
                "bbox_x1": 2.0,
                "bbox_y1": 12.0,
                "bbox_x2": 22.0,
                "bbox_y2": 32.0,
                "track_confidence": 0.6,
            },
            {
                "frame_idx": 0,
                "timestamp_s": 0.0,
                "track_id": 1,
                "track_label": "child",
                "bbox_x1": 100.0,
                "bbox_y1": 110.0,
                "bbox_x2": 130.0,
                "bbox_y2": 160.0,
                "track_confidence": 0.9,
            },
            {
                "frame_idx": 2,
                "timestamp_s": 2.0,
                "track_id": 1,
                "track_label": "child",
                "bbox_x1": 120.0,
                "bbox_y1": 130.0,
                "bbox_x2": 150.0,
                "bbox_y2": 180.0,
                "track_confidence": 0.7,
            },
        ]
    )
    tracks.to_csv(camera_dir / "tracks_2d.csv", index=False)

    pose_rows = []
    for track_id, track_label, offset in [(0, "parent", 0.0), (1, "child", 10.0)]:
        for keypoint_name in ["kp_000", "kp_001"]:
            pose_rows.append(
                {
                    "frame_idx": 0,
                    "timestamp_s": 0.0,
                    "track_id": track_id,
                    "track_label": track_label,
                    "keypoint_name": keypoint_name,
                    "x_m": offset + 0.0,
                    "y_m": offset + 1.0,
                    "z_m": offset + 2.0,
                    "keypoint_confidence": 0.8,
                }
            )
            pose_rows.append(
                {
                    "frame_idx": 2,
                    "timestamp_s": 2.0,
                    "track_id": track_id,
                    "track_label": track_label,
                    "keypoint_name": keypoint_name,
                    "x_m": offset + 2.0,
                    "y_m": offset + 3.0,
                    "z_m": offset + 4.0,
                    "keypoint_confidence": 0.6,
                }
            )
    pd.DataFrame(pose_rows).to_csv(camera_dir / "pose_3d.csv", index=False)


def test_interpolate_camera_outputs_generates_interpolated_files(tmp_path: Path):
    camera_dir = tmp_path / "camera_a"
    camera_dir.mkdir(parents=True, exist_ok=True)
    _build_mock_camera_dir(camera_dir)

    summary = interpolate_camera_outputs(
        InterpolationConfig(camera_dir=str(camera_dir), target_fps=1.0)
    )

    tracks_interp = pd.read_csv(camera_dir / "tracks_2d_interpolated.csv")
    pose_interp = pd.read_csv(camera_dir / "pose_3d_interpolated.csv")
    summary_json = json.loads(
        (camera_dir / "interpolation_summary.json").read_text(encoding="utf-8")
    )

    assert summary["tracks_rows_output"] == len(tracks_interp)
    assert summary["pose_rows_output"] == len(pose_interp)
    assert summary_json["target_fps"] == 1.0

    # timestamps: 0, 1, 2 seconds
    assert set(tracks_interp["timestamp_s"].round(5).tolist()) == {0.0, 1.0, 2.0}
    assert len(tracks_interp) == 6  # 2 tracks * 3 timestamps

    parent_mid = tracks_interp[
        (tracks_interp["track_id"] == 0) & (tracks_interp["timestamp_s"] == 1.0)
    ].iloc[0]
    assert parent_mid["bbox_x1"] == 1.0
    assert parent_mid["track_confidence"] == 0.7

    assert len(pose_interp) == 12  # 2 tracks * 2 keypoints * 3 timestamps
    pose_mid = pose_interp[
        (pose_interp["track_id"] == 1)
        & (pose_interp["keypoint_name"] == "kp_001")
        & (pose_interp["timestamp_s"] == 1.0)
    ].iloc[0]
    assert pose_mid["x_m"] == 11.0
    assert pose_mid["keypoint_confidence"] == 0.7
