import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def test_render_pose_track_video_writes_output(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    from video_analysis.visualize_pose_tracks import (  # noqa: E402
        VisualizationConfig,
        render_pose_track_video,
    )

    camera_dir = tmp_path / "camera_a"
    frames_dir = camera_dir / "frames"
    intermediate_dir = camera_dir / "intermediate"
    frames_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "frame_000001.jpg"), image)
    cv2.imwrite(str(frames_dir / "frame_000002.jpg"), image)

    payload = {
        "frames": [
            {
                "frame_idx": 0,
                "image_name": "frame_000001.jpg",
                "persons": [
                    {
                        "track_id": 0,
                        "track_label": "parent",
                        "bbox_xyxy": [20, 20, 90, 100],
                        "confidence": 0.9,
                        "keypoints_3d": [[30, 30, 0, 0.9], [40, 40, 0, 0.9]],
                    }
                ],
            },
            {
                "frame_idx": 1,
                "image_name": "frame_000002.jpg",
                "persons": [
                    {
                        "track_id": 1,
                        "track_label": "child",
                        "bbox_xyxy": [40, 25, 110, 105],
                        "confidence": 0.8,
                        "keypoints_3d": [[50, 30, 0, 0.8], [60, 45, 0, 0.8]],
                    }
                ],
            },
        ]
    }
    (intermediate_dir / "inference_raw.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    output_video = camera_dir / "pose_tracks_preview.mp4"
    summary = render_pose_track_video(
        VisualizationConfig(
            camera_dir=str(camera_dir),
            output_video=str(output_video),
            output_fps=5.0,
        )
    )

    assert summary["rendered_frames"] == 2
    assert output_video.exists()
    assert output_video.stat().st_size > 0
