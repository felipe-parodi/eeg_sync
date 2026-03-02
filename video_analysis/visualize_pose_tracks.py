"""Render pose and track overlays onto extracted frames."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import pandas as pd

COCO17_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

TRACK_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 128, 255),   # orange in BGR (person_00)
    1: (255, 160, 0),   # blue in BGR (person_01)
    2: (0, 220, 0),     # green in BGR (person_02)
    3: (180, 0, 255),   # purple in BGR (person_03)
}


@dataclass
class VisualizationConfig:
    """Configuration for pose+track overlay rendering."""

    camera_dir: str
    output_video: str
    input_json: str = "intermediate/inference_raw.json"
    frames_dir: str = "frames"
    output_fps: float = 12.0
    keypoint_conf_thresh: float = 0.2
    max_frames: Optional[int] = None
    start_frame: int = 0
    draw_bbox: bool = True
    pose_csv: Optional[str] = None
    tracks_csv: Optional[str] = None


def _color_for_track(track_id: int) -> Tuple[int, int, int]:
    return TRACK_COLORS.get(track_id, (180, 180, 180))


def _draw_person(
    image,
    person: Dict[str, object],
    keypoint_conf_thresh: float,
    draw_bbox: bool,
) -> None:
    track_id = int(person.get("track_id", -1))
    track_label = str(person.get("track_label", "unknown"))
    color = _color_for_track(track_id)

    bbox = person.get("bbox_xyxy", [])
    if draw_bbox and isinstance(bbox, Sequence) and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(float(value)) for value in bbox[:4]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{track_label} id={track_id}"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    keypoints = person.get("keypoints_3d", [])
    parsed_points: List[Optional[Tuple[int, int, float]]] = []
    for keypoint in keypoints:
        if not isinstance(keypoint, Sequence) or len(keypoint) < 2:
            parsed_points.append(None)
            continue
        x = int(float(keypoint[0]))
        y = int(float(keypoint[1]))
        conf = float(keypoint[3]) if len(keypoint) >= 4 else 1.0
        parsed_points.append((x, y, conf))

    for edge_start, edge_end in COCO17_EDGES:
        if edge_start >= len(parsed_points) or edge_end >= len(parsed_points):
            continue
        point_a = parsed_points[edge_start]
        point_b = parsed_points[edge_end]
        if point_a is None or point_b is None:
            continue
        if point_a[2] < keypoint_conf_thresh or point_b[2] < keypoint_conf_thresh:
            continue
        cv2.line(image, (point_a[0], point_a[1]), (point_b[0], point_b[1]), color, 2)

    for point in parsed_points:
        if point is None or point[2] < keypoint_conf_thresh:
            continue
        cv2.circle(image, (point[0], point[1]), 3, color, thickness=-1, lineType=cv2.LINE_AA)


def _load_frames_from_csv(
    pose_path: Path,
    tracks_path: Path,
    frames_dir: Path,
) -> List[Dict[str, object]]:
    """Build frame entries from pose + tracks CSVs (same format as JSON loader).

    Uses frame_index.csv (written by the frame extractor) to correctly map
    CSV timestamps to physical frame images. This is necessary because
    interpolation renumbers frame_idx, breaking the original file numbering.
    """
    import numpy as np

    pose_df = pd.read_csv(pose_path)
    tracks_df = pd.read_csv(tracks_path)

    # Build timestamp → image_name map from frame_index.csv.
    frame_index_path = frames_dir / "frame_index.csv"
    if frame_index_path.exists():
        frame_index = pd.read_csv(frame_index_path)
        frame_timestamps = frame_index["timestamp_s"].values.astype(float)
        frame_image_names = frame_index["image_name"].values
    else:
        # Fallback: assume 1-indexed files match frame_idx directly.
        available = sorted(frames_dir.glob("frame_*.jpg"))
        frame_timestamps = np.arange(len(available), dtype=float)
        frame_image_names = np.array([p.name for p in available])

    # For each frame image timestamp, find the nearest CSV timestamp.
    csv_timestamps = sorted(tracks_df["timestamp_s"].unique())
    csv_ts_array = np.array(csv_timestamps, dtype=float)

    frames: List[Dict[str, object]] = []
    for img_ts, img_name in zip(frame_timestamps, frame_image_names):
        if len(csv_ts_array) == 0:
            break
        nearest_idx = int(np.argmin(np.abs(csv_ts_array - img_ts)))
        nearest_ts = csv_ts_array[nearest_idx]
        # Skip if nearest CSV timestamp is too far (> 0.5s).
        if abs(nearest_ts - img_ts) > 0.5:
            continue

        t_rows = tracks_df[tracks_df["timestamp_s"] == nearest_ts]
        p_rows = pose_df[pose_df["timestamp_s"] == nearest_ts]

        persons: List[Dict[str, object]] = []
        for _, t_row in t_rows.iterrows():
            tid = int(t_row["track_id"])
            person_pose = p_rows[p_rows["track_id"] == tid].sort_values("keypoint_name")
            keypoints_3d = [
                [float(r["x_m"]), float(r["y_m"]), float(r["z_m"]), float(r["keypoint_confidence"])]
                for _, r in person_pose.iterrows()
            ]
            persons.append(
                {
                    "track_id": tid,
                    "track_label": str(t_row["track_label"]),
                    "bbox_xyxy": [
                        float(t_row["bbox_x1"]),
                        float(t_row["bbox_y1"]),
                        float(t_row["bbox_x2"]),
                        float(t_row["bbox_y2"]),
                    ],
                    "confidence": float(t_row["track_confidence"]),
                    "keypoints_3d": keypoints_3d,
                }
            )

        frames.append(
            {"frame_idx": len(frames), "image_name": str(img_name), "persons": persons}
        )

    return frames


def render_pose_track_video(config: VisualizationConfig) -> Dict[str, object]:
    camera_dir = Path(config.camera_dir)
    inference_path = camera_dir / config.input_json
    frames_dir = camera_dir / config.frames_dir
    output_path = Path(config.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        raise FileNotFoundError(f"frames dir not found: {frames_dir}")

    if config.pose_csv and config.tracks_csv:
        pose_path = camera_dir / config.pose_csv
        tracks_path = camera_dir / config.tracks_csv
        if not pose_path.exists():
            raise FileNotFoundError(f"pose CSV not found: {pose_path}")
        if not tracks_path.exists():
            raise FileNotFoundError(f"tracks CSV not found: {tracks_path}")
        frames = _load_frames_from_csv(pose_path, tracks_path, frames_dir)
        source_label = f"{config.pose_csv} + {config.tracks_csv}"
    else:
        if not inference_path.exists():
            raise FileNotFoundError(f"inference JSON not found: {inference_path}")
        payload = json.loads(inference_path.read_text(encoding="utf-8"))
        frames = payload.get("frames", [])
        source_label = str(inference_path)
    if config.start_frame > 0:
        frames = frames[config.start_frame :]
    if config.max_frames is not None:
        frames = frames[: config.max_frames]
    if not frames:
        raise ValueError("No frames available for rendering.")

    first_image_name = frames[0].get("image_name")
    first_image = cv2.imread(str(frames_dir / str(first_image_name)))
    if first_image is None:
        raise FileNotFoundError(
            f"Could not load first frame image: {frames_dir / str(first_image_name)}"
        )
    height, width = first_image.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        config.output_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at: {output_path}")

    rendered = 0
    skipped = 0
    for frame_entry in frames:
        image_name = frame_entry.get("image_name")
        frame_path = frames_dir / str(image_name)
        image = cv2.imread(str(frame_path))
        if image is None:
            skipped += 1
            continue

        for person in frame_entry.get("persons", []):
            _draw_person(
                image=image,
                person=person,
                keypoint_conf_thresh=config.keypoint_conf_thresh,
                draw_bbox=config.draw_bbox,
            )

        cv2.putText(
            image,
            f"frame={frame_entry.get('frame_idx', '?')} image={image_name}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(image)
        rendered += 1

        if rendered % 200 == 0:
            print(f"[video-visualize] rendered {rendered} frames")

    writer.release()
    summary = {
        "camera_dir": str(camera_dir),
        "source": source_label,
        "output_video": str(output_path),
        "rendered_frames": rendered,
        "skipped_frames": skipped,
        "output_fps": config.output_fps,
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-visualize",
        description="Render track IDs and 2D poses onto extracted frames.",
    )
    parser.add_argument("--camera-dir", required=True, type=str)
    parser.add_argument("--output-video", required=True, type=str)
    parser.add_argument("--input-json", default="intermediate/inference_raw.json", type=str)
    parser.add_argument("--frames-dir", default="frames", type=str)
    parser.add_argument("--output-fps", default=12.0, type=float)
    parser.add_argument("--keypoint-conf-thresh", default=0.2, type=float)
    parser.add_argument("--max-frames", default=None, type=int)
    parser.add_argument("--start-frame", default=0, type=int)
    parser.add_argument("--draw-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pose-csv", default=None, type=str,
                        help="Pose CSV file (relative to camera-dir) to use instead of JSON.")
    parser.add_argument("--tracks-csv", default=None, type=str,
                        help="Tracks CSV file (relative to camera-dir) to use instead of JSON.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = VisualizationConfig(
        camera_dir=args.camera_dir,
        output_video=args.output_video,
        input_json=args.input_json,
        frames_dir=args.frames_dir,
        output_fps=args.output_fps,
        keypoint_conf_thresh=args.keypoint_conf_thresh,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        draw_bbox=args.draw_bbox,
        pose_csv=args.pose_csv,
        tracks_csv=args.tracks_csv,
    )
    summary = render_pose_track_video(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

