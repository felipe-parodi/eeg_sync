"""Render pose and track overlays onto extracted frames."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

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
    0: (0, 128, 255),   # orange-ish in BGR (parent)
    1: (255, 160, 0),   # blue-ish in BGR (child)
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


def render_pose_track_video(config: VisualizationConfig) -> Dict[str, object]:
    camera_dir = Path(config.camera_dir)
    inference_path = camera_dir / config.input_json
    frames_dir = camera_dir / config.frames_dir
    output_path = Path(config.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not inference_path.exists():
        raise FileNotFoundError(f"inference JSON not found: {inference_path}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames dir not found: {frames_dir}")

    payload = json.loads(inference_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
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
        "input_json": str(inference_path),
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
    )
    summary = render_pose_track_video(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

