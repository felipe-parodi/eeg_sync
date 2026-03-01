# Video Inference Pipeline

This module provides an end-to-end CLI for:

1. Video compression
2. Frame extraction
3. SAM-3D-Body inference
4. Temporal ID stabilization (`parent/child` for 2-person internal mode or `person_XX` for multi-person tracker mode)
5. Schema export and validation

Backends:
- `sam3d` (default, currently 2-person workflow)
- `ultralytics` (2D pose + configurable multi-person ID assignment)

## Rapid pre-compression (recommended for CPU runs)

```bash
video-compress-rapid \
  --video video_inference/data/camera_a_raw.mov \
  --video video_inference/data/camera_b_raw.mov \
  --output-dir video_inference/compressed
```

## Command

```bash
video-infer run \
  --video-a video_inference/compressed/camera_a_raw_rapid.mp4 \
  --video-b video_inference/compressed/camera_b_raw_rapid.mp4 \
  --checkpoint-path /path/to/model.ckpt \
  --mhr-path /path/to/mhr_model.pt \
  --output-dir video_inference/output \
  --device cpu \
  --skip-compress
```

Ultralytics backend example:

```bash
video-infer run \
  --video-a video_inference/compressed/camera_a_raw_rapid.mp4 \
  --video-b video_inference/compressed/camera_b_raw_rapid.mp4 \
  --inference-backend ultralytics \
  --ultralytics-model-path yolo11n-pose.pt \
  --tracker-backend internal \
  --device cpu \
  --skip-compress
```

Ultralytics + Roboflow tracker (ByteTrack) example:

```bash
video-infer run \
  --video-a video_inference/compressed/camera_a_raw_rapid.mp4 \
  --video-b video_inference/compressed/camera_b_raw_rapid.mp4 \
  --inference-backend ultralytics \
  --ultralytics-model-path yolo11n-pose.pt \
  --tracker-backend roboflow \
  --tracker-name bytetrack \
  --device auto \
  --skip-compress
```

Two-camera, 4-person workflow (recommended for your current videos):

```bash
video-infer run \
  --video-a video_inference/compressed/camera_a_raw_rapid.mp4 \
  --video-b video_inference/compressed/camera_b_raw_rapid.mp4 \
  --inference-backend ultralytics \
  --ultralytics-model-path video_inference/output/models/yolo11n-pose.pt \
  --tracker-backend roboflow \
  --tracker-name bytetrack \
  --max-persons 4 \
  --frame-rate 1 \
  --device cpu \
  --skip-compress
```

Low-FPS inference (recommended for fast demos and CPU):

```bash
video-infer run \
  --video-a video_inference/compressed/camera_a_raw_rapid.mp4 \
  --inference-backend ultralytics \
  --ultralytics-model-path video_inference/output/models/yolo11n-pose.pt \
  --tracker-backend internal \
  --device cpu \
  --frame-rate 1 \
  --skip-compress
```

Then interpolate outputs to higher FPS:

```bash
video-interpolate \
  --camera-dir video_inference/output/<session_id>/camera_a \
  --target-fps 8
```

## Output structure

For each camera, the pipeline writes:

- `manifest.json`
- `tracks_2d.csv`
- `pose_3d.csv`
- `frames/frame_*.jpg`
- `intermediate/inference_raw.json`

Session-level summary:

- `session_summary.json`

## Notes

- `--device auto` prefers CUDA and falls back to CPU.
- For pre-compressed inputs, add `--skip-compress` to avoid recompression.
- `--frame-rate` controls inference sampling rate; use `1` for lightweight runs.
- `--max-persons` sets expected number of tracked individuals.
- `--tracker-backend roboflow --tracker-name bytetrack` enables external MOT tracking.
- `--enforce-exact-person-count` can be enabled if you want only frames with exactly N detected people.
- CPU mode uses safer defaults for current upstream compatibility.
- For Ultralytics + optional Roboflow trackers, install extras with `pip install .[video]`.
- Raw participant videos should remain local (`video_inference/data/` is gitignored).
