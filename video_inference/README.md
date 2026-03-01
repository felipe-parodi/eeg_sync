# Video Inference Pipeline

This module provides an end-to-end CLI for:

1. Video compression
2. Frame extraction
3. SAM-3D-Body inference
4. Two-person temporal ID stabilization (`parent`, `child`)
5. Schema export and validation

Backends:
- `sam3d` (default)
- `ultralytics` (2D pose + temporal ID assignment)

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
- CPU mode uses safer defaults for current upstream compatibility.
- For Ultralytics + optional Roboflow trackers, install extras with `pip install .[video]`.
- Raw participant videos should remain local (`video_inference/data/` is gitignored).
