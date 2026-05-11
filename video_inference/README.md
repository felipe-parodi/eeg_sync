# video_inference

Pose inference pipeline — frame extraction, model inference, temporal tracking, and schema export. The CLI here is `video-infer run`; the full stage-by-stage reference lives in [`../docs/PIPELINE.md`](../docs/PIPELINE.md).

For an end-to-end colleague-facing walkthrough including the post-inference cleanup steps (`video-filter-tracks`, `video-smooth`, metrics), read [`../docs/COLLEAGUE_QUICKSTART.md`](../docs/COLLEAGUE_QUICKSTART.md).

## Backends

| Backend | Flag | Output dimensions | Notes |
|---|---|---|---|
| Ultralytics YOLOv11-pose | `--inference-backend ultralytics` | 2D (`z_m=0.0`) | **Default for the colleague workflow.** CPU-friendly. |
| RTMLib | `--inference-backend rtmlib` (`--rtmlib-3d` for 3D) | 2D Body or 3D Wholebody | onnxruntime / opencv / openvino execution providers. |
| SAM-3D-Body | `--inference-backend sam3d` | True 3D in metres | Slower; requires the `third_party/sam-3d-body` submodule. |

## Tracker modes

| Tracker | Behaviour |
|---|---|
| `--tracker-backend internal` | Two-person heuristic with IoU continuity. Only suitable when `--max-persons 2` and detections are clean. |
| `--tracker-backend roboflow --tracker-name bytetrack` | ByteTrack with `track_buffer=60` frames. Recommended for `--max-persons > 2` and any session with possible occlusions. |

The output ID slot is bounded by `--max-persons`; ByteTrack's internal IDs are recycled into stable `0..max_persons-1` slots.

## Rapid pre-compression

```bash
video-compress-rapid \
  --video video_inference/data/camera_a_raw.mov \
  --video video_inference/data/camera_b_raw.mov \
  --output-dir video_inference/compressed
```

Always run this before `video-infer` unless your inputs are already compressed; raw GoPro files are 2–10 GB.

## End-to-end command (recommended for the colleague workflow)

```bash
video-infer run \
  --video-a video_inference/compressed/camera_a_raw_rapid.mp4 \
  --video-b video_inference/compressed/camera_b_raw_rapid.mp4 \
  --inference-backend ultralytics \
  --ultralytics-model-path yolo11m-pose.pt \
  --tracker-backend roboflow --tracker-name bytetrack \
  --max-persons 4 \
  --frame-rate 5 \
  --device auto \
  --skip-compress \
  --session-id <SESSION_ID>
```

Set `--max-persons 4` (not 2) when an experimenter may briefly enter frame — the extra tracks are pruned to parent + child afterwards by `video-filter-tracks`.

## Backend-specific examples

### SAM-3D (true 3D)
```bash
video-infer run \
  --video-a video_inference/compressed/cam_a.mp4 \
  --inference-backend sam3d \
  --checkpoint-path <path/to/sam3d.ckpt> \
  --mhr-path        <path/to/mhr.pt> \
  --device cpu --skip-compress
```

### RTMLib 3D
```bash
video-infer run \
  --video-a video_inference/compressed/cam_a.mp4 \
  --inference-backend rtmlib --rtmlib-3d \
  --rtmlib-backend onnxruntime --rtmlib-mode balanced \
  --device auto --skip-compress
```

### Ultralytics internal 2-person tracker
```bash
video-infer run \
  --video-a video_inference/compressed/cam_a.mp4 \
  --inference-backend ultralytics \
  --ultralytics-model-path yolo11n-pose.pt \
  --tracker-backend internal \
  --max-persons 2 \
  --device cpu --skip-compress
```

## Per-camera outputs

```
<output-dir>/<session_id>/<camera>/
  manifest.json
  tracks_2d.csv
  pose_3d.csv
  frames/frame_*.jpg
  intermediate/inference_raw.json
```

Plus session-level `session_summary.json`. Schemas and column dictionaries: [`../docs/OUTPUTS.md`](../docs/OUTPUTS.md).

## Notes

- `--device auto` prefers CUDA → MPS → CPU.
- `--frame-rate` controls inference sampling rate; `1` for lightweight previews, `5` is the recommended default.
- `--enforce-exact-person-count` drops frames that don't have exactly `--max-persons` detections (off by default).
- Raw participant videos must stay local (`video_inference/data/` is gitignored).
- After `video-infer run`, run `video-filter-tracks` to prune to parent + child and assign stable IDs. See [`../docs/PIPELINE.md`](../docs/PIPELINE.md#stage-3--video-filter-tracks).
