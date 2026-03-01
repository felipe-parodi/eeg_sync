# Video Inference Pipeline

This module provides an end-to-end CLI for:

1. Video compression
2. Frame extraction
3. SAM-3D-Body inference
4. Two-person temporal ID stabilization (`parent`, `child`)
5. Schema export and validation

## Command

```bash
video-infer run \
  --video-a video_inference/data/P001c_Short_Full.mov \
  --video-b video_inference/data/P001c_Tall_Full.mov \
  --checkpoint-path /path/to/model.ckpt \
  --mhr-path /path/to/mhr_model.pt \
  --output-dir video_inference/output \
  --device auto
```

## Output structure

For each camera, the pipeline writes:

- `manifest.json`
- `tracks_2d.csv`
- `pose_3d.csv`
- `frames/frame_*.jpg`
- `intermediate/sam3d_raw.json`

Session-level summary:

- `session_summary.json`

## Notes

- `--device auto` prefers CUDA and falls back to CPU.
- CPU mode uses safer defaults for current upstream compatibility.
- Raw participant videos should remain local (`video_inference/data/` is gitignored).
