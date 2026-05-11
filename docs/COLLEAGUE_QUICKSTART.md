# Quickstart: video → analyzable outputs

**Audience**: research collaborators who have session videos and want pose/gaze metrics out.
**You will**: install once, then for each session run six commands. End result: per-block plots, CSVs you can analyze in Python/R/Excel, and overlay videos you can sanity-check.

If you're driving this with an LLM (Codex, Claude, etc.), point it at this file plus `docs/PIPELINE.md` and `docs/OUTPUTS.md`. Both are written to be machine-readable.

---

## 0. Before you start: input contract

Three things have to be true **before** the pipeline will produce trustworthy metrics:

1. **Exactly two people in frame during analyzed segments.** The pipeline assumes parent + child. If an experimenter steps in, that's a third track and metrics like "torso proximity" will mix them in. Two ways to handle it:
   - Easy: exclude that window from your segment list (see step 2 below).
   - If you can't exclude it: keep the window but log it in `exclusion_windows.csv`; you'll relabel the extra person to a non-parent/child track with `video-annotate-tracks` later.
2. **Per-session segment timings.** For each video, the start/end times of the segments you want analyzed, in `MM:SS` (e.g. `"Block 1: 13:26–23:40"`). Template: [`templates/segment_timings.example.csv`](../templates/segment_timings.example.csv).
3. **Exclusion windows, if any** — moments when someone other than parent + child was visible. Template: [`templates/exclusion_windows.example.csv`](../templates/exclusion_windows.example.csv).

Both CSVs are **inputs to you, the analyst**, not inputs the CLIs read directly. You'll transcribe them into `session_config.json` and the `--blocks` argument below.

---

## 1. Install (once per machine)

```bash
git clone https://github.com/felipe-parodi/eeg_sync.git
cd eeg_sync
pip install -e ".[dev,video,gaze]"
```

That installs everything: core, video inference (Ultralytics), gaze (Gazelle dependencies), and dev tools. If you only need video pose (no gaze), `pip install -e ".[video]"` is enough.

ffmpeg must be on `PATH`. If `which ffmpeg` is empty, `brew install ffmpeg` (macOS) or `conda env create -f environment.yml` (cross-platform).

CPU is the default and is fine. The pipeline auto-detects CUDA / Apple Silicon if available.

---

## 2. Set up the session

For each session, do this once:

**a. Drop the raw videos in `video_inference/data/`.** They stay local — that folder is gitignored. Two cameras supported (`video-a`, `video-b`); one camera is fine too.

**b. Copy `templates/session_config.example.json` to `session_config.json`** at repo root, then edit:

```json
{
  "session_id": "P001c",
  "output_dir": "video_inference/output/P001c",
  "image_width": 854,
  "image_height": 480,
  "frame_rate": 5.0,
  "session_blocks": [
    {"name": "grocery_free_play",      "start_s":  806, "end_s": 1420, "color": "green"},
    {"name": "synchrony_intervention", "start_s": 1676, "end_s": 1725, "color": "orange"},
    {"name": "storybook_reading",      "start_s": 1762, "end_s": 2226, "color": "blue"}
  ],
  "camera_mappings": [
    {"camera_id": "camera_a", "parent_track_id": 0, "child_track_id": 1},
    {"camera_id": "camera_b", "parent_track_id": 0, "child_track_id": 1}
  ]
}
```

- `session_blocks.start_s` / `end_s` are **seconds from the start of the video**. Convert `MM:SS` to seconds by hand or use the snippet in [`templates/README.md`](../templates/README.md).
- You can ignore `camera_mappings` if you plan to run `video-filter-tracks` in step 5 — it auto-assigns parent/child.

---

## 3. Compress (always)

Raw GoPro files are 2–10 GB and will make every downstream step painful. Compress first to ≤50 MB.

```bash
video-compress-rapid \
  --video video_inference/data/camera_a_raw.mov \
  --video video_inference/data/camera_b_raw.mov \
  --output-dir video_inference/compressed
```

Output: `video_inference/compressed/<name>_rapid.mp4`. Use those paths everywhere downstream and add `--skip-compress` to the next step.

---

## 4. Run pose inference

```bash
video-infer run \
  --video-a video_inference/compressed/camera_a_raw_rapid.mp4 \
  --video-b video_inference/compressed/camera_b_raw_rapid.mp4 \
  --inference-backend ultralytics \
  --ultralytics-model-path yolo11m-pose.pt \
  --tracker-backend roboflow \
  --tracker-name bytetrack \
  --max-persons 4 \
  --frame-rate 5 \
  --device auto \
  --skip-compress \
  --session-id P001c
```

What this does: extracts frames at 5 fps, detects up to 4 people per frame with YOLO11m-pose, and tracks them across frames with ByteTrack. We allow up to 4 detected people (`--max-persons 4`) so the pipeline doesn't drop the parent/child if an experimenter briefly enters; you'll prune to two in step 5.

Outputs per camera in `video_inference/output/P001c/<camera>/`:
- `manifest.json`
- `tracks_2d.csv` (bounding boxes + track IDs)
- `pose_3d.csv` (17 COCO keypoints per person per frame)
- `frames/frame_*.jpg`
- `intermediate/inference_raw.json`

If the run takes too long on CPU, drop to `--frame-rate 1` for a fast preview pass.

---

## 5. Clean up tracks (the "experimenter walked in" step)

Most of the time `video-filter-tracks` is all you need. It keeps the two most persistent tracks within each block and assigns parent=0 / child=1 by mean bbox area:

```bash
video-filter-tracks \
  --session-dir video_inference/output/P001c \
  --camera camera_a \
  --blocks "grocery_free_play,13:26,23:40;synchrony_intervention,27:56,28:45;storybook_reading,29:22,37:06" \
  --source-fps 5.0 \
  --n-keep 2
```

`--source-fps` must match the `--frame-rate` you used in step 4.

Outputs: `tracks_2d_filtered.csv` and `pose_3d_filtered.csv` next to the originals.

**When the auto-assignment is wrong** (parent and child swapped, or filter kept the experimenter and dropped the child), open the interactive annotator:

```bash
video-annotate-tracks \
  --session-dir video_inference/output/P001c \
  --camera camera_a \
  --start-time 8:00 --end-time 9:00
```

Controls: click a bounding box, press `0`–`3` to reassign its track ID, `P` to propagate the correction forward, `S` to save, `Q` to quit. Writes `track_corrections.json`. Then apply:

```bash
video-correct-tracks --session-dir video_inference/output/P001c --camera camera_a
```

That produces `tracks_2d_corrected.csv` / `pose_3d_corrected.csv`. Re-run `video-filter-tracks` pointing at the corrected files if you want filtered + corrected output.

Repeat for `--camera camera_b` if you have two cameras.

---

## 6. Smooth (optional but recommended)

Pose detections jitter frame-to-frame. Smooth with a confidence-gated EMA — high-confidence points are kept as-is, low-confidence points get smoothed:

```bash
video-smooth \
  --camera-dir video_inference/output/P001c/camera_a \
  --pose-input pose_3d_filtered.csv \
  --tracks-input tracks_2d_filtered.csv \
  --tau 0.15 --conf-gate 0.3
```

Outputs: `pose_3d_smooth.csv`, `tracks_2d_smooth.csv`, `smoothing_summary.json`.

> **Always pass `--pose-input` / `--tracks-input` explicitly.** The defaults across CLIs are inconsistent (`video-smooth` looks for `_interpolated.csv`; `video-gaze-metrics` looks for `_filtered_5hz.csv`). Pointing them at the file you actually have is the only safe way.

---

## 7. Compute per-block metrics

**Pose metrics** (torso proximity, movement cross-correlation):

```bash
video-pose-metrics \
  --camera-dir video_inference/output/P001c/camera_a \
  --session-config session_config.json \
  --pose-input pose_3d_smooth.csv
```

**Gaze metrics** (mutual gaze, joint attention, gaze convergence) — requires running `gaze-infer` first; see [PIPELINE.md](PIPELINE.md#stage-5bd--gaze).

```bash
gaze-infer \
  --camera-dir video_inference/output/P001c/camera_a \
  --session-config session_config.json \
  --pose-input pose_3d_smooth.csv \
  --tracks-input tracks_2d_smooth.csv

video-gaze-metrics \
  --camera-dir video_inference/output/P001c/camera_a \
  --session-config session_config.json \
  --pose-input pose_3d_smooth.csv \
  --tracks-input tracks_2d_smooth.csv
```

Both metrics CLIs write per-block plots (PNG) and a metrics CSV into the camera directory.

---

## 8. Sanity-check the tracking

Overlay videos let you eyeball whether the pose tracking actually matched the right people:

```bash
# 2D overlay (bounding boxes + skeletons on the frames)
video-visualize \
  --camera-dir video_inference/output/P001c/camera_a \
  --tracks-input tracks_2d_filtered.csv \
  --pose-input  pose_3d_filtered.csv

# 3D skeletons in matplotlib (optional)
video-visualize-3d \
  --pose-csv  video_inference/output/P001c/camera_a/pose_3d_smooth.csv \
  --output-dir video_inference/output/P001c/camera_a/viz_3d
```

If the overlay looks wrong, go back to step 5.

---

## What you hand back

Per session, the deliverables live under `video_inference/output/<session_id>/<camera>/`:

| File | Purpose |
|---|---|
| `tracks_2d_smooth.csv`, `pose_3d_smooth.csv` | Cleaned per-frame data for downstream analysis |
| `pose_metrics_*.png`, `gaze_metrics_*.png` | Per-block summary plots |
| `synchrony_metrics.csv`, `gaze_metrics.csv` | Numeric metrics per block, one row per frame |
| `viz/*.mp4` | Sanity-check overlay videos |

Full schema definitions: [OUTPUTS.md](OUTPUTS.md).

---

## When something looks off

- **Two cameras disagree on parent/child**: check `camera_mappings` in `session_config.json`, or re-run `video-filter-tracks` on each camera separately.
- **A third person shows up in a metric**: re-do step 5 with `video-annotate-tracks`. If the pipeline ran with `--max-persons 2`, redo step 4 with `--max-persons 4` so the extra detection doesn't clobber the real tracks.
- **Pose looks broken**: bump to `yolo11m-pose.pt` (already in repo root) or try `--inference-backend rtmlib --rtmlib-3d` for a different model. See [PIPELINE.md](PIPELINE.md#backends).
- **Whole-pipeline reference**: [PIPELINE.md](PIPELINE.md). **Output column dictionaries**: [OUTPUTS.md](OUTPUTS.md). **Inputs (formats, conventions)**: [INPUTS.md](INPUTS.md).
- **EEG synchronization** (optional, separate workflow): see top-level [README.md](../README.md#optional-eeg-video-synchronization).
