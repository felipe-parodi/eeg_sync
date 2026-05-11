# Pipeline reference

Stage-by-stage reference for the full video → analyzable outputs pipeline. This is the doc to point an LLM at when it needs to know which CLI to invoke at which step and what inputs/outputs each one expects.

For a hand-held walkthrough, read [COLLEAGUE_QUICKSTART.md](COLLEAGUE_QUICKSTART.md) first. For column-level output schemas, read [OUTPUTS.md](OUTPUTS.md). For input formats, read [INPUTS.md](INPUTS.md).

## Pipeline at a glance

```
                                                           ┌─ video-pose-metrics ─┐
                                                           │                      │
raw video → compress → infer → filter-tracks → smooth → ──┤                      ├──→ per-block plots + CSVs
                          │                                │                      │
                          │     (optional)                 └─ video-gaze-metrics ─┘
                          │      annotate-tracks → correct-tracks
                          │
                          └──────→ video-visualize, video-visualize-3d (sanity-check overlays)
                                          │
                          gaze-infer  ─→  gaze-synchrony  ─→  gaze-plot

(separate side track)
EEG files →  eeg-sync  →  sync_results.json  (offsets for joining EEG ↔ video time)
```

## Stage table

| # | Stage | CLI | Reads | Writes |
|---|---|---|---|---|
| 1 | Compress | `video-compress-rapid` | Raw video(s) | `<name>_rapid.mp4` |
| 2 | Inference | `video-infer run` | Compressed video(s), pose model | `tracks_2d.csv`, `pose_3d.csv`, `frames/`, `manifest.json`, `intermediate/inference_raw.json` per camera; `session_summary.json` |
| 3 | Block-filter | `video-filter-tracks` | `tracks_2d.csv`, `pose_3d.csv` | `tracks_2d_filtered.csv`, `pose_3d_filtered.csv` |
| 3b | Annotate (optional) | `video-annotate-tracks` | `tracks_2d.csv`, `frames/` | `track_corrections.json` |
| 3c | Apply corrections | `video-correct-tracks` | `track_corrections.json`, `tracks_2d.csv`, `pose_3d.csv` | `tracks_2d_corrected.csv`, `pose_3d_corrected.csv` |
| 4 | Interpolate (optional) | `video-interpolate` | `tracks_2d*.csv`, `pose_3d*.csv` | `tracks_2d_interpolated.csv`, `pose_3d_interpolated.csv` |
| 4b | Smooth | `video-smooth` | `tracks_2d*.csv`, `pose_3d*.csv` | `tracks_2d_smooth.csv`, `pose_3d_smooth.csv`, `smoothing_summary.json` |
| 5a | Pose metrics | `video-pose-metrics` | `pose_3d*.csv`, `session_config.json` | `pose_metrics_*.png`, per-block CSVs |
| 5b | Gaze inference | `gaze-infer` | `tracks_2d*.csv`, `pose_3d*.csv`, `frames/`, `session_config.json` | `gaze_heatmap.csv`, `gaze_heatmaps.npz` |
| 5c | Synchrony | `gaze-synchrony` | All of the above | `synchrony_metrics.csv`, `synchrony_summary.json` |
| 5d | Synchrony plot | `gaze-plot` | `synchrony_metrics.csv`, `session_config.json` | `synchrony_dashboard.png` |
| 5e | Gaze metrics | `video-gaze-metrics` | `gaze_heatmap.csv`, `gaze_heatmaps.npz`, pose/tracks, `session_config.json` | `gaze_metrics_*.png`, per-block CSV |
| 5f | Gaze snapshots | `video-gaze-snapshots` | Same as 5e + `frames/` | sample heatmap-over-frame PNGs per block |
| 6 | 2D overlay | `video-visualize` | `tracks_2d*.csv`, `pose_3d*.csv`, `frames/` | `<name>.mp4` |
| 6b | 3D viz | `video-visualize-3d` | `pose_3d*.csv` | PNG snapshots, optional `.mp4` |
| — | EEG sync (separate) | `eeg-sync` | OpenBCI files, video(s) | `sync_results.json`, optional `sync_timeline.png` |

`*.csv` means "any of the suffix chain": `tracks_2d.csv` → `_filtered.csv` → `_corrected.csv` → `_smooth.csv`. Most downstream CLIs take `--tracks-input` / `--pose-input` so you choose which stage's output to feed in.

---

## Stage 1 — `video-compress-rapid`

Drops file size by ≥98% so all downstream steps run in reasonable time on CPU. **Required** before inference.

```bash
video-compress-rapid \
  --video video_inference/data/camera_a_raw.mov \
  --video video_inference/data/camera_b_raw.mov \
  --output-dir video_inference/compressed
```

Key flags: `--target-fps 10` (default), `--max-width 960` (default), `--crf 30` (default — lower = larger file, higher quality). Output filename = `<input_stem>_rapid.mp4`. Writes `rapid_compression_summary.json` in `--output-dir`.

---

## Stage 2 — `video-infer run`

End-to-end orchestrator that extracts frames, runs pose inference, and writes a per-camera output directory.

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
  --output-dir video_inference/output \
  --session-id P001c
```

| Flag | Default | Notes |
|---|---|---|
| `--video-a` (req) / `--video-b` | — | Source videos (use compressed paths + `--skip-compress`). |
| `--output-dir` | `video_inference/output` | Per-session subdir is created. |
| `--session-id` | `session_YYYYMMDD_HHMMSS` | Subfolder name. |
| `--device` | `auto` | `auto` → CUDA > MPS > CPU. |
| `--inference-backend` | `sam3d` | `sam3d`, `ultralytics`, or `rtmlib`. |
| `--frame-rate` | `5.0` | Inference fps. Lower = faster, less data. |
| `--max-persons` | `2` | Upper bound on simultaneous tracks. Set to `4` when extra people may briefly appear; prune later with `video-filter-tracks`. |
| `--enforce-exact-person-count` | False | Drops frames that don't have exactly `--max-persons` detections. Off by default. |
| `--skip-compress` | False | Skip stage 1 if your inputs are already compressed. |

### Backends

| Backend | Output | Reads | When to use |
|---|---|---|---|
| `ultralytics` | 2D keypoints (`pose_3d.csv` is 2D when `z_m=0.0`) | `--ultralytics-model-path` (e.g. `yolo11m-pose.pt`) | **Default for the colleague workflow.** CPU-friendly. |
| `sam3d` | True 3D keypoints in metres | `--checkpoint-path`, `--mhr-path` (SAM-3D weights) | When you need 3D coordinates. Slower; requires the `third_party/sam-3d-body` submodule. |
| `rtmlib` | 2D (Body) or 3D (Wholebody3d) via onnxruntime | (downloads weights on first run) | Alternative if Ultralytics quality is poor. Set `--rtmlib-3d` for 3D output. |

### Trackers

| Tracker | Behaviour |
|---|---|
| `--tracker-backend internal` | Two-person heuristic: parent = largest bbox at frame 0, then IoU continuity. Only suitable when `--max-persons 2` and detections are clean. |
| `--tracker-backend roboflow --tracker-name bytetrack` | ByteTrack with `track_buffer=60` (frames). Recommended for `--max-persons > 2` or any session with possible occlusions. |

### Per-camera outputs

```
<output-dir>/<session_id>/<camera>/
  manifest.json
  tracks_2d.csv
  pose_3d.csv
  frames/frame_000001.jpg ...
  intermediate/inference_raw.json
```

Plus session-level `session_summary.json`.

Schema details: [OUTPUTS.md](OUTPUTS.md).

---

## Stage 3 — `video-filter-tracks`

Keeps only the two (or `--n-keep`) most persistent tracks within each named block, and assigns `track_id=0` to the larger person (parent) and `track_id=1` to the smaller (child) by mean bbox area within that block.

```bash
video-filter-tracks \
  --session-dir video_inference/output/P001c \
  --camera camera_a \
  --blocks "grocery_free_play,13:26,23:40;synchrony_intervention,27:56,28:45;storybook_reading,29:22,37:06" \
  --source-fps 5.0 \
  --n-keep 2
```

| Flag | Default | Notes |
|---|---|---|
| `--session-dir` (req) | — | The `<output-dir>/<session_id>` from stage 2. |
| `--camera` | `camera_a` | |
| `--blocks` (req) | — | `name,start,end;name,start,end` — times in `MM:SS` or `HH:MM:SS`. Must match `session_blocks` in `session_config.json`. |
| `--source-fps` | `30.0` | **Set this to match the `--frame-rate` from stage 2.** It's how time strings convert to frame indices. |
| `--n-keep` | `2` | Top-N tracks per block. Use `2` for parent + child. |

Outputs in the camera dir: `tracks_2d_filtered.csv`, `pose_3d_filtered.csv` (if `pose_3d.csv` exists). Frames outside any block are dropped.

**Caveat**: assignment is per-block by mean bbox area. If the parent is sometimes physically smaller in frame than the child (e.g. parent sitting, child standing close to camera), assignment may flip between blocks. In that case use the annotator (stage 3b) to enforce a global mapping.

---

## Stage 3b — `video-annotate-tracks` (optional)

Interactive OpenCV viewer for click-to-reassign correction of track IDs.

```bash
video-annotate-tracks \
  --session-dir video_inference/output/P001c \
  --camera camera_a \
  --start-time 8:00 --end-time 9:00 \
  --display-fps 5.0
```

Controls in-window:

| Key | Action |
|---|---|
| Click bbox | Select |
| `0` / `1` / `2` / `3` | Assign track_id to selected bbox |
| `P` | Propagate corrections forward (IoU continuity) |
| `S` | Save `track_corrections.json` |
| `←`/`→` or `A`/`D` | ±1 frame |
| `↑`/`↓` or `W`/`S` | ±30 frames (~1 s at 30 fps) |
| `[` / `]` | ±1 minute |
| `N` / `B` | ±keyframe interval (default 60 s) |
| `Q` / `Esc` | Quit (auto-saves) |

Writes `track_corrections.json` in the camera dir.

---

## Stage 3c — `video-correct-tracks`

Applies the corrections JSON. Per-frame remap is atomic (handles `0↔1` swaps correctly).

```bash
video-correct-tracks --session-dir video_inference/output/P001c --camera camera_a
```

Writes `tracks_2d_corrected.csv`, `pose_3d_corrected.csv`. Use these as inputs to subsequent stages with `--tracks-input` / `--pose-input`.

---

## Stage 4 — `video-interpolate` (optional)

Linearly upsamples low-FPS outputs (e.g. 1 fps → 8 fps). Gap-aware: won't interpolate across `--max-gap-s` (default ~10 frames worth). Usually **not needed** if you ran inference at 5 fps already.

```bash
video-interpolate \
  --camera-dir video_inference/output/P001c/camera_a \
  --target-fps 8 \
  --tracks-input tracks_2d.csv \
  --pose-input pose_3d.csv
```

---

## Stage 4b — `video-smooth`

Confidence-gated bidirectional EMA. Keypoints above `--conf-gate` pass through untouched; below get smoothed with time constant `--tau`. A second knob, `--conf-threshold` (default 0.3), controls which raw keypoints get NaN-infilled before smoothing — usually safe to leave at the default. The recommended cleanup before metrics.

```bash
video-smooth \
  --camera-dir video_inference/output/P001c/camera_a \
  --tau 0.15 \
  --conf-gate 0.3 \
  --pose-input pose_3d_filtered.csv \
  --tracks-input tracks_2d_filtered.csv
```

Writes `pose_3d_smooth.csv`, `tracks_2d_smooth.csv`, `smoothing_summary.json`.

> **Footgun**: `video-smooth --pose-input` defaults to `pose_3d_interpolated.csv` (expects you ran stage 4 first). Pass it explicitly to point at whichever upstream file you actually have.

---

## Stage 5a — `video-pose-metrics`

Per-block torso proximity and movement cross-correlation between parent and child.

```bash
video-pose-metrics \
  --camera-dir video_inference/output/P001c/camera_a \
  --session-config session_config.json \
  --pose-input pose_3d_smooth.csv \
  --xcorr-window 5.0 \
  --xcorr-max-lag 2.0
```

Reads `session_blocks` from `session_config.json`. Writes per-block plots + a per-block CSV (`pose_metrics_<block>.csv`).

---

## Stage 5b–d — gaze

Three CLIs run in order: `gaze-infer` → `gaze-synchrony` → `gaze-plot`. Requires the optional Gazelle dependencies (`pip install -e ".[gaze]"`) and the `third_party/gazelle` submodule (`git submodule update --init`).

```bash
gaze-infer \
  --camera-dir video_inference/output/P001c/camera_a \
  --session-config session_config.json \
  --pose-input pose_3d_smooth.csv \
  --tracks-input tracks_2d_smooth.csv \
  --device auto

gaze-synchrony \
  --camera-dir video_inference/output/P001c/camera_a \
  --session-config session_config.json \
  --pose-input pose_3d_smooth.csv \
  --tracks-input tracks_2d_smooth.csv

gaze-plot \
  --camera-dir video_inference/output/P001c/camera_a \
  --session-config session_config.json
```

`gaze-infer` writes `gaze_heatmap.csv` (per-frame scalars) and `gaze_heatmaps.npz` (64×64 heatmaps).
`gaze-synchrony` writes `synchrony_metrics.csv` (one row per frame, all four metrics) and `synchrony_summary.json`.
`gaze-plot` writes `synchrony_dashboard.png` (4-panel, shaded by `session_blocks` colors).

---

## Stage 5e — `video-gaze-metrics`

Per-block gaze category proportions (mutual, joint, parent-watching, child-watching, independent) and gaze convergence.

```bash
video-gaze-metrics \
  --camera-dir video_inference/output/P001c/camera_a \
  --session-config session_config.json \
  --pose-input pose_3d_smooth.csv \
  --tracks-input tracks_2d_smooth.csv
```

> **Footgun**: `video-gaze-metrics` (and `video-gaze-snapshots`) default `--pose-input` to `pose_3d_filtered_5hz.csv` — a file the standard colleague workflow never produces. Always pass `--pose-input` and `--tracks-input` explicitly.

Writes `gaze_metrics_<block>.png` and a per-block CSV.

---

## Stage 5f — `video-gaze-snapshots`

Samples N frames per block and renders the gaze heatmap overlaid on the source frame. Useful for sanity-checking that gaze actually landed where it claims.

```bash
video-gaze-snapshots \
  --camera-dir video_inference/output/P001c/camera_a \
  --session-config session_config.json \
  --pose-input pose_3d_smooth.csv \
  --tracks-input tracks_2d_smooth.csv \
  --n-samples 8 \
  --output-dir block_checks
```

---

## Stage 6 — overlay videos

`video-visualize` renders 2D pose + bboxes onto the original frames:

```bash
video-visualize \
  --camera-dir video_inference/output/P001c/camera_a \
  --output-video video_inference/output/P001c/camera_a/overlay.mp4 \
  --tracks-input tracks_2d_smooth.csv \
  --pose-input pose_3d_smooth.csv \
  --output-fps 12
```

`video-visualize-3d` renders 3D skeletons in matplotlib (headless), as PNG snapshots + optional MP4:

```bash
video-visualize-3d \
  --pose-csv video_inference/output/P001c/camera_a/pose_3d_smooth.csv \
  --output-dir video_inference/output/P001c/camera_a/viz_3d \
  --output-video video_inference/output/P001c/camera_a/viz_3d/skeleton.mp4 \
  --snapshot-interval 30 \
  --rotate
```

---

## EEG sync (separate workflow)

Run `eeg-sync` from the repo root for an interactive walkthrough that produces `sync_results.json`. This is independent of the video pipeline — its only output (offsets in seconds) is what you use later to convert between EEG time and video time. Full walkthrough in the top-level [README](../README.md#optional-eeg-video-synchronization).
