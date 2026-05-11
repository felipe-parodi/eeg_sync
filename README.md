# eeg_sync

Pose, gaze, and synchrony analysis for parent–child interaction videos, plus an interactive EEG ↔ video synchronization tool. Originally built for the Cayo Santiago studies; now a general video → analyzable outputs pipeline.

> **New here?** Go to [`docs/COLLEAGUE_QUICKSTART.md`](docs/COLLEAGUE_QUICKSTART.md). It's the one doc to read.
>
> **Driving this with an LLM?** Point it at [`docs/PIPELINE.md`](docs/PIPELINE.md), [`docs/INPUTS.md`](docs/INPUTS.md), and [`docs/OUTPUTS.md`](docs/OUTPUTS.md). All three are written for both humans and machines.

---

## What this does

Given one or two videos of a parent–child session, the pipeline produces:

- **Per-frame pose tracks** (2D bounding boxes + 17 COCO keypoints per person) — CSV
- **Per-frame gaze estimates** (heatmaps + peak coordinates per person) — CSV + NPZ
- **Per-block synchrony metrics** — torso proximity, movement cross-correlation, gaze categories (mutual / joint / one-watching / independent), gaze convergence — CSV + PNG dashboards
- **Sanity-check overlay videos** so you can watch the tracker working

Optionally, it also synchronizes those videos with OpenBCI EEG recordings.

---

## Quick start

```bash
git clone https://github.com/felipe-parodi/eeg_sync.git
cd eeg_sync
pip install -e ".[dev,video,gaze]"

# One session, two cameras, end-to-end
video-compress-rapid \
    --video raw/cam_a.mov --video raw/cam_b.mov \
    --output-dir video_inference/compressed

video-infer run \
    --video-a video_inference/compressed/cam_a_rapid.mp4 \
    --video-b video_inference/compressed/cam_b_rapid.mp4 \
    --inference-backend ultralytics \
    --ultralytics-model-path yolo11m-pose.pt \
    --tracker-backend roboflow --tracker-name bytetrack \
    --max-persons 4 --frame-rate 5 \
    --device auto --skip-compress --session-id P001c

video-filter-tracks --session-dir video_inference/output/P001c --camera camera_a \
    --blocks "free_play,13:26,23:40;storybook,29:22,37:06" --source-fps 5.0

video-smooth --camera-dir video_inference/output/P001c/camera_a \
    --pose-input pose_3d_filtered.csv --tracks-input tracks_2d_filtered.csv

video-pose-metrics  --camera-dir video_inference/output/P001c/camera_a --session-config session_config.json
video-gaze-metrics  --camera-dir video_inference/output/P001c/camera_a --session-config session_config.json
```

Outputs land in `video_inference/output/P001c/camera_a/`. Full walkthrough — including the "what if someone else walks in?" cases — lives in [`docs/COLLEAGUE_QUICKSTART.md`](docs/COLLEAGUE_QUICKSTART.md).

---

## Commands at a glance

| Command | Purpose | Stage |
|---|---|---|
| `video-compress-rapid` | Shrink raw GoPro to ≤50 MB | 1 |
| `video-infer run` | Pose inference + tracking (`ultralytics` / `sam3d` / `rtmlib`) | 2 |
| `video-filter-tracks` | Keep top-N tracks per named block, auto-assign parent=0/child=1 | 3 |
| `video-annotate-tracks` | Interactive viewer to fix track IDs (click + press 0–3) | 3b (optional) |
| `video-correct-tracks` | Apply the annotator's JSON to tracks/pose CSVs | 3c (optional) |
| `video-interpolate` | Upsample low-FPS outputs | 4 (optional) |
| `video-smooth` | Confidence-gated EMA smoothing | 4b |
| `video-pose-metrics` | Per-block torso proximity + movement xcorr | 5 |
| `gaze-infer` | Gaze heatmaps via Gazelle | 5 |
| `gaze-synchrony` | All four synchrony metrics → CSV | 5 |
| `gaze-plot` | 4-panel synchrony dashboard | 5 |
| `video-gaze-metrics` | Per-block gaze categories + convergence | 5 |
| `video-gaze-snapshots` | Spot-check gaze heatmaps over frames | 5 |
| `video-visualize` | 2D pose overlay video | 6 |
| `video-visualize-3d` | 3D skeleton PNGs / MP4 | 6 |
| `eeg-sync` | Interactive EEG ↔ video offset finder | (separate) |

Detail and inputs/outputs per command: [`docs/PIPELINE.md`](docs/PIPELINE.md).

---

## Repo layout

```
sync_eeg_vid/       # EEG ↔ video sync (interactive CLI: eeg-sync)
video_inference/    # Pose inference: ultralytics / sam3d / rtmlib backends + ByteTrack
video_analysis/     # Post-processing: smooth, interpolate, filter-tracks, annotate-tracks,
                    #                  correct-tracks, visualize (2D + 3D), metrics
gaze_analysis/      # Gazelle gaze inference + 4-metric synchrony + dashboard
docs/               # All user-facing documentation (read these in order: QUICKSTART → PIPELINE → OUTPUTS)
templates/          # Starter session_config.json + segment / exclusion CSV examples
tests/              # pytest suite; 168 tests, all green on master
third_party/        # Git submodules: sam-3d-body, gazelle
```

Per-module READMEs: [`video_inference/README.md`](video_inference/README.md).

---

## Install options

```bash
pip install -e .                          # core only (sync + base utilities)
pip install -e ".[video]"                 # + Ultralytics 2D pose
pip install -e ".[video,gaze]"            # + Gazelle gaze stack
pip install -e ".[video,gaze,dev]"        # + pytest, ruff, black, mypy
conda env create -f environment.yml       # full env including ffmpeg, CUDA-ready
```

CPU is the default and fine for the pipeline. CUDA / Apple Silicon are auto-detected by `--device auto`. `ffmpeg` must be on `PATH`.

---

## Optional: EEG video synchronization

Phase 1 of this repo was an interactive tool to find time offsets between OpenBCI EEG streams (IR-blaster pulse) and GoPro video (red-light frame mark + optional multi-clap audio cue). It still works and lives in `sync_eeg_vid/`.

Run it interactively:

```bash
eeg-sync
```

You'll be walked through:
1. Optional dual-EEG sync (IR pulse in both EEG files)
2. EEG → Video A (red light at known timestamp)
3. Optional Video A → Video B (claps)

Output is `sync_results.json` with the offsets to use elsewhere. Conventions:

```
video_time = eeg_time + offset
eeg_time   = video_time - offset
eeg_a_time = eeg_b_time + eeg_a_to_eeg_b_offset
```

### `sync_results.json` schema

```json
{
  "eeg_a_to_eeg_b": {
    "eeg_file_a": "...",
    "eeg_file_b": "...",
    "eeg_sync_time_a": 81.348,
    "eeg_sync_time_b": 1.332,
    "offset": 80.016,
    "note": "To convert EEG B time to EEG A time: time_a = time_b + 80.016"
  },
  "eeg_to_video_a": {
    "eeg_file": "...",
    "video_file": "...",
    "eeg_sync_time": 81.348,
    "video_sync_time": 83.123,
    "video_frame": 2493,
    "offset": 1.775,
    "note": "To convert EEG time to video time: video_time = eeg_time + 1.775"
  },
  "video_a_to_video_b": {
    "video_a": "...",
    "video_b": "...",
    "sync_time_a": 165.432,
    "sync_time_b": 22.865,
    "offset": 142.567,
    "note": "To convert Video B time to Video A time: time_a = time_b + 142.567"
  }
}
```

Tutorials for downstream EEG analysis:

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) — first multi-modal analysis (≈10 min read)
- [docs/HOW_TO_USE_SYNC.md](docs/HOW_TO_USE_SYNC.md) — using `sync_results.json` in code
- [docs/TUTORIAL_POSE_EEG.md](docs/TUTORIAL_POSE_EEG.md) — joining pose data to EEG
- [docs/CLOCK_DRIFT.md](docs/CLOCK_DRIFT.md) — when single-point sync isn't enough

### Interactive video viewer controls

| Mode | Key | Action |
|---|---|---|
| All | `A`/`D` or `←`/`→` | ±1 frame |
| All | `W`/`S` or `↑`/`↓` | ±1 second |
| All | `,` / `.` | ±10 seconds |
| All | `[` / `]` | ±1 minute |
| Red light | `Space` | Mark frame and exit |
| Multi-clap | `C` | Mark clap at current frame |
| Multi-clap | `U` | Undo last clap |
| Multi-clap | `Enter`/`Space` | Save claps and exit |
| All | `Q` / `Esc` | Quit (no save) |

Performance details (frame caching, auto-downscale, instant seeking) in [`sync_eeg_vid/`](sync_eeg_vid/).

---

## Development

```bash
pytest -q                       # full suite (168 tests)
ruff check . && black --check . # lint + format check
black . && ruff check --fix .   # auto-fix
```

Style: Python 3.10+, Google docstrings, type hints on all signatures, fail-loudly error handling. No `_v2` / `_new` file suffixes — edit in place. See [`AGENTS.md`](AGENTS.md) for the LLM-collaboration guide.

CI runs lint + format check + tests on Python 3.10 and 3.11 (see `.github/workflows/`).

### Data policy

- **Never commit** raw participant videos, EEG recordings, or generated outputs.
- `video_inference/data/`, `video_inference/compressed/`, `video_inference/output/` are gitignored.
- Tests use synthetic fixtures in `tests/fixtures/`.
- The one tracked JSON is `session_config.json` (per-session analysis windows — no PII).

---

## License

MIT.

## Citation

```
eeg_sync — Pose, gaze, and synchrony analysis for parent–child videos
https://github.com/felipe-parodi/eeg_sync
```
