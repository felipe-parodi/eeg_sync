# AGENTS.md -- AI Assistant Guide

This file provides guidance to AI coding assistants working with this repository.

## Project Overview

Pose, gaze, and synchrony analysis pipeline for parent-child interaction videos, plus an interactive EEG-video synchronization tool. Originally built for the Cayo Santiago studies. The primary user is a non-technical research colleague driving the pipeline with an LLM -- all CLI output, error messages, and documentation must be clear and jargon-free.

**Two workflows:**
- **Video pipeline (primary):** usually run through the Video Processing Portal (`portal/`, `start-portal.ps1`). The steps: `video-infer run --tracker-backend offline`, anchor labels, `video-assoc`, `video-refine-keypoints`, `video-interpolate`, `video-smooth`, then per-block metrics (`video-pose-metrics`, `gaze-infer`, `video-gaze-metrics`) and `video-session-report`. Older track cleanup (`video-filter-tracks`, `video-annotate-tracks`, `video-correct-tracks`) remains for jobs without a detection cache. Three backends: `ultralytics` (default), `rtmlib`, `sam3d`.
- **EEG sync (side track):** Interactive CLI (`eeg-sync`) that finds time offsets between EEG streams and video files using IR pulses, red-light frame marking, and multi-clap audio sync.

**Doc map:**
- `docs/COLLEAGUE_QUICKSTART.md` — one-doc end-to-end colleague walkthrough
- `docs/PIPELINE.md` — stage-by-stage CLI reference (LLM-readable)
- `docs/INPUTS.md` — input contract (folder layout, session_config, segment/exclusion formats)
- `docs/OUTPUTS.md` — output catalog with column dictionaries for every CSV
- `docs/eeg_sync/GETTING_STARTED.md`, `docs/eeg_sync/HOW_TO_USE_SYNC.md`, `docs/eeg_sync/TUTORIAL_POSE_EEG.md`, `docs/eeg_sync/CLOCK_DRIFT.md` — EEG-sync–side tutorials
- `docs/plans/` — dated feature plans (`name_YYMMDD.md`) and the old portal roadmap; history and rationale, not current instructions
- `templates/` — starter `session_config.example.json`, `segment_timings.example.csv`, `exclusion_windows.example.csv`

## Build & Run Commands

```bash
# Install (pick one)
pip install -e .                          # editable install
pip install -e ".[dev,video]"             # with dev tools + video stack
conda env create -f environment.yml       # full conda env (includes ffmpeg)

# Tests
pytest -q                                 # all tests, quiet (smoke tests skip)
pytest tests/test_sync_eeg_vid.py -v      # single module
pytest tests/test_sync_eeg_vid.py::test_find_sync_pulse_from_csv -v  # single test

# End-to-end smoke tests (run the real pipeline on a real compressed video).
# Skipped by default. Set RUN_SMOKE_TESTS=1 and optionally point SMOKE_VIDEO_PATH
# at a compressed mp4 outside the repo:
#   PowerShell:
#     $env:RUN_SMOKE_TESTS = "1"
#     $env:SMOKE_VIDEO_PATH = "C:\path\to\sample.mp4"   # optional
#     uv run pytest tests/test_smoke_end_to_end.py

# Lint & format
ruff check .                              # lint
black --check .                           # format check
black . && ruff check --fix .             # auto-fix
```

## Architecture

### Timing Convention (critical)

All offset math follows one pattern -- get this wrong and everything is wrong:
```
video_time = eeg_time + offset
eeg_time   = video_time - offset
eeg_a_time = eeg_b_time + eeg_a_to_eeg_b_offset
```
Offsets are always stored in `sync_results.json` with a human-readable `note` field explaining direction.

### Core Sync Package (`sync_eeg_vid/`)

Split into focused modules with a backward-compatible `__init__.py`:
- **`util.py`:** File validation, timestamp parsing, interactive prompts. No internal deps.
- **`eeg_io.py`:** EEG file parsing, IR pulse detection, segment extraction. IR blaster on `Analog Channel 0`, baseline = 257, sample rate = 250 Hz.
- **`viewer.py`:** `VideoFrameViewer` class -- OpenCV window with 1ms polling, frame-perfect navigation, auto-downscaling. Two modes: red-light (single mark) and multi-clap (mark/undo/save).
- **`plotting.py`:** `plot_sync_timeline`, `plot_eeg_data` (matplotlib, lazy-imported).
- **`sync_pipeline.py`:** `sync_two_eeg_files`, `sync_eeg_to_video`, `sync_videos`.
- **`cli.py`:** File collection, workflow orchestration, `main()` entry point.
- **`red_light.py`:** `eeg-red-light`. Finds when the red-light board is held up to camera A (`find_presentation_windows`; it is never shown to camera B) and searches for LED onsets inside those holds (`detect_red_light_events`). On 8–10 fps compressed copies the flash itself is not visible, so the exact frame is still marked by hand; validate onsets on full-rate originals before trusting them. Not wired into the portal.

All public symbols are re-exported from `sync_eeg_vid.__init__`, so `from sync_eeg_vid import find_sync_pulse` still works.

### Frame timing (critical)

Frames are extracted at 5 fps by keeping every k-th source frame (k = round(source fps / 5); 10 fps portal uploads give every 2nd frame), counted from the start of the video so all blocks share one time grid. `frames/frame_index.csv` holds each frame's true source time; `frames/extraction_info.json` holds the achieved rate, which the portal uses for everything downstream. Do not go back to ffmpeg's `fps=` filter: with a rate that does not divide the source rate it produces uneven frame spacing under evenly spaced timestamps.

### Video Inference (`video_inference/`)

Pipeline stages run in sequence: **compress -> extract frames -> infer -> track -> schema export -> validate**.

- **`pipeline.py`:** Orchestrator CLI (`video-infer run`). Wires together all stages per camera. Supports three backends: `sam3d`, `ultralytics`, `rtmlib`.
- **`ultralytics_runner.py`:** YOLOv11 2D pose backend. Two tracking modes: internal (parent/child heuristic by bbox area) and external (Roboflow ByteTrack with `lost_track_buffer=60` and ID slot recycling).
- **`rtmlib_runner.py`:** RTMLib pose backend (2D `Body` and 3D `Wholebody3d`) via onnxruntime / opencv / openvino. `--rtmlib-3d` toggles the 3D model. The runner bypasses upstream `PoseTracker` (which returned the wrong tuple arity for `RTMPose3d`) and calls the solution class directly per frame.
- **`sam3d_runner.py`:** SAM-3D-Body true-3D pose backend. Patches upstream CUDA hardcoding for CPU fallback. Imports from `third_party/sam-3d-body/` submodule. Override path via `EEG_SYNC_SAM3D_ROOT` env var.
- **`offline_assoc.py`:** Offline two-person identity association (portal default, `--tracker-backend offline`). Keypoint-aware dedup, conservative tracklets, per-role global best chains scored by appearance + per-camera size model, anchor labels local to +-10 s, rescue pass, gap fill. CLI re-solves with portal `anchors.json`.
- **`detection_cache.py`:** `DetCache` (.npz of every detection, keypoints, OSNet ReID embeddings) and `build_detection_cache` (batched YOLO + ReID).
- **`topdown_refine.py`:** Re-estimates tracked keypoints with ViTPose-plus-Large (GPU-side crop/decode), used only where it matches the YOLO skeleton (OKS >= 0.5).
- **`tracking.py`:** `TwoPersonTrackerState` with IoU-based temporal continuity. Frame 0 uses area prior (parent = largest bbox), subsequent frames use IoU matching.
- **`schema.py`:** Validates output CSVs (manifest, `tracks_2d.csv`, `pose_3d.csv`). Returns `ValidationResult` with error list, not exceptions.
- **`device.py`:** `resolve_device("auto")` returns `"cpu"`, `"mps"`, or `"cuda"`. Auto-detection order: CUDA > MPS > CPU. `resolve_inference_mode` selects `"full"` vs `"body"` based on hardware.

### Video Analysis (`video_analysis/`)

Post-inference cleanup, metrics, and visualization. Each module exposes a `build_arg_parser` + `main` CLI.

- **`track_filter.py`:** `video-filter-tracks`. Keeps top-N persistent tracks per named time block and assigns `parent=0`/`child=1` by mean bbox area within that block. Legacy; the portal uses `video_inference.offline_assoc` when a detection cache exists.
- **`track_annotator.py`:** `video-annotate-tracks`. Interactive OpenCV viewer — click bbox, press `0`-`3` to reassign, `P` to propagate forward, `S` to save. Writes `track_corrections.json`.
- **`track_correction.py`:** `video-correct-tracks`. Applies the annotator's JSON with atomic per-frame ID remapping (handles `0↔1` swaps correctly). Writes `*_corrected.csv`.
- **`interpolate.py`:** `video-interpolate`. Fills short gaps (<= `--max-gap-seconds`, default 1 s) at the extracted frame times in `frames/frame_index.csv`, never across blocks; longer gaps stay empty. Filled rows are flagged `interpolated`.
- **`temporal_smooth.py`:** `video-smooth`. Confidence-gated bidirectional EMA smoothing. High-confidence keypoints pass through raw; only low-confidence ones get smoothed. Default `tau=0.15`, `conf_gate=0.3`.
- **`pose_metrics.py`:** `video-pose-metrics`. Per-block torso proximity + windowed movement cross-correlation + tracking coverage (`coverage_both`, `low_coverage`). Uses the camera's own blocks (`--camera-id`).
- **`session_report.py`:** `video-session-report`. `session_metrics.csv` (one row per camera x block), `report.html`, and the study-wide `all_sessions_metrics.csv`.
- **`gaze_metrics.py`:** `video-gaze-metrics`. Per-block gaze category shares with chance levels (`<cat>_chance_pct`) + gaze convergence with `convergence_chance`.
- **`gaze_snapshots.py`:** `video-gaze-snapshots`. Samples N frames per block and renders gaze heatmap over the source frame for sanity checks.
- **`visualize_pose_tracks.py`:** `video-visualize`. Renders COCO17 skeletons + bounding boxes + track IDs onto video frames. Supports both JSON and CSV input modes.
- **`visualize_pose_3d.py`:** `video-visualize-3d`. Matplotlib 3D skeleton snapshots + optional MP4 from `pose_3d.csv`.

### Portal (`portal/`)

- **`app.py`:** FastAPI portal. Chunked uploads; per-camera blocks and extra-person windows (each in that video's own clock); anchor labeling; job page with per-block coverage; optional gaze labeling page (`/jobs/{id}/gaze-label`); `/metrics/all_sessions.csv`. Startup purges job folders (session outputs are kept).
- **`jobs.py`:** Builds and runs phase 1 (pose) and phase 2 (association, refinement, interpolation, smoothing, metrics, overlays, gaze), one thread per camera, then the session report and results zip. Records the real frame size after phase 1. `PortalJobConfig.gaze_model` selects `page` (default) or `gazelle`.
- **`gaze_labels.py`:** Samples frames with both people visible and validates and saves `gaze_labels.json` next to the outputs.

### CSV suffix chain

Each cleanup stage writes a **new** suffix; earlier files are never overwritten. Downstream CLIs accept `--tracks-input` / `--pose-input` to pick which stage:
```
tracks_2d.csv  →  tracks_2d_filtered.csv  →  tracks_2d_corrected.csv  →  tracks_2d_smooth.csv
pose_3d.csv    →  pose_3d_filtered.csv    →  pose_3d_corrected.csv    →  pose_3d_smooth.csv
```

### Gaze Analysis (`gaze_analysis/`)

Parent-child gaze estimation (PaGE ViT-B by default; Gaze-LLE optional) and synchrony analysis. Requires a `session_config.json` with manual parent/child track ID mapping per camera and session block definitions.

- **`config.py`:** Session configuration — loads JSON with camera-person mappings and session block timestamps.
- **`head_bbox.py`:** Derives normalized head bounding boxes from COCO keypoints (kp_000-kp_004). Falls back to top 30% of body bbox when head keypoints are low confidence.
- **`page_runner.py`:** Batch PaGE ViT-B gaze inference (portal default; pinned HF revisions). Same outputs as `gazelle_runner.py`.
- **`gazelle_runner.py`:** Batch Gazelle (Gaze-LLE) inference, kept as `gaze_model="gazelle"`. Loads frames + head bboxes, runs model, outputs `gaze_heatmap.csv` (scalar summaries) + `gaze_heatmaps.npz` (64x64 heatmaps).
- **`synchrony.py`:** Four metrics — torso proximity (2D Euclidean distance), movement cross-correlation (windowed xcorr of velocity), gaze categories, gaze convergence (background-removed heatmap overlap).
- **`gaze_categories.py`:** Frame-level categories from each gaze point vs the other person's head/body boxes in pixels (mutual_gaze, joint_attention, parent/child_watching, both_watching, independent, unknown), one-frame blip absorption, and chance levels (child gaze shifted 60 s).
- **`label_eval.py`:** `gaze-label-eval`. Scores gaze tables against the portal's human gaze labels (`gaze_labels.json`).
- **`config.frame_image_size`:** Real frame size from `frames/`. Use it for head-box normalization, because the session config size can be wrong (854x480 configured, 960x540 uploaded).
- **`plotting.py`:** 4-panel dashboard with session block coloring (grocery=green, synchrony=orange, storybook=blue).
- **`gaze_schema.py`:** Column definitions for gaze output CSVs.

### CLI Entry Points (all in `pyproject.toml`)

| Command | Purpose |
|---|---|
| `eeg-sync` | Interactive EEG-video synchronization |
| `video-compress-rapid` | Batch video compression |
| `video-infer run` | End-to-end pose inference pipeline |
| `video-infer-sam3d` | SAM-3D 3D pose (standalone) |
| `video-infer-ultralytics` | Ultralytics 2D pose (standalone) |
| `video-infer-rtmlib` | RTMLib 2D/3D pose (standalone) |
| `video-filter-tracks` | Block-based filter, auto-assign parent/child |
| `video-annotate-tracks` | Interactive track ID reassignment |
| `video-correct-tracks` | Apply annotator corrections to tracks/pose CSVs |
| `video-interpolate` | Upsample low-FPS outputs (optional) |
| `video-smooth` | Confidence-gated EMA smoothing |
| `video-pose-metrics` | Per-block torso proximity + movement xcorr |
| `video-gaze-metrics` | Per-block gaze categories + convergence |
| `video-gaze-snapshots` | Per-block gaze heatmap-over-frame samples |
| `video-visualize` | 2D pose overlay video |
| `video-visualize-3d` | 3D skeleton PNG/MP4 from pose_3d.csv |
| `gaze-infer` | PaGE ViT-B gaze estimation (`gaze-infer-gazelle`: Gaze-LLE) |
| `gaze-label-eval` | Score gaze output against portal gaze labels |
| `video-assoc` | Offline two-person association with anchors |
| `video-refine-keypoints` | ViTPose-Large keypoint refinement |
| `video-session-report` | Session table, report.html, all-sessions table |
| `video-processing-portal` | Web portal (use `start-portal.ps1`) |
| `eeg-red-light` | Find the red-light board presentation in a video |
| `gaze-synchrony` | Compute parent-child synchrony metrics |
| `gaze-plot` | Plot synchrony dashboard |

## How to Explore This Codebase

1. Read `README.md` for the user-facing overview.
2. Read `docs/PIPELINE.md` for the full stage-by-stage CLI map (video pipeline).
3. Read `docs/OUTPUTS.md` for the per-file schemas.
4. For the video pipeline orchestration, read `video_inference/pipeline.py` (`PipelineConfig`, `_run_camera_pipeline`).
5. For sync, read `sync_eeg_vid/sync_pipeline.py` (`sync_two_eeg_files`, `sync_eeg_to_video`, `sync_videos`) plus `sync_eeg_vid/eeg_io.py` (`find_sync_pulse`, `extract_eeg_segment`) and `sync_eeg_vid/cli.py:main`.
6. Read `tests/test_sync_eeg_vid.py` and `tests/test_video_inference_*.py` to understand protected behavior.
7. Validate inference outputs with `video_inference.schema.validate_session_output`.

## Data Policy

- **Never commit** raw participant videos or identifiable recordings.
- `video_inference/data/` is gitignored for local-only media.
- Keep generated artifacts local (`sync_results.json`, plots, output CSVs).
- Tests use synthetic fixtures in `tests/fixtures/`.

### IMPORTANT: Video Compression Requirement

**NEVER run video inference on raw/uncompressed video files.** Raw GoPro recordings are typically 2-10 GB each and will cause extremely slow frame extraction and excessive disk usage.

Always compress videos first. Target file size: **under 50 MB**. Use one of:
1. `video-compress-rapid` CLI to batch-compress raw videos.
2. The pipeline's built-in compression (the default -- do NOT pass `--skip-compress` on raw files).

Pre-compressed videos in `video_inference/compressed/` are safe to use with `--skip-compress`.

## Testing Patterns

- Synthetic fixture data lives in `tests/fixtures/video_inference/mock_session/`.
- Video/GPU tests mock ffmpeg, image files, and `torch.cuda.is_available()`.
- Schema tests validate column presence, types, and cross-file consistency.
- `tomllib` (Python 3.11+) is used in one test -- Python 3.10 skips it gracefully.

## ID Assignment Policy

Three valid sources of parent/child track IDs, in order of preference:

0. **Offline two-person association (portal default since 2026-10).** `--tracker-backend offline` in phase 1, then `python -m video_inference.offline_assoc --anchors-json ...` writes `*_filtered.csv` with parent=0/child=1. Benchmarked with `pose_training/bench.py` against held-out colleague anchor labels.
1. **`video-filter-tracks` (previous colleague workflow; still used for jobs without a detection cache).** Block-aware: top-N most persistent tracks per `session_blocks[i]` keep, parent=0/child=1 assigned by mean bbox area within that block. Produces `tracks_2d_filtered.csv` / `pose_3d_filtered.csv`.
2. **`video-annotate-tracks` + `video-correct-tracks`.** Interactive override when auto-assignment is wrong (e.g. parent physically smaller, experimenter in frame, parent-child swap). Writes `track_corrections.json` → applied to `*_corrected.csv`.
3. **Inference-time tracker.** `--tracker-backend internal` uses the two-person heuristic (parent = largest bbox at frame 0, IoU continuity after). `--tracker-backend roboflow --tracker-name bytetrack` uses ByteTrack with `lost_track_buffer=60` and recycles IDs into stable `0..max_persons-1` slots.

Configured per inference run via `--max-persons` and `--enforce-exact-person-count`. Downstream metrics CLIs accept `--tracks-input` / `--pose-input` to pick which stage's CSV to read from.

## Style

- Python 3.10+, Google-style docstrings, type hints on all signatures.
- `black` for formatting, `ruff` for linting (config in `pyproject.toml`).
- Favor CPU fallback paths by default unless GPU is explicitly available.
- No silent error swallowing -- fail loudly with clear messages.
- If introducing new outputs, define a stable schema and version it.
