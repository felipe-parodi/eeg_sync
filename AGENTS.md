# AGENTS.md -- AI Assistant Guide

This file provides guidance to AI coding assistants working with this repository.

## Project Overview

EEG-video synchronization tool for aligning OpenBCI EEG recordings with GoPro videos. Built for parent-child interaction studies at Cayo Santiago. The primary user is a non-technical research colleague -- all CLI output, error messages, and documentation must be clear and jargon-free.

**Two phases:**
- **Phase 1 (complete):** Interactive CLI (`eeg-sync`) that finds time offsets between EEG streams and video files using IR pulses, red-light frame marking, and multi-clap audio sync.
- **Phase 2 (in progress):** Video pose inference pipeline (`video-infer`) with 2D/3D backends, temporal tracking, interpolation, and visualization.

## Build & Run Commands

```bash
# Install (pick one)
pip install -e .                          # editable install
pip install -e ".[dev,video]"             # with dev tools + video stack
conda env create -f environment.yml       # full conda env (includes ffmpeg)

# Tests
pytest -q                                 # all tests, quiet
pytest tests/test_sync_eeg_vid.py -v      # single module
pytest tests/test_sync_eeg_vid.py::test_find_sync_pulse_from_csv -v  # single test

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

All public symbols are re-exported from `sync_eeg_vid.__init__`, so `from sync_eeg_vid import find_sync_pulse` still works.

### Video Inference (`video_inference/`)

Pipeline stages run in sequence: **compress -> extract frames -> infer -> track -> schema export -> validate**.

- **`pipeline.py`:** Orchestrator CLI (`video-infer run`). Wires together all stages per camera.
- **`ultralytics_runner.py`:** YOLOv11 2D pose backend. Two tracking modes: internal (parent/child heuristic by bbox area) and external (Roboflow ByteTrack with ID slot recycling).
- **`sam3d_runner.py`:** SAM-3D-Body 3D pose backend. Patches upstream CUDA hardcoding for CPU fallback. Imports from `third_party/sam-3d-body/` submodule. Override path via `EEG_SYNC_SAM3D_ROOT` env var.
- **`tracking.py`:** `TwoPersonTrackerState` with IoU-based temporal continuity. Frame 0 uses area prior (parent = largest bbox), subsequent frames use IoU matching.
- **`schema.py`:** Validates output CSVs (manifest, `tracks_2d.csv`, `pose_3d.csv`). Returns `ValidationResult` with error list, not exceptions.
- **`device.py`:** `resolve_device("auto")` returns `"cpu"`, `"mps"`, or `"cuda"`. Auto-detection order: CUDA > MPS > CPU. `resolve_inference_mode` selects `"full"` vs `"body"` based on hardware.

### Video Analysis (`video_analysis/`)

- **`interpolate.py`:** Linear upsampling of low-FPS track/pose data (e.g., 1 FPS -> 8 FPS). Gap-aware: won't interpolate across large timestamp gaps. **Optional** — 5 FPS native is usually sufficient.
- **`temporal_smooth.py`:** Confidence-gated bidirectional EMA smoothing. High-confidence keypoints pass through raw; only low-confidence ones get smoothed. Default `tau=0.15`, `conf_gate=0.3`.
- **`visualize_pose_tracks.py`:** Renders COCO17 skeletons + bounding boxes + track IDs onto video frames. Supports both JSON and CSV input modes.

### Gaze Analysis (`gaze_analysis/`)

Parent-child gaze estimation and synchrony analysis using Gazelle (CVPR 2025). Requires a `session_config.json` with manual parent/child track ID mapping per camera and session block definitions.

- **`config.py`:** Session configuration — loads JSON with camera-person mappings and session block timestamps.
- **`head_bbox.py`:** Derives normalized head bounding boxes from COCO keypoints (kp_000-kp_004). Falls back to top 30% of body bbox when head keypoints are low confidence.
- **`gazelle_runner.py`:** Batch Gazelle inference. Loads frames + head bboxes, runs model, outputs `gaze_heatmap.csv` (scalar summaries) + `gaze_heatmaps.npz` (64x64 heatmaps).
- **`synchrony.py`:** Four metrics — torso proximity (2D Euclidean distance), movement cross-correlation (windowed xcorr of velocity), gaze categories (mutual/joint/watching/independent), gaze convergence (cosine similarity of heatmaps).
- **`plotting.py`:** 4-panel dashboard with session block coloring (grocery=green, synchrony=orange, storybook=blue).
- **`gaze_schema.py`:** Column definitions for gaze output CSVs.

### CLI Entry Points (all in `pyproject.toml`)

| Command | Purpose |
|---|---|
| `eeg-sync` | Interactive EEG-video synchronization |
| `video-infer run` | End-to-end pose inference pipeline |
| `video-infer-sam3d` | SAM-3D 3D pose (standalone) |
| `video-infer-ultralytics` | Ultralytics 2D pose (standalone) |
| `video-compress-rapid` | Batch video compression |
| `video-interpolate` | Upsample low-FPS outputs (optional) |
| `video-smooth` | Confidence-gated EMA smoothing |
| `video-visualize` | Pose overlay visualization |
| `gaze-infer` | Gazelle gaze estimation on video frames |
| `gaze-synchrony` | Compute parent-child synchrony metrics |
| `gaze-plot` | Plot synchrony dashboard |

## How to Explore This Codebase

1. Read `README.md` for user workflow and expected outputs.
2. Read `sync_eeg_vid.py` high-level functions:
   - `find_sync_from_raw_eeg`, `find_sync_from_csv`, `find_sync_pulse`
   - `sync_two_eeg_files`, `sync_eeg_to_video`, `sync_videos`
   - `extract_eeg_segment`, `main`
3. Read `tests/test_sync_eeg_vid.py` to understand protected behavior.
4. Validate inference outputs with `video_inference.schema.validate_session_output`.

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

- **Two-person internal mode:** parent/child heuristic with temporal IoU continuity.
- **Multi-person mode:** tracker-stable IDs (`person_00`, `person_01`, ...).
- Configured per run via `max_persons` and `enforce_exact_person_count`.

## Style

- Python 3.10+, Google-style docstrings, type hints on all signatures.
- `black` for formatting, `ruff` for linting (config in `pyproject.toml`).
- Favor CPU fallback paths by default unless GPU is explicitly available.
- No silent error swallowing -- fail loudly with clear messages.
- If introducing new outputs, define a stable schema and version it.
