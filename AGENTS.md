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

### Core Sync Module (`sync_eeg_vid.py`)

Single-file module (~2450 lines) containing the entire Phase 1 pipeline:
- **EEG I/O:** Parses OpenBCI TXT (raw) and CSV (cleaned IR) files. Auto-prefers CSV when available. IR blaster on `Analog Channel 0`, baseline value = 257, sample rate = 250 Hz.
- **Interactive video viewer:** OpenCV window with 1ms keyboard polling, frame-perfect navigation, auto-downscaling for >1920px videos. Two modes: red-light (single mark) and multi-clap (mark/undo/save).
- **Sync pipeline:** `sync_two_eeg_files` -> `sync_eeg_to_video` -> `sync_videos` -> JSON + PNG export.

Key entry functions: `find_sync_pulse`, `find_sync_from_raw_eeg`, `find_sync_from_csv`, `extract_eeg_segment`, `main`.

### Video Inference (`video_inference/`)

Pipeline stages run in sequence: **compress -> extract frames -> infer -> track -> schema export -> validate**.

- **`pipeline.py`:** Orchestrator CLI (`video-infer run`). Wires together all stages per camera.
- **`ultralytics_runner.py`:** YOLOv11 2D pose backend. Two tracking modes: internal (parent/child heuristic by bbox area) and external (Roboflow ByteTrack with ID slot recycling).
- **`sam3d_runner.py`:** SAM-3D-Body 3D pose backend. Patches upstream CUDA hardcoding for CPU fallback. Imports from `third_party/sam-3d-body/` submodule. Override path via `EEG_SYNC_SAM3D_ROOT` env var.
- **`tracking.py`:** `TwoPersonTrackerState` with IoU-based temporal continuity. Frame 0 uses area prior (parent = largest bbox), subsequent frames use IoU matching.
- **`schema.py`:** Validates output CSVs (manifest, `tracks_2d.csv`, `pose_3d.csv`). Returns `ValidationResult` with error list, not exceptions.
- **`device.py`:** `resolve_device("auto")` returns `"cpu"` or `"cuda"`. `resolve_inference_mode` selects `"full"` vs `"body"` based on hardware.

### Video Analysis (`video_analysis/`)

- **`interpolate.py`:** Cubic spline upsampling of low-FPS track/pose data (e.g., 1 FPS -> 8 FPS).
- **`visualize_pose_tracks.py`:** Renders COCO17 skeletons + bounding boxes + track IDs onto video frames.

### CLI Entry Points (all in `pyproject.toml`)

| Command | Purpose |
|---|---|
| `eeg-sync` | Interactive EEG-video synchronization |
| `video-infer run` | End-to-end pose inference pipeline |
| `video-infer-sam3d` | SAM-3D 3D pose (standalone) |
| `video-infer-ultralytics` | Ultralytics 2D pose (standalone) |
| `video-compress-rapid` | Batch video compression |
| `video-interpolate` | Upsample low-FPS outputs |
| `video-visualize` | Pose overlay visualization |

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
