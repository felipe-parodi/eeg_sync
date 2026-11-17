"""Where the project keeps code, data, models and logs.

Layout (everything except ``src/``, ``docs/``, ``tests/``, ``scripts/`` and
``third_party/`` is local and gitignored)::

    <repo>/
      src/<packages>        code
      data/output/          processed sessions (one folder per session)
      data/uploads/         portal uploads (chunked video files per job)
      data/compressed/      compressed copies made by the pipeline
      data/videos/          collaborator-compressed session videos
      data/training/        pose-training data (labels, caches, backups)
      data/runs/            Ultralytics training runs
      models/               model weights (*.pt)
      logs/                 portal tunnel and server logs
      third_party/          git submodules

``EEG_SYNC_DATA_DIR`` moves the whole data folder elsewhere (e.g. a bigger
drive) without touching code.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("EEG_SYNC_DATA_DIR", REPO_ROOT / "data"))
OUTPUT_DIR = DATA_DIR / "output"
UPLOAD_DIR = DATA_DIR / "uploads"
COMPRESSED_DIR = DATA_DIR / "compressed"
VIDEOS_DIR = DATA_DIR / "videos"
TRAINING_DATA_DIR = DATA_DIR / "training"
RUNS_DIR = DATA_DIR / "runs"
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
