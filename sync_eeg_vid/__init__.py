"""
EEG-Video Synchronization Tool

Synchronizes OpenBCI EEG recordings with video files using IR blaster pulses
and visual/audio cues. Optimized for long recordings (2+ hours).

Features:
- Frame-perfect video synchronization
- Multi-camera support
- Fast seeking for long videos
- Cross-platform compatibility
"""

from .cli import main, validate_file_paths
from .eeg_io import (
    extract_eeg_segment,
    find_sync_from_csv,
    find_sync_from_raw_eeg,
    find_sync_pulse,
)
from .plotting import plot_eeg_data, plot_sync_timeline
from .sync_pipeline import sync_eeg_to_video, sync_two_eeg_files, sync_videos
from .viewer import VideoFrameViewer

__all__ = [
    # eeg_io
    "extract_eeg_segment",
    "find_sync_from_csv",
    "find_sync_from_raw_eeg",
    "find_sync_pulse",
    # cli
    "main",
    "validate_file_paths",
    # plotting
    "plot_eeg_data",
    "plot_sync_timeline",
    # sync_pipeline
    "sync_eeg_to_video",
    "sync_two_eeg_files",
    "sync_videos",
    # viewer
    "VideoFrameViewer",
]
