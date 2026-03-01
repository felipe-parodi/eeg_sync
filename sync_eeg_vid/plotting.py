"""Plotting and visualization for EEG synchronization data."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .eeg_io import DEFAULT_IR_COLUMN, NUM_EEG_CHANNELS, _read_eeg_header
from .util import _validate_file_exists


def plot_sync_timeline(
    eeg_sync_result: Dict[str, Any],
    video_sync_result: Optional[Dict[str, Any]] = None,
    eeg_eeg_sync_result: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (14, 8),
):
    """
    Visualize synchronization timeline between EEG(s) and video(s).

    Shows all data streams:
    - Dual EEG: EEG A, EEG B, Video A, Video B (4 streams)
    - Single EEG: EEG, Video A, Video B (3 streams)

    Args:
        eeg_sync_result: Result from sync_eeg_to_video()
        video_sync_result: Optional result from sync_videos()
        eeg_eeg_sync_result: Optional result from sync_two_eeg_files()
        figsize: Figure size tuple

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Determine layout based on available data
    has_dual_eeg = eeg_eeg_sync_result is not None
    has_video_b = video_sync_result is not None

    # Y-axis positions (bottom to top)
    y_pos = 0
    yticks = []
    yticklabels = []

    # Draw Video B (if present) - bottom track
    if has_video_b:
        if "claps_b" in video_sync_result:
            # Multi-clap sync
            claps_b = video_sync_result["claps_b"]
            max_time_b = max(c["timestamp"] for c in claps_b)
            ax.barh(
                y_pos,
                max_time_b,
                left=0,
                height=0.4,
                color="#9b59b6",
                alpha=0.7,
                label="Video B",
            )
            for i, clap in enumerate(claps_b):
                ax.plot(
                    clap["timestamp"],
                    y_pos,
                    "g^",
                    markersize=8,
                    label=f"Clap {i+1} (Video B)" if i == 0 else "",
                )
        yticks.append(y_pos)
        yticklabels.append("Video B")
        y_pos += 1

    # Draw Video A
    video_a_time = eeg_sync_result["video_sync_time"]
    ax.barh(
        y_pos,
        video_a_time,
        left=0,
        height=0.4,
        color="#e74c3c",
        alpha=0.7,
        label="Video A",
    )
    ax.plot(video_a_time, y_pos, "r*", markersize=12, label="Red Light (Video A)")

    # Add claps if present
    if has_video_b and "claps_a" in video_sync_result:
        for i, clap in enumerate(video_sync_result["claps_a"]):
            ax.plot(
                clap["timestamp"],
                y_pos,
                "g^",
                markersize=8,
                label=f"Clap {i+1} (Video A)" if i == 0 else "",
            )

    yticks.append(y_pos)
    yticklabels.append("Video A")
    y_pos += 1

    # Draw EEG B (if dual EEG) - second from top
    if has_dual_eeg:
        eeg_b_time = eeg_eeg_sync_result["eeg_sync_time_b"]
        ax.barh(
            y_pos,
            eeg_b_time,
            left=0,
            height=0.4,
            color="#2ecc71",
            alpha=0.7,
            label="EEG B",
        )
        ax.plot(eeg_b_time, y_pos, "ro", markersize=8, label="IR Pulse (EEG B)")
        yticks.append(y_pos)
        yticklabels.append("EEG B")
        y_pos += 1

    # Draw EEG A - top track
    eeg_a_time = eeg_sync_result["eeg_sync_time"]
    ax.barh(
        y_pos,
        eeg_a_time,
        left=0,
        height=0.4,
        color="#3498db",
        alpha=0.7,
        label="EEG A" if has_dual_eeg else "EEG",
    )
    ax.plot(
        eeg_a_time,
        y_pos,
        "ro",
        markersize=8,
        label="IR Pulse (EEG A)" if has_dual_eeg else "IR Pulse",
    )
    yticks.append(y_pos)
    yticklabels.append("EEG A" if has_dual_eeg else "EEG")

    # Draw sync arrows and labels
    # EEG A/B sync (if dual EEG)
    if has_dual_eeg:
        eeg_b_y = yticks[-2]  # EEG B position
        eeg_a_y = yticks[-1]  # EEG A position
        eeg_b_time = eeg_eeg_sync_result["eeg_sync_time_b"]
        eeg_a_time = eeg_eeg_sync_result["eeg_sync_time_a"]
        eeg_offset = eeg_eeg_sync_result["offset"]

        ax.annotate(
            "",
            xy=(eeg_a_time, eeg_a_y - 0.25),
            xytext=(eeg_b_time, eeg_b_y + 0.25),
            arrowprops=dict(arrowstyle="<->", color="purple", lw=2),
        )
        ax.text(
            (eeg_a_time + eeg_b_time) / 2,
            (eeg_a_y + eeg_b_y) / 2,
            f"Δ={eeg_offset:.3f}s",
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    # EEG → Video A sync
    video_a_y = yticks[1 if has_video_b else 0]  # Video A position
    eeg_ref_y = yticks[-1]  # Top EEG position
    eeg_to_vid_offset = eeg_sync_result["offset"]

    ax.annotate(
        "",
        xy=(video_a_time, video_a_y + 0.25),
        xytext=(eeg_a_time, eeg_ref_y - 0.25),
        arrowprops=dict(arrowstyle="<->", color="green", lw=2),
    )
    ax.text(
        (eeg_a_time + video_a_time) / 2,
        (video_a_y + eeg_ref_y) / 2,
        f"Δ={eeg_to_vid_offset:.3f}s",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Video A → Video B sync (if present)
    if has_video_b:
        video_b_y = yticks[0]
        v_offset = video_sync_result["offset"]

        # Use first clap times for arrow
        if "claps_a" in video_sync_result and "claps_b" in video_sync_result:
            sync_time_a = video_sync_result["claps_a"][0]["timestamp"]
            sync_time_b = video_sync_result["claps_b"][0]["timestamp"]
        else:
            sync_time_a = video_sync_result.get("sync_time_a", video_a_time)
            sync_time_b = video_sync_result.get("sync_time_b", 0)

        ax.annotate(
            "",
            xy=(sync_time_a, video_a_y - 0.25),
            xytext=(sync_time_b, video_b_y + 0.25),
            arrowprops=dict(arrowstyle="<->", color="orange", lw=2),
        )

        # Show offset with std if multiple claps
        if "offset_std" in video_sync_result and video_sync_result["offset_std"] > 0:
            offset_text = (
                f"Δ={v_offset:.3f}s\n±{video_sync_result['offset_std']*1000:.0f}ms"
            )
        else:
            offset_text = f"Δ={v_offset:.3f}s"

        ax.text(
            (sync_time_a + sync_time_b) / 2,
            (video_a_y + video_b_y) / 2,
            offset_text,
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    # Styling
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time (seconds)", fontsize=12)

    title_parts = []
    if has_dual_eeg:
        title_parts.append("Dual EEG")
    if has_video_b:
        title_parts.append("Dual Video")
    title = (
        " + ".join(title_parts) + " Synchronization"
        if title_parts
        else "EEG-Video Synchronization"
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    return fig


def plot_eeg_data(
    eeg_filepath: Union[str, Path],
    start_time: float = 0,
    duration: float = 10,
    channels: Optional[List[int]] = None,
    sync_time: Optional[float] = None,
    show_ir: bool = True,
):
    """
    Plot EEG channels from raw data file.

    Args:
        eeg_filepath: Path to raw EEG .txt file
        start_time: Start time in seconds
        duration: Duration to plot in seconds
        channels: List of channel indices to plot (None = all)
        sync_time: If provided, mark the sync point with vertical line
        show_ir: Whether to show IR blaster channel

    Returns:
        Matplotlib figure

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    import matplotlib.pyplot as plt

    filepath = _validate_file_exists(eeg_filepath)

    # Read header
    sample_rate, header_lines, column_names = _read_eeg_header(filepath)

    # Read data
    data = pd.read_csv(
        filepath, skiprows=header_lines, names=column_names, low_memory=False
    )

    # Calculate sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)

    # Extract segment
    segment = data.iloc[start_sample:end_sample]
    time_axis = np.arange(len(segment)) / sample_rate + start_time

    # Determine channels
    if channels is None:
        channels = list(range(NUM_EEG_CHANNELS))
    eeg_channels = [f"EXG Channel {i}" for i in channels]

    # Create subplots
    n_plots = len(eeg_channels) + (1 if show_ir else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    # Plot EEG channels
    for i, ch_name in enumerate(eeg_channels):
        if ch_name in segment.columns:
            axes[i].plot(time_axis, segment[ch_name], linewidth=0.5)
            axes[i].set_ylabel(ch_name.replace("EXG Channel ", "Ch "), fontsize=10)
            axes[i].grid(alpha=0.3)

            # Mark sync point
            if (
                sync_time is not None
                and start_time <= sync_time <= start_time + duration
            ):
                axes[i].axvline(
                    sync_time,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Sync Point",
                )
                if i == 0:
                    axes[i].legend(loc="upper right")

    # Plot IR channel
    if show_ir and DEFAULT_IR_COLUMN in segment.columns:
        axes[-1].plot(
            time_axis, segment[DEFAULT_IR_COLUMN], linewidth=1, color="orange"
        )
        axes[-1].set_ylabel("IR Blaster", fontsize=10)
        axes[-1].grid(alpha=0.3)

        if sync_time is not None and start_time <= sync_time <= start_time + duration:
            axes[-1].axvline(sync_time, color="red", linestyle="--", alpha=0.7)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    fig.suptitle(
        f"EEG Data: {start_time}s to {start_time + duration}s",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig
