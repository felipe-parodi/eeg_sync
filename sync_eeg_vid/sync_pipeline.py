"""High-level synchronization pipelines for EEG and video data."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .eeg_io import find_sync_pulse
from .viewer import VideoFrameViewer


def sync_two_eeg_files(
    eeg_filepath_a: Union[str, Path], eeg_filepath_b: Union[str, Path]
) -> Optional[Dict[str, Any]]:
    """
    Synchronize two EEG recordings using IR blaster pulses.

    Both EEG files should have IR blaster pulses at approximately the same
    real-world time. This calculates the offset between them.

    Args:
        eeg_filepath_a: Path to first EEG .txt file (reference)
        eeg_filepath_b: Path to second EEG .txt file

    Returns:
        Dict with synchronization results, or None if error
    """
    print("=" * 60)
    print("STEP 1: Synchronizing Two EEG Recordings")
    print("=" * 60)

    # Find IR pulse in EEG A (prefers clean CSV files)
    print("\n[1/2] Analyzing EEG file A for IR blaster pulse...")
    try:
        eeg_sync_time_a, source_a = find_sync_pulse(eeg_filepath_a)
        print(f"  Using: {source_a}")
        print(
            f"✓ IR blaster pulse found at {eeg_sync_time_a:.4f} seconds "
            "in EEG file A"
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error in EEG file A: {e}")
        return None

    # Find IR pulse in EEG B (prefers clean CSV files)
    print("\n[2/2] Analyzing EEG file B for IR blaster pulse...")
    try:
        eeg_sync_time_b, source_b = find_sync_pulse(eeg_filepath_b)
        print(f"  Using: {source_b}")
        print(
            f"✓ IR blaster pulse found at {eeg_sync_time_b:.4f} seconds "
            "in EEG file B"
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error in EEG file B: {e}")
        return None

    # Calculate offset
    offset = eeg_sync_time_a - eeg_sync_time_b

    result = {
        "eeg_file_a": str(eeg_filepath_a),
        "eeg_file_b": str(eeg_filepath_b),
        "eeg_sync_time_a": eeg_sync_time_a,
        "eeg_sync_time_b": eeg_sync_time_b,
        "offset": offset,
        "note": (
            f"To convert EEG B time to EEG A time: " f"time_a = time_b + {offset:.4f}"
        ),
    }

    print("\n✓ EEG synchronization complete!")
    print(f"  EEG A IR pulse: {eeg_sync_time_a:.4f} seconds")
    print(f"  EEG B IR pulse: {eeg_sync_time_b:.4f} seconds")
    print(f"  Time offset:    {offset:.4f} seconds")

    return result


def sync_eeg_to_video(
    eeg_filepath: Union[str, Path],
    video_path: Union[str, Path],
    approx_video_time: Optional[Union[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Synchronize EEG data to video using IR blaster pulse.

    Workflow:
    1. Find IR blaster pulse in EEG data
    2. User finds exact frame of red light in video
    3. Calculate time offset

    Args:
        eeg_filepath: Path to raw EEG .txt file
        video_path: Path to video file
        approx_video_time: Approximate timestamp of red light in video

    Returns:
        Dict with synchronization results, or None if cancelled

    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If file formats are invalid
    """
    print("=" * 60)
    print("STEP 1: Synchronizing EEG to Video")
    print("=" * 60)

    # Find IR pulse in EEG (prefers clean CSV files)
    print("\n[1/3] Analyzing EEG data for IR blaster pulse...")
    try:
        eeg_sync_time, source = find_sync_pulse(eeg_filepath)
        print(f"  Using: {source}")
        print(f"✓ IR blaster pulse found at {eeg_sync_time:.4f} seconds " "in EEG data")
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error: {e}")
        return None

    # Find red light in video
    print("\n[2/3] Finding exact frame of red light in video...")
    if approx_video_time:
        print(f"Starting at approximate time: {approx_video_time}")

    try:
        viewer = VideoFrameViewer(video_path)
        video_result = viewer.find_frame(
            window_name="Find RED LIGHT sync", start_timestamp=approx_video_time
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error: {e}")
        return None

    if not video_result or not video_result.get("marked"):
        print("✗ Synchronization cancelled")
        return None

    video_sync_time = video_result["timestamp"]
    print(f"✓ Red light found at {video_sync_time:.4f} seconds in video")

    # Calculate offset
    print("\n[3/3] Calculating synchronization offset...")
    offset = video_sync_time - eeg_sync_time

    result = {
        "eeg_file": str(eeg_filepath),
        "video_file": str(video_path),
        "eeg_sync_time": eeg_sync_time,
        "video_sync_time": video_sync_time,
        "video_frame": video_result["frame_number"],
        "offset": offset,
        "note": (
            f"To convert EEG time to video time: "
            f"video_time = eeg_time + {offset:.4f}"
        ),
    }

    print("\n✓ Synchronization complete!")
    print(f"  EEG IR pulse:    {eeg_sync_time:.4f} seconds")
    print(
        f"  Video red light: {video_sync_time:.4f} seconds "
        f"(frame {video_result['frame_number']})"
    )
    print(f"  Time offset:     {offset:.4f} seconds")

    return result


def sync_videos(
    video_a_path: Union[str, Path],
    video_b_path: Union[str, Path],
    approx_time_a: Optional[Union[str, float]] = None,
    approx_time_b: Optional[Union[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Synchronize two videos using multiple claps/sync events.

    Workflow:
    1. Mark claps in Video A (C to mark, U to undo, ENTER to save)
    2. Mark same number of claps in Video B
    3. Calculate offset using average of all clap pairs

    Args:
        video_a_path: Path to first video (reference)
        video_b_path: Path to second video
        approx_time_a: Approximate timestamp of claps in video A
        approx_time_b: Approximate timestamp of claps in video B

    Returns:
        Dict with synchronization results, or None if cancelled
    """
    from .util import ask_yes_no

    print("\n" + "=" * 60)
    print("STEP 2: Synchronizing Video B to Video A (Multi-Clap)")
    print("=" * 60)

    # Mark claps in Video A
    print("\n[1/3] Marking claps in Video A...")
    if approx_time_a:
        print(f"Starting at approximate time: {approx_time_a}")

    try:
        viewer_a = VideoFrameViewer(video_a_path)
        claps_a = viewer_a.find_multiple_frames(
            window_name="Mark CLAPS in Video A", start_timestamp=approx_time_a
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error: {e}")
        return None

    if not claps_a:
        print("✗ Synchronization cancelled (no claps marked in Video A)")
        return None

    num_claps = len(claps_a)
    print(f"\n✓ Marked {num_claps} clap(s) in Video A:")
    for i, clap in enumerate(claps_a, 1):
        print(f"  Clap {i}: {clap['time_formatted']} (frame {clap['frame_number']})")

    # Mark claps in Video B (with retry if counts don't match)
    claps_b = None
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        print(f"\n[2/3] Marking {num_claps} clap(s) in Video B...")
        print(f"⚠ Please mark the SAME {num_claps} clap(s) in Video B")
        if approx_time_b:
            print(f"Starting at approximate time: {approx_time_b}")

        try:
            viewer_b = VideoFrameViewer(video_b_path)
            claps_b = viewer_b.find_multiple_frames(
                window_name="Mark CLAPS in Video B",
                start_timestamp=approx_time_b,
                expected_count=num_claps,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ Error: {e}")
            return None

        if not claps_b:
            print("✗ Synchronization cancelled (no claps marked in Video B)")
            return None

        # Check if counts match
        if len(claps_b) == num_claps:
            break  # Success!
        else:
            print(
                f"\n⚠ Mismatch: Video A has {num_claps} claps, but you marked {len(claps_b)} in Video B"
            )
            if ask_yes_no("  Do you want to re-mark the claps in Video B?"):
                retry_count += 1
                continue
            else:
                print("  Proceeding with mismatched clap counts...")
                break

    print(f"\n✓ Marked {len(claps_b)} clap(s) in Video B:")
    for i, clap in enumerate(claps_b, 1):
        print(f"  Clap {i}: {clap['time_formatted']} (frame {clap['frame_number']})")

    # Calculate offset(s)
    print("\n[3/3] Calculating synchronization offset...")

    if len(claps_b) != num_claps:
        print(
            f"⚠ Warning: Different number of claps "
            f"(A: {num_claps}, B: {len(claps_b)})"
        )

    # Calculate offset for each clap pair
    offsets = []
    for i in range(min(num_claps, len(claps_b))):
        offset_i = claps_a[i]["timestamp"] - claps_b[i]["timestamp"]
        offsets.append(offset_i)
        print(
            f"  Clap {i+1} offset: {offset_i:.4f}s "
            f"(A: {claps_a[i]['time_formatted']}, "
            f"B: {claps_b[i]['time_formatted']})"
        )

    # Use average offset
    avg_offset = sum(offsets) / len(offsets)

    # Calculate offset variability
    if len(offsets) > 1:
        offset_std = np.std(offsets)
        print(f"\n  Average offset: {avg_offset:.4f}s")
        print(f"  Std deviation:  {offset_std:.4f}s")
        if offset_std > 0.1:
            print(f"  ⚠ Warning: High variability in offsets (>{0.1}s)")
    else:
        print(f"\n  Offset: {avg_offset:.4f}s")

    result = {
        "video_a": str(video_a_path),
        "video_b": str(video_b_path),
        "claps_a": claps_a,
        "claps_b": claps_b,
        "offsets": offsets,
        "offset": avg_offset,
        "offset_std": np.std(offsets) if len(offsets) > 1 else 0.0,
        "note": (
            f"To convert Video B time to Video A time: "
            f"time_a = time_b + {avg_offset:.4f}"
        ),
    }

    print("\n✓ Video synchronization complete!")
    print(f"  Final offset: {avg_offset:.4f} seconds")

    return result
