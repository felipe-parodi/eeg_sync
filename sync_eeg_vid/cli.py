"""Command-line interface for EEG-video synchronization."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .plotting import plot_sync_timeline
from .sync_pipeline import sync_eeg_to_video, sync_two_eeg_files, sync_videos
from .util import (
    _strip_quotes,
    _validate_file_exists,
    ask_file_path,
    ask_yes_no,
)


def collect_all_files() -> Dict[str, Optional[str]]:
    """
    Collect all file paths from user upfront.

    Returns:
        Dict with keys: eeg_file_a, eeg_file_b, video_a, video_b
    """
    print("\n" + "=" * 60)
    print("FILE SELECTION")
    print("=" * 60)
    print("\nPlease provide paths to your files.")
    print("Required: At least 1 EEG file AND 1 Video file")
    print("Optional: Second EEG and/or second Video")
    print("\nYou can:")
    print("  - Drag and drop files into the terminal")
    print("  - Type/paste the full file path")
    print("  - Press ENTER to skip optional files")
    print("\n⚠️  IMPORTANT: If your path contains spaces, enclose it in quotes")
    print("   Example: \"path/to/my folder/data.csv\"")

    files = {}

    # EEG File A
    print("\n" + "-" * 60)
    files["eeg_file_a"] = ask_file_path(
        "EEG File A (.csv/.txt) - REQUIRED", must_exist=True
    )

    # EEG File B (optional)
    print("\n" + "-" * 60)
    response = input(
        "EEG File B (.csv/.txt) - OPTIONAL (press ENTER to skip): "
    ).strip()
    if response:
        try:
            response = _strip_quotes(response)
            files["eeg_file_b"] = str(_validate_file_exists(response))
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠ Warning: {e}")
            print("  → Skipping EEG File B")
            files["eeg_file_b"] = None
    else:
        files["eeg_file_b"] = None

    # Video A
    print("\n" + "-" * 60)
    files["video_a"] = ask_file_path(
        "Video A (.mp4/.mov/.avi) - REQUIRED (with red light sync)", must_exist=True
    )

    # Video B (optional)
    print("\n" + "-" * 60)
    response = input(
        "Video B (.mp4/.mov/.avi) - OPTIONAL (press ENTER to skip): "
    ).strip()
    if response:
        try:
            response = _strip_quotes(response)
            files["video_b"] = str(_validate_file_exists(response))
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠ Warning: {e}")
            print("  → Skipping Video B")
            files["video_b"] = None
    else:
        files["video_b"] = None

    return files


def validate_file_paths(files: Dict[str, Optional[str]]) -> Tuple[bool, List[str]]:
    """
    Validate collected file paths.

    Args:
        files: Dict with file paths

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check minimum requirements
    if not files.get("eeg_file_a"):
        errors.append("❌ At least one EEG file is required")
    if not files.get("video_a"):
        errors.append("❌ At least one Video file is required")

    # Check for duplicates
    provided_files = {k: v for k, v in files.items() if v}
    file_paths = list(provided_files.values())
    file_labels = list(provided_files.keys())

    for i in range(len(file_paths)):
        for j in range(i + 1, len(file_paths)):
            if file_paths[i] == file_paths[j]:
                errors.append(
                    f"❌ Duplicate file: {file_labels[i]} and {file_labels[j]} "
                    f"are using the same file"
                )

    # Validate extensions
    eeg_exts = [".csv", ".txt"]
    video_exts = [".mp4", ".mov", ".avi"]

    for key, filepath in files.items():
        if filepath:
            path = Path(filepath)
            if "eeg" in key.lower():
                if path.suffix.lower() not in eeg_exts:
                    errors.append(
                        f"⚠️ {key}: Expected EEG file (.csv/.txt), got {path.suffix}"
                    )
            elif "video" in key.lower():
                if path.suffix.lower() not in video_exts:
                    errors.append(
                        f"⚠️ {key}: Expected video file (.mp4/.mov/.avi), got {path.suffix}"
                    )

    return len(errors) == 0, errors


def determine_workflow(files: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Determine which sync steps will run based on provided files.

    Args:
        files: Dict with file paths

    Returns:
        Dict with workflow info
    """
    workflow = {
        "sync_eeg_to_eeg": False,
        "sync_eeg_to_video": False,
        "sync_video_to_video": False,
        "steps": [],
    }

    # Check what syncs are possible
    if files.get("eeg_file_a") and files.get("eeg_file_b"):
        workflow["sync_eeg_to_eeg"] = True
        workflow["steps"].append("✓ Step 1: Sync EEG A ↔ EEG B")

    if (files.get("eeg_file_a") or files.get("eeg_file_b")) and files.get("video_a"):
        workflow["sync_eeg_to_video"] = True
        eeg_label = "EEG A" if files.get("eeg_file_a") else "EEG B"
        workflow["steps"].append(f"✓ Step 2: Sync {eeg_label} → Video A")

    if files.get("video_a") and files.get("video_b"):
        workflow["sync_video_to_video"] = True
        workflow["steps"].append("✓ Step 3: Sync Video A ↔ Video B")

    return workflow


def ask_red_light_timestamp() -> Optional[str]:
    """
    Ask for red light timestamp hint for EEG→Video sync.

    Returns:
        Timestamp hint or None
    """
    print("\n" + "=" * 60)
    print("RED LIGHT TIMESTAMP HINT")
    print("=" * 60)
    print("\nTo help locate the red light sync point in the video:")
    print("Format: 'M:SS' (e.g., '1:23') or seconds (e.g., '83')")
    print("Press ENTER to start at 0:00")

    response = input("\nApproximate time of RED LIGHT in Video A: ").strip()
    return response if response else None


def ask_clap_timestamps() -> Tuple[Optional[str], Optional[str]]:
    """
    Ask for clap timestamp hints for Video A↔Video B sync.

    Returns:
        Tuple of (video_a_hint, video_b_hint)
    """
    print("\n" + "=" * 60)
    print("CLAP TIMESTAMP HINTS")
    print("=" * 60)
    print("\nTo help locate the clap sync points in both videos:")
    print("Format: 'M:SS' (e.g., '2:30') or seconds (e.g., '150')")
    print("Press ENTER to start at 0:00 for any field.")

    print("\n" + "-" * 60)
    response_a = input("Approximate time of CLAPS in Video A: ").strip()
    approx_a = response_a if response_a else None

    print("\n" + "-" * 60)
    response_b = input("Approximate time of CLAPS in Video B: ").strip()
    approx_b = response_b if response_b else None

    return approx_a, approx_b


def main():
    """
    Main interactive workflow for EEG-video synchronization.

    Workflow:
    1. Collect all file paths upfront
    2. Validate files
    3. Show workflow preview
    4. Collect timestamp hints
    5. Run synchronization steps
    6. Save results
    """
    import json

    print("\n" + "=" * 60)
    print("EEG-VIDEO SYNCHRONIZATION TOOL")
    print("=" * 60)

    # Step 1: Collect all files
    files = collect_all_files()

    # Step 2: Validate files
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    is_valid, errors = validate_file_paths(files)

    if not is_valid:
        print("\n⚠️ Validation errors found:")
        for error in errors:
            print(f"  {error}")
        print("\nPlease fix the errors and try again.")
        return

    print("✓ All files validated successfully!")

    # Step 3: Determine workflow
    workflow = determine_workflow(files)

    print("\n" + "=" * 60)
    print("WORKFLOW PREVIEW")
    print("=" * 60)
    print("\nBased on your files, the following sync steps will run:\n")
    for step in workflow["steps"]:
        print(f"  {step}")

    # Confirm to proceed
    if not ask_yes_no("\nProceed with synchronization?"):
        print("Cancelled.")
        return

    # Step 4: Run synchronization workflow
    print("\n" + "=" * 60)
    print("SYNCHRONIZATION")
    print("=" * 60)

    eeg_eeg_sync = None
    eeg_sync = None
    video_sync = None

    # Determine which EEG file to use as reference
    eeg_ref_file = files.get("eeg_file_a") or files.get("eeg_file_b")

    # Step 4a: Sync EEG A ↔ EEG B (no timestamp hint needed - uses IR pulse)
    if workflow["sync_eeg_to_eeg"]:
        try:
            eeg_eeg_sync = sync_two_eeg_files(files["eeg_file_a"], files["eeg_file_b"])
        except (FileNotFoundError, ValueError) as e:
            print(f"\n✗ Error syncing EEG files: {e}")
            print("\nExiting...")
            return

        if eeg_eeg_sync is None:
            print("\nEEG sync cancelled. Exiting...")
            return

    # Step 4b: Sync EEG → Video A (ask for red light timestamp just-in-time)
    if workflow["sync_eeg_to_video"]:
        # Ask for timestamp hint NOW (right before opening video)
        approx_red = ask_red_light_timestamp()

        try:
            eeg_sync = sync_eeg_to_video(eeg_ref_file, files["video_a"], approx_red)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n✗ Error syncing EEG to video: {e}")
            print("\nExiting...")
            return

        if eeg_sync is None:
            print("\nEEG-Video sync cancelled. Exiting...")
            return

    # Step 4c: Sync Video A ↔ Video B (ask for clap timestamps just-in-time)
    if workflow["sync_video_to_video"]:
        # Ask for timestamp hints NOW (right before opening videos)
        approx_claps_a, approx_claps_b = ask_clap_timestamps()

        try:
            video_sync = sync_videos(
                files["video_a"], files["video_b"], approx_claps_a, approx_claps_b
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"\n✗ Error syncing videos: {e}")
            print("Continuing without video sync...")

    # Print summary
    print("\n" + "=" * 60)
    print("SYNCHRONIZATION SUMMARY")
    print("=" * 60)

    if eeg_eeg_sync:
        print(f"\nEEG File A: {eeg_eeg_sync['eeg_file_a']}")
        print(f"EEG File B: {eeg_eeg_sync['eeg_file_b']}")
        print("\nEEG A ↔ EEG B:")
        print(f"  Offset: {eeg_eeg_sync['offset']:.4f} seconds")
        print(f"  {eeg_eeg_sync['note']}")
    else:
        print(f"\nEEG File: {eeg_ref_file}")

    if eeg_sync:
        print(f"Video A:  {files['video_a']}")
        print("\nEEG ↔ Video A:")
        print(f"  Offset: {eeg_sync['offset']:.4f} seconds")
        print(f"  {eeg_sync['note']}")

    if video_sync:
        print(f"\nVideo B:  {video_sync['video_b']}")
        print("\nVideo A ↔ Video B:")
        print(f"  Offset: {video_sync['offset']:.4f} seconds")
        print(f"  {video_sync['note']}")

    # Save results
    save = ask_yes_no("\nSave synchronization results to file?")
    if save:
        output_file = "sync_results.json"
        results = {
            "eeg_a_to_eeg_b": eeg_eeg_sync,
            "eeg_to_video_a": eeg_sync,
            "video_a_to_video_b": video_sync,
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {output_file}")

    # Offer to generate plots
    if eeg_sync:  # Only offer if we have EEG-video sync
        plot = ask_yes_no("\nGenerate synchronization timeline plot?")
        if plot:
            try:
                plot_sync_timeline(eeg_sync, video_sync, eeg_eeg_sync)
                plot_file = "sync_timeline.png"
                import matplotlib.pyplot as plt

                plt.savefig(plot_file, dpi=150, bbox_inches="tight")
                print(f"✓ Plot saved to {plot_file}")
                plt.close()
            except Exception as e:
                print(f"⚠ Could not generate plot: {e}")

    # Show how to use the data
    print("\n" + "=" * 60)
    print("HOW TO USE THIS DATA")
    print("=" * 60)
    print(
        """
To use the synchronization offsets in your analysis:

1. Load the sync results:
   import json
   with open('sync_results.json') as f:
       sync = json.load(f)

2. Convert timestamps:
   # EEG B time → EEG A time
   eeg_a_time = eeg_b_time + sync['eeg_a_to_eeg_b']['offset']

   # EEG time → Video time
   video_time = eeg_time + sync['eeg_to_video_a']['offset']

3. Extract aligned data:
   from sync_eeg_vid import extract_eeg_segment

   # Get EEG data for video segment 1:30 to 2:00
   eeg_data = extract_eeg_segment(
       eeg_filepath='path/to/eeg.txt',
       video_start=90,   # 1:30 in seconds
       video_end=120,    # 2:00 in seconds
       eeg_video_offset=sync['eeg_to_video_a']['offset']
   )
   # Returns DataFrame with Time_EEG and Time_Video columns

See README.md for more examples!
"""
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
