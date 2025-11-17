# EEG-Video Synchronization Tool

**ULTRA-FAST** synchronization tool for aligning OpenBCI EEG data with GoPro (or any) video recordings. Optimized for long recordings (2+ hours).

## Features

- ⚡ **Lightning Fast**: Efficient frame seeking, no loading entire videos into memory
- 🎯 **Frame-Perfect Accuracy**: Navigate frame-by-frame to find exact sync points
- 📊 **Visualization**: Plot EEG timelines and sync points
- 🎬 **Multi-Video Support**: Sync EEG with multiple cameras using clap/audio cues
- 📝 **Export Results**: Save sync offsets as JSON for reproducible analysis

## Quick Start

### Installation

**Option 1: uv (Recommended - FAST!)**

[uv](https://github.com/astral-sh/uv) is 10-100x faster than pip/conda:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project
cd asilver_eeg_sync

# Create venv and install dependencies (takes ~5 seconds!)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**Option 2: pip**
```bash
pip install -e .
# or
pip install -r requirements.txt
```

### Basic Usage

**Quick Start:**

```bash
python sync_eeg_vid.py
```

The script will guide you through:
1. **File Selection** - Provide all files upfront (EEG A/B, Video A/B)
2. **Validation** - Automatic file validation and workflow preview
3. **Dual EEG sync** (if you have 2 EEG files) - syncs two EEG recordings using IR pulse
4. **EEG-to-Video sync** - syncs EEG to video using red light (you'll be prompted for timestamp)
5. **Video-to-Video sync** (optional) - syncs second video using claps (you'll be prompted for timestamps)

**Files Accepted:**
- Can provide either `.txt` (raw) or `.csv` (cleaned IR data) files
- Script **automatically prefers CSV** files (cleaner, no noise)
- Falls back to TXT if CSV not found

## How It Works

### Synchronization Process

**For Dual-EEG Setup (e.g., two EEG headbands):**

**Step 1: EEG A ↔ EEG B (IR Blaster Sync)**
1. Script detects IR blaster pulse in both EEG files
2. Calculates time offset between the two EEG recordings
3. Uses **clean CSV files** (noise removed) for best accuracy

**Step 2: EEG → Video A (Red Light Sync)**
1. Script automatically detects IR blaster pulse in EEG data (~4ms precision)
2. You navigate video to find the exact frame of the red light
3. Script calculates time offset between EEG and video

**Step 3: Video A → Video B (Clap Sync)** *(optional)*
1. You find the exact frame of a clap/sync event in Video A
2. You find the same event in Video B
3. Script calculates offset between the two videos

**Result:** All 4 data sources synchronized to a common timeline!

### Interactive Video Viewer

The video viewer is optimized for SPEED with long videos:

**Navigation Controls (Red Light Mode):**
- `A/D` or `LEFT/RIGHT arrows`: Move ±1 frame (frame-perfect precision)
- `W/S` or `UP/DOWN arrows`: Jump ±1 second
- `,` / `.` keys: Jump ±10 seconds (fast seeking)
- `[` / `]` keys: Jump ±1 minute (very fast seeking)
- `SPACE`: Mark current frame and exit
- `Q` or `ESC`: Quit without marking

**Multi-Clap Marking Controls:**
- `C`: Mark clap at current frame (counter updates immediately!)
- `U`: Undo last marked clap
- `LEFT/RIGHT arrows`: Move ±1 frame (arrow keys only)
- `UP/DOWN arrows`: Jump ±1 second (arrow keys only)
- `,` / `.` keys: Jump ±10 seconds
- `[` / `]` keys: Jump ±1 minute
- `SPACE` or `ENTER`: Save marked claps and exit
- `Q` or `ESC`: Quit without saving

**Performance Features:**
- Auto-downscaling for large videos (faster display)
- Efficient frame seeking (no buffering)
- Frame caching for instant ±1 frame navigation
- 1ms keyboard polling for responsive controls
- Immediate visual feedback when marking claps

## Workflow Example

```bash
$ python sync_eeg_vid.py

============================================================
EEG-VIDEO SYNCHRONIZATION TOOL
============================================================

============================================================
FILE SELECTION
============================================================

Please provide paths to your files.
Required: At least 1 EEG file AND 1 Video file
Optional: Second EEG and/or second Video

You can:
  - Drag and drop files into the terminal
  - Type/paste the full file path
  - Press ENTER to skip optional files

------------------------------------------------------------
EEG File A (.csv) - REQUIRED: TEp_OpenBCI-RAW-2025-11-14_10-40-38.csv

------------------------------------------------------------
EEG File B (.csv) - OPTIONAL (press ENTER to skip): TEc_OpenBCI-RAW-2025-11-14_10-40-39.csv

------------------------------------------------------------
Video A (.mp4/.mov/.avi) - REQUIRED (with red light sync): gopro_main.mp4

------------------------------------------------------------
Video B (.mp4/.mov/.avi) - OPTIONAL (press ENTER to skip): gopro_angle2.mp4

============================================================
VALIDATION
============================================================

✓ All files validated successfully!

============================================================
WORKFLOW PREVIEW
============================================================

Based on your files, the following sync steps will run:

  ✓ Step 1: Sync EEG A ↔ EEG B
  ✓ Step 2: Sync EEG A → Video A
  ✓ Step 3: Sync Video A ↔ Video B

Proceed with synchronization? (y/n): y

============================================================
SYNCHRONIZATION
============================================================

============================================================
STEP 1: Synchronizing Two EEG Recordings
============================================================

[1/2] Analyzing EEG file A for IR blaster pulse...
  Using: CSV: TEp_OpenBCI-RAW-2025-11-14_10-40-38fixed_irBlaster.csv
✓ IR blaster pulse found at 81.3480 seconds in EEG file A

[2/2] Analyzing EEG file B for IR blaster pulse...
  Using: CSV: TEc_OpenBCI-RAW-2025-11-14_10-40-39fixed_irBlaster.csv
✓ IR blaster pulse found at 1.3320 seconds in EEG file B

✓ EEG synchronization complete!
  EEG A IR pulse: 81.3480 seconds
  EEG B IR pulse: 1.3320 seconds
  Time offset:    80.0160 seconds

============================================================
RED LIGHT TIMESTAMP HINT
============================================================

To help locate the red light sync point in the video:
Format: 'M:SS' (e.g., '1:23') or seconds (e.g., '83')
Press ENTER to start at 0:00

Approximate time of RED LIGHT in Video A: 1:21

============================================================
STEP 1: Synchronizing EEG to Video
============================================================

[1/3] Analyzing EEG data for IR blaster pulse...
  Using: CSV: TEp_OpenBCI-RAW-2025-11-14_10-40-38fixed_irBlaster.csv
✓ IR blaster pulse found at 81.3480 seconds in EEG data

[2/3] Finding exact frame of red light in video...

============================================================
OPENING VIDEO WINDOW - LOOK FOR IT ON YOUR SCREEN!
============================================================

Video: gopro_main.mp4
Duration: 120:35.12 | FPS: 29.97 | Frames: 217053
Auto-downscaling for speed: 3840px → 1920px
→ Jumped to 1:21.00 (frame 2430)

Controls:
  A/D or ← →:         ±1 frame
  W/S or ↑ ↓:         ±1 second
  , / . keys:         ±10 seconds
  [ / ] keys:         ±1 minute
  SPACE:              Mark frame and EXIT
  Q/ESC:              Quit

>>> VIDEO WINDOW OPENED: 'Find RED LIGHT sync' - Check your screen! <<<

# [Interactive video window opens - you navigate and press SPACE]

✓ Marked: Frame 2493 at 1:23.123 (83.1230s)
✓ Red light found at 83.1230 seconds in video

[3/3] Calculating synchronization offset...

✓ Synchronization complete!
  EEG IR pulse:    81.3480 seconds
  Video red light: 83.1230 seconds (frame 2493)
  Time offset:     1.7750 seconds

============================================================
CLAP TIMESTAMP HINTS
============================================================

To help locate the clap sync points in both videos:
Format: 'M:SS' (e.g., '2:30') or seconds (e.g., '150')
Press ENTER to start at 0:00 for any field.

------------------------------------------------------------
Approximate time of CLAPS in Video A: 2:45

------------------------------------------------------------
Approximate time of CLAPS in Video B: 0:23

============================================================
STEP 2: Synchronizing Video B to Video A (Multi-Clap)
============================================================

[1/3] Marking claps in Video A...

============================================================
MULTI-CLAP MARKING MODE
============================================================

Video: gopro_main.mp4
Duration: 120:35.12 | FPS: 29.97 | Frames: 217053

⚠ Please mark exactly 3 claps
→ Jumped to 2:45.00

Controls:
  C:                  Mark clap at current frame
  U:                  Undo last marked clap
  ENTER/SPACE:        Save and EXIT
  ← →:                ±1 frame
  ↑ ↓:                ±1 second
  , / . keys:         ±10 seconds
  [ / ] keys:         ±1 minute
  Q/ESC:              Quit without saving

# [Mark 3 claps in Video A, then same for Video B]

✓ Synchronization complete!
  Final offset: 142.5670 seconds

============================================================
SYNCHRONIZATION SUMMARY
============================================================

EEG File A: TEp_OpenBCI-RAW-2025-11-14_10-40-38.csv
EEG File B: TEc_OpenBCI-RAW-2025-11-14_10-40-39.csv

EEG A ↔ EEG B:
  Offset: 80.0160 seconds
  To convert EEG B time to EEG A time: time_a = time_b + 80.0160

Video A:  gopro_main.mp4

EEG ↔ Video A:
  Offset: 1.7750 seconds
  To convert EEG time to video time: video_time = eeg_time + 1.7750

Video B:  gopro_angle2.mp4

Video A ↔ Video B:
  Offset: 142.5670 seconds
  To convert Video B time to Video A time: time_a = time_b + 142.5670

Save synchronization results to file? (y/n): y
✓ Results saved to sync_results.json

Generate synchronization timeline plot? (y/n): y
✓ Plot saved to sync_timeline.png

Done!
```

## 📚 Tutorials & Guides

**New to multi-modal analysis?** Start here:

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Your first multi-modal analysis (10 min read)
  - Absolute beginner friendly
  - Explains basic concepts (sync, offsets, sampling rates)
  - Copy-paste examples with YOUR data
  - Compares brain activity during movement vs. rest

- **[HOW_TO_USE_SYNC.md](HOW_TO_USE_SYNC.md)** - Using sync results in your analysis
  - Quick answer: "I saw something at 1:45 in video, what was EEG doing?"
  - Real examples with your sync_results.json
  - Formulas cheat sheet (print and keep at desk!)
  - Troubleshooting common mistakes

- **[TUTORIAL_POSE_EEG.md](TUTORIAL_POSE_EEG.md)** - Advanced: Pose + EEG, Behavior + EEG
  - Align 3D pose tracking with EEG
  - Behavioral state labeling → EEG frequency analysis
  - Complete analysis pipeline (copy-paste ready)
  - Multi-channel analysis, spectrograms, time-frequency

- **[CLOCK_DRIFT.md](CLOCK_DRIFT.md)** - Do you need multiple sync points?
  - Single-point sync vs. affine transform
  - When to worry about clock drift
  - Your 2-hour recordings are fine with single sync!

**Recommended path**:
1. GETTING_STARTED.md (if totally new)
2. HOW_TO_USE_SYNC.md (to use your sync_results.json)
3. TUTORIAL_POSE_EEG.md (for pose/behavior analysis)

## Analysis & Visualization

### Load Sync Results

```python
import json

with open('sync_results.json', 'r') as f:
    sync = json.load(f)

eeg_to_video_offset = sync['eeg_to_video_a']['offset']
video_b_to_a_offset = sync['video_a_to_video_b']['offset']
```

### Extract EEG for Video Segment

```python
from sync_eeg_vid import extract_eeg_segment

# Extract EEG data for video segment from 2:00 to 2:30
eeg_data = extract_eeg_segment(
    eeg_filepath='TEp_OpenBCI-RAW-2025-11-14_10-40-38.txt',
    video_start=120.0,  # 2:00 in video
    video_end=150.0,    # 2:30 in video
    eeg_video_offset=eeg_to_video_offset
)

# eeg_data now has columns:
# - EXG Channel 0-7 (EEG channels)
# - Time_EEG (time in EEG recording)
# - Time_Video (time in video)
print(eeg_data[['Time_Video', 'EXG Channel 0']].head())
```

### Plot Synchronization Timeline

```python
from sync_eeg_vid import plot_sync_timeline
import matplotlib.pyplot as plt

fig = plot_sync_timeline(
    sync['eeg_to_video_a'],
    sync['video_a_to_video_b']
)
plt.savefig('sync_timeline.png', dpi=150)
plt.show()
```

### Plot EEG Data Around Sync Point

```python
from sync_eeg_vid import plot_eeg_data
import matplotlib.pyplot as plt

# Plot 10 seconds of EEG data around the sync point
sync_time = sync['eeg_to_video_a']['eeg_sync_time']

fig = plot_eeg_data(
    'TEp_OpenBCI-RAW-2025-11-14_10-40-38.txt',
    start_time=sync_time - 2,  # 2 seconds before
    duration=10,               # 10 seconds total
    channels=[0, 1, 2, 3],     # First 4 channels
    sync_time=sync_time,       # Mark sync point
    show_ir=True               # Show IR blaster channel
)
plt.savefig('eeg_sync_window.png', dpi=150)
plt.show()
```

## File Formats

### Input Files

**EEG Data Files:**

Two formats supported:

1. **CSV files (RECOMMENDED)** - `*fixed_irBlaster.csv`
   - Pre-cleaned IR blaster data (noise removed)
   - Faster to parse
   - More reliable sync detection
   - Format:
     ```
     Time (sec),Value
     0.0,257.0
     0.004,257.0
     ...
     81.344,1      # IR pulse (consecutive 1s)
     81.348,1
     ```

2. **TXT files (fallback)** - `*OpenBCI-RAW-*.txt`
   - Raw OpenBCI format with header
   - Contains 8 EEG channels + IR blaster on Analog Channel 0
   - Sample rate: 250 Hz (typical)
   - IR blaster values: 257 (off), varies when on (may have noise)
   - Script automatically uses matching CSV if available

**Video Files:**
- Any format supported by OpenCV (mp4, mov, avi, etc.)
- GoPro files work great
- Tested with 2+ hour recordings

### Output Files

**sync_results.json**
```json
{
  "eeg_a_to_eeg_b": {
    "eeg_file_a": "TEp_OpenBCI-RAW-2025-11-14_10-40-38.txt",
    "eeg_file_b": "TEc_OpenBCI-RAW-2025-11-14_10-40-39.txt",
    "eeg_sync_time_a": 81.348,
    "eeg_sync_time_b": 1.332,
    "offset": 80.016,
    "note": "To convert EEG B time to EEG A time: time_a = time_b + 80.016"
  },
  "eeg_to_video_a": {
    "eeg_file": "TEp_OpenBCI-RAW-2025-11-14_10-40-38.txt",
    "video_file": "gopro_main.mp4",
    "eeg_sync_time": 81.348,
    "video_sync_time": 83.123,
    "video_frame": 2493,
    "offset": 1.775,
    "note": "To convert EEG time to video time: video_time = eeg_time + 1.775"
  },
  "video_a_to_video_b": {
    "video_a": "gopro_main.mp4",
    "video_b": "gopro_angle2.mp4",
    "sync_time_a": 165.432,
    "sync_time_b": 22.865,
    "offset": 142.567,
    "note": "To convert Video B time to Video A time: time_a = time_b + 142.567"
  }
}
```

## Performance Tips

### For Very Long Videos (2+ hours)

The tool is already optimized, but here are some tips:

1. **Provide approximate timestamps**: Jumping directly to ~1:30 is instant
   ```
   Approximate time of red light: 1:30
   ```

2. **Use the keyboard shortcuts**:
   - `[` / `]` for ±1 minute jumps
   - `,` / `.` for ±10 second jumps

3. **Large videos auto-downscale**: Videos wider than 1920px automatically scale down for faster display (doesn't affect accuracy)

4. **Frame seeking is instant**: OpenCV seeks directly to frames without loading the entire video

### Typical Performance

- **Video loading**: < 1 second
- **Frame seeking**: Instant (even in 2-hour videos)
- **Frame navigation**: ~30-60 fps display rate
- **EEG sync detection**: < 2 seconds for 2-hour recording

## Troubleshooting

### Video won't open
- Check file path is correct
- Ensure video codec is supported (try converting to mp4 with H.264)

### Can't find IR pulse
- Check that IR blaster was connected to Analog Channel 0
- Verify IR blaster fired (check CSV file for non-257 values)
- Ensure .txt file is the RAW OpenBCI file, not preprocessed

### Arrow keys not working
- Make sure video window is in focus (click on it)
- Try using `,` `.` `[` `]` as alternatives

### Slow performance
- Video should auto-downscale if > 1920px wide
- Close other applications
- Try converting video to lower resolution if needed

## Technical Details

### Time Offset Calculations

**EEG to Video:**
```
video_time = eeg_time + offset
eeg_time = video_time - offset
```

**Video B to Video A:**
```
time_a = time_b + offset
time_b = time_a - offset
```

**Chaining offsets (Video B to EEG):**
```
eeg_time = time_b - video_b_to_a_offset - eeg_to_video_a_offset
```

### Precision

- **EEG IR detection**: ~4ms (1 sample at 250 Hz)
- **Video frame marking**: 1 frame (33ms at 30fps, 16ms at 60fps)
- **Overall sync accuracy**: Typically within 1-2 frames

## Citation

If you use this tool in your research, please cite:

```
EEG-Video Synchronization Tool
https://github.com/[your-repo]/asilver_eeg_sync
```

## License

MIT License - feel free to use and modify for your research.

## Support

For issues or questions, please open an issue on GitHub.
