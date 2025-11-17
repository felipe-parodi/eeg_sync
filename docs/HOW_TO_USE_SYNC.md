# How to Use Your Synchronized EEG and Video Data

**For: Anyone analyzing the data**
**Written**: November 2025

This guide shows you **exactly** how to use the sync results to analyze what was happening in the EEG when something occurred in the video.

---

## The Big Picture

After running the sync script, you have **4 synchronized data streams**:
1. **EEG A** (TEc headband)
2. **EEG B** (TEp headband)
3. **Video A** (main GoPro - GOPR0290.MP4)
4. **Video B** (angle 2 GoPro - GX010825.MP4)

They're all synced to a **common timeline**, so you can find what was happening in ALL streams at any moment!

---

## Quick Answer: "I saw something in the video - what was the EEG doing?"

### Example: Video Event at 1:45 (1 minute 45 seconds)

**Step 1**: Convert video time to EEG time using the offset from `sync_results.json`:

```python
import json

# Load sync results
with open('sync_results.json') as f:
    sync = json.load(f)

# Video A time (what you saw in the video)
video_time = 105.0  # 1 minute 45 seconds = 105 seconds

# Get the offset
eeg_to_video_offset = sync['eeg_to_video_a']['offset']  # 7.954 seconds

# Convert to EEG time
eeg_time = video_time - eeg_to_video_offset
print(f"Video time {video_time}s = EEG time {eeg_time}s")
# Output: Video time 105.0s = EEG time 97.046s
```

**Step 2**: Extract that moment from the EEG data:

```python
import pandas as pd

# Load EEG data
eeg_data = pd.read_csv('TEc_OpenBCI-RAW-2025-11-14_10-40-39.txt',
                       skiprows=5,  # Skip header lines
                       comment='%')

# EEG is sampled at 250 Hz (250 samples per second)
sample_rate = 250

# Get EEG data around that time (±2 seconds)
start_sample = int((eeg_time - 2) * sample_rate)
end_sample = int((eeg_time + 2) * sample_rate)

eeg_segment = eeg_data.iloc[start_sample:end_sample]
print(f"Extracted {len(eeg_segment)} EEG samples (4 seconds @ 250 Hz)")
```

**Step 3**: Plot it!

```python
import matplotlib.pyplot as plt
import numpy as np

# Get just the EEG channels (first 8 columns)
eeg_channels = eeg_segment.iloc[:, :8]

# Create time axis
time_axis = np.arange(len(eeg_channels)) / sample_rate + (eeg_time - 2)

# Plot all 8 channels
fig, axes = plt.subplots(8, 1, figsize=(12, 10), sharex=True)
for i in range(8):
    axes[i].plot(time_axis, eeg_channels.iloc[:, i])
    axes[i].set_ylabel(f'Ch {i}')
    axes[i].axvline(eeg_time, color='red', linestyle='--', label='Event')

axes[0].legend()
axes[-1].set_xlabel('Time (seconds)')
plt.suptitle(f'EEG around video time {video_time}s')
plt.tight_layout()
plt.savefig('eeg_at_video_event.png')
plt.show()
```

**That's it!** You now have a plot showing the EEG during your video event.

---

## Real Example Using YOUR Data

From your `sync_results.json`:

```json
{
  "eeg_a_to_eeg_b": {
    "offset": -0.776
  },
  "eeg_to_video_a": {
    "offset": 7.954
  },
  "video_a_to_video_b": {
    "offset": 20.988
  }
}
```

### Scenario: You saw a clap at 1:31.825 in Video A

**What was EEG A doing?**

```python
video_time = 91.825  # From Video A
eeg_time = video_time - 7.954  # Apply offset
print(f"EEG A time: {eeg_time:.2f}s")  # → 83.87 seconds
```

**What was EEG B doing?**

```python
# EEG A to EEG B offset
eeg_b_time = eeg_time - (-0.776)  # Note: subtracting negative = adding
print(f"EEG B time: {eeg_b_time:.2f}s")  # → 84.65 seconds
```

**Was that same clap visible in Video B?**

```python
# Video B time
video_b_time = 91.825 - 20.988  # Apply Video A→B offset
print(f"Video B time: {video_b_time:.2f}s")  # → 70.84 seconds
```

**Check**: This matches clap 1 in Video B (70.837s) ✓

---

## The Formulas (Print This Out!)

### Video A → EEG A
```python
eeg_a_time = video_a_time - 7.954
```

### Video A → EEG B
```python
eeg_b_time = video_a_time - 7.954 - (-0.776)
# Simplified:
eeg_b_time = video_a_time - 7.178
```

### Video B → EEG A
```python
eeg_a_time = (video_b_time + 20.988) - 7.954
# Simplified:
eeg_a_time = video_b_time + 13.034
```

### Video B → EEG B
```python
eeg_b_time = (video_b_time + 20.988) - 7.178
# Simplified:
eeg_b_time = video_b_time + 13.810
```

---

## Common Use Cases

### 1. "Extract EEG for specific video segment"

**Example**: Get EEG for Video A from 2:00 to 2:30

```python
from sync_eeg_vid import extract_eeg_segment
import json

# Load sync data
with open('sync_results.json') as f:
    sync = json.load(f)

# Extract EEG for video segment
eeg_data = extract_eeg_segment(
    eeg_filepath='TEc_OpenBCI-RAW-2025-11-14_10-40-39.txt',
    video_start=120.0,  # 2:00 in seconds
    video_end=150.0,    # 2:30 in seconds
    eeg_video_offset=sync['eeg_to_video_a']['offset']
)

# Data now has columns: Time_EEG, Time_Video, EXG Channel 0-7
print(eeg_data[['Time_Video', 'EXG Channel 0']].head())
```

### 2. "Find when EEG event occurred in video"

**Example**: EEG shows spike at 100.5 seconds - when was that in the video?

```python
eeg_time = 100.5
video_time = eeg_time + 7.954
print(f"EEG spike at {eeg_time}s = Video time {video_time:.2f}s")
# → Video time 108.45s = 1:48.45
```

### 3. "Sync across all 4 streams"

**Example**: Event in Video B at 1:10.837 - find it everywhere

```python
# Video B time
video_b_time = 70.837

# Convert to Video A
video_a_time = video_b_time + 20.988  # → 91.825s

# Convert to EEG A
eeg_a_time = video_a_time - 7.954  # → 83.871s

# Convert to EEG B
eeg_b_time = eeg_a_time + 0.776  # → 84.647s

print(f"""
Event timeline across all streams:
  Video B:  {video_b_time:.2f}s  (1:10.84)
  Video A:  {video_a_time:.2f}s  (1:31.82)
  EEG A:    {eeg_a_time:.2f}s
  EEG B:    {eeg_b_time:.2f}s
""")
```

---

## Troubleshooting

### "The times don't line up!"

**Check 1**: Are you using the right offset?
- Video A → EEG: use `eeg_to_video_a` offset
- Video B → Video A: use `video_a_to_video_b` offset

**Check 2**: Are you adding or subtracting correctly?
- To go FROM video TO eeg: **subtract** offset
- To go FROM eeg TO video: **add** offset

**Check 3**: Print intermediate steps:
```python
print(f"Video time: {video_time}")
print(f"Offset: {offset}")
print(f"EEG time: {video_time - offset}")
```

### "EEG data has no column 'EXG Channel 0'"

The raw TXT file has different column names. Use the helper function:

```python
from sync_eeg_vid import extract_eeg_segment

# This handles column names automatically
eeg_data = extract_eeg_segment(...)
```

---

## Accuracy

Your sync is **very precise**:
- **EEG sync**: ±4ms (1 EEG sample @ 250 Hz)
- **Video sync**: ±33ms (1 video frame @ 30 fps)
- **Clap sync**: ±17ms (standard deviation from 2 claps)

**This means**: You can trust timestamps within about 50ms (0.05 seconds).

---

## Next Steps

1. **Annotate events**: Mark interesting moments in the video
2. **Extract EEG segments**: Use the formulas above
3. **Analyze patterns**: Look for EEG changes during events
4. **Visualize**: Plot EEG overlaid with video timestamps

**Questions?** Check the main README.md or the sync script comments.

---

## Quick Reference Card

```
OFFSETS (from your data):
EEG A ↔ EEG B:    -0.776s
EEG → Video A:    +7.954s
Video A → Video B: +20.988s

TO CONVERT:
Video A time → EEG A time: SUBTRACT 7.954
EEG A time → Video A time: ADD 7.954
Video B time → Video A time: ADD 20.988

ACCURACY:
±50ms (about 1-2 video frames)
```

**Print this card and keep it with your analysis notebook!**
