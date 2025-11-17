# Getting Started: Your First Multi-Modal Analysis

**For: Complete beginners to EEG and video analysis**
**Reading time**: 10 minutes

---

## What You Have

After running the sync script, you now have:
- 📹 **2 synchronized videos** (GoPro cameras)
- 🧠 **2 synchronized EEG recordings** (brain activity from 2 headbands)
- 🔗 **Sync file** (`sync_results.json`) that connects them all

Think of it like a movie with subtitles in 4 different languages - everything happens at the same time, but uses different "clocks."

---

## The Magic Number: Your Offset

Open `sync_results.json` and look for this number:

```json
"eeg_to_video_a": {
  "offset": 7.954
}
```

**This is your magic number: 7.954 seconds**

It means: When the video shows time "1:00", the EEG was actually at time "0:52" (60 - 7.954 = 52.046).

---

## Three Simple Questions You Can Answer

### Question 1: "What was the brain doing at this video moment?"

**Example**: You saw something interesting at 2:15 in the video.

```python
import json

# Load the sync file
with open('sync_results.json') as f:
    sync = json.load(f)

# Video time (2 minutes 15 seconds = 135 seconds)
video_time = 135.0

# The magic formula
offset = sync['eeg_to_video_a']['offset']
eeg_time = video_time - offset

print(f"Video 2:15 = EEG at {eeg_time:.1f} seconds")
# Output: Video 2:15 = EEG at 127.0 seconds
```

**That's it!** Now you know that EEG time 127.0 seconds matches video time 2:15.

---

### Question 2: "When did this brain event happen in the video?"

**Example**: EEG shows a spike at 100 seconds.

```python
# EEG time
eeg_time = 100.0

# Reverse the formula
video_time = eeg_time + offset

print(f"EEG spike at {eeg_time}s = Video at {video_time:.1f}s")
# Output: EEG spike at 100s = Video at 107.9s (1:48)
```

---

### Question 3: "What's the same moment across ALL 4 streams?"

**Example**: Clap happened at 1:31 in Video A. Where is it in everything else?

```python
with open('sync_results.json') as f:
    sync = json.load(f)

# Clap in Video A
video_a_time = 91.0  # 1:31

# Find in EEG
eeg_offset = sync['eeg_to_video_a']['offset']
eeg_time = video_a_time - eeg_offset

# Find in Video B
video_b_offset = sync['video_a_to_video_b']['offset']
video_b_time = video_a_time - video_b_offset

# Find in EEG B
eeg_b_offset = sync['eeg_a_to_eeg_b']['offset']
eeg_b_time = eeg_time - eeg_b_offset

print(f"""
Clap at Video A 1:31 appears in:
  Video A:  {video_a_time:.1f}s (1:31)
  Video B:  {video_b_time:.1f}s (1:10)
  EEG A:    {eeg_time:.1f}s
  EEG B:    {eeg_b_time:.1f}s
""")
```

---

## Your First Real Analysis

Let's do something cool: **Compare brain activity during movement vs. rest**

### Step 1: Identify time periods

Watch your video and note:
- When was the person **sitting still**? (e.g., 1:00 to 1:30)
- When were they **moving**? (e.g., 2:00 to 2:30)

### Step 2: Convert to EEG times

```python
import json

with open('sync_results.json') as f:
    sync = json.load(f)

offset = sync['eeg_to_video_a']['offset']

# Sitting still: Video 1:00-1:30 (60-90 seconds)
sitting_start_eeg = 60.0 - offset  # 52.046s
sitting_end_eeg = 90.0 - offset    # 82.046s

# Moving: Video 2:00-2:30 (120-150 seconds)
moving_start_eeg = 120.0 - offset  # 112.046s
moving_end_eeg = 150.0 - offset    # 142.046s

print(f"Sitting: EEG {sitting_start_eeg:.1f}-{sitting_end_eeg:.1f}s")
print(f"Moving:  EEG {moving_start_eeg:.1f}-{moving_end_eeg:.1f}s")
```

### Step 3: Load and plot EEG

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load EEG data
eeg_data = pd.read_csv('TEc_OpenBCI-RAW-2025-11-14_10-40-39.txt',
                       skiprows=5, comment='%')

# EEG is sampled at 250 Hz (250 samples per second)
sample_rate = 250

# Extract the two time periods
sitting_start_sample = int(sitting_start_eeg * sample_rate)
sitting_end_sample = int(sitting_end_eeg * sample_rate)
sitting_eeg = eeg_data.iloc[sitting_start_sample:sitting_end_sample]

moving_start_sample = int(moving_start_eeg * sample_rate)
moving_end_sample = int(moving_end_eeg * sample_rate)
moving_eeg = eeg_data.iloc[moving_start_sample:moving_end_sample]

# Plot one channel (channel 0)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# Sitting
time_sitting = np.arange(len(sitting_eeg)) / sample_rate
ax1.plot(time_sitting, sitting_eeg['EXG Channel 0'])
ax1.set_title('EEG During Sitting Still')
ax1.set_ylabel('Amplitude (µV)')

# Moving
time_moving = np.arange(len(moving_eeg)) / sample_rate
ax2.plot(time_moving, moving_eeg['EXG Channel 0'], color='orange')
ax2.set_title('EEG During Movement')
ax2.set_ylabel('Amplitude (µV)')
ax2.set_xlabel('Time (seconds)')

plt.tight_layout()
plt.savefig('my_first_eeg_comparison.png')
plt.show()

print("✓ Saved plot to my_first_eeg_comparison.png")
```

### Step 4: Calculate brain wave power

```python
from scipy import signal

def calculate_alpha_power(eeg_segment):
    """
    Alpha waves = 8-13 Hz
    Associated with relaxation, eyes closed
    """
    # Get the EEG signal
    eeg_signal = eeg_segment['EXG Channel 0'].values

    # Calculate frequency content
    freqs, power = signal.welch(eeg_signal, fs=250, nperseg=512)

    # Find alpha band (8-13 Hz)
    alpha_range = (freqs >= 8) & (freqs <= 13)
    alpha_power = np.mean(power[alpha_range])

    return alpha_power

# Compare alpha power
sitting_alpha = calculate_alpha_power(sitting_eeg)
moving_alpha = calculate_alpha_power(moving_eeg)

print(f"""
Alpha Power (8-13 Hz):
  During sitting: {sitting_alpha:.2f}
  During moving:  {moving_alpha:.2f}
  Difference:     {sitting_alpha - moving_alpha:.2f}
""")

# Typically, sitting still has MORE alpha than moving
if sitting_alpha > moving_alpha:
    print("✓ Expected pattern: More alpha during rest!")
else:
    print("⚠ Unexpected: More alpha during movement")
```

---

## Common Mistakes (and How to Avoid Them)

### Mistake 1: Wrong direction

```python
# ❌ WRONG:
eeg_time = video_time + offset

# ✅ RIGHT:
eeg_time = video_time - offset
```

**Remember**: Subtract to go from video → EEG

### Mistake 2: Forgetting sample rate

```python
# ❌ WRONG: Time 100 seconds = row 100
eeg_row = eeg_data.iloc[100]

# ✅ RIGHT: Time 100 seconds = row 100 * 250
eeg_sample = int(100 * 250)  # = 25000
eeg_row = eeg_data.iloc[eeg_sample]
```

**Remember**: EEG is 250 samples/second, so multiply by 250!

### Mistake 3: Column names

```python
# ❌ WRONG:
eeg_data['Channel 0']  # Column doesn't exist

# ✅ RIGHT:
eeg_data['EXG Channel 0']  # Full column name
```

**Tip**: Print `eeg_data.columns` to see all column names

---

## Key Concepts Explained Simply

### What is "synchronization"?

Imagine you're watching a movie on your laptop while your friend watches on their phone, but they started 10 seconds later. The movie is the same, but if you say "that funny moment at 5:00," they need to look at 5:10 on their screen.

Synchronization just means knowing that "10 second difference" so you can talk about the same moment.

### What is EEG sampling rate?

Your video is 30 frames per second (30 "pictures" per second).

Your EEG is 250 samples per second (250 "measurements" per second).

**That means**: EEG has 8x more data points than video per second!

### What are frequency bands?

EEG oscillates (goes up and down). How fast it oscillates tells us about brain states:
- **Slow** oscillations (4-8 Hz = theta): Drowsy, meditation
- **Medium** oscillations (8-13 Hz = alpha): Relaxed, eyes closed
- **Fast** oscillations (13-30 Hz = beta): Thinking, focusing

---

## Next Steps

1. **Try the examples above** with your own data
2. **Identify interesting video moments** and extract EEG
3. **Read TUTORIAL_POSE_EEG.md** for advanced analyses
4. **Read HOW_TO_USE_SYNC.md** for more formulas

---

## Cheat Sheet

**Print this and keep it at your desk!**

```
===========================================
ESSENTIAL FORMULAS
===========================================

Video → EEG:
  eeg_time = video_time - 7.954

EEG → Video:
  video_time = eeg_time + 7.954

Time → Sample number:
  sample = int(time_in_seconds * 250)

Sample number → Time:
  time = sample_number / 250

===========================================
FREQUENCY BANDS
===========================================

Delta:   0.5-4 Hz   (deep sleep)
Theta:   4-8 Hz     (meditation)
Alpha:   8-13 Hz    (relaxed)
Beta:    13-30 Hz   (focused)
Gamma:   30-50 Hz   (cognition)

===========================================
QUICK TIPS
===========================================

✓ Always subtract offset going video→EEG
✓ EEG = 250 samples/second
✓ Video = 30 frames/second
✓ Use 'EXG Channel 0' not 'Channel 0'
✓ Print your data to check column names
===========================================
```

**You're ready! Start with the "First Real Analysis" section above and you'll be analyzing multi-modal data in no time!**
