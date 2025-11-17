# Tutorial: Relating 3D Pose & Behavior to EEG

**For: Non-technical researchers new to multi-modal analysis**
**Use Cases**: Pose estimation + EEG, Behavioral states + EEG frequencies

---

## The Problem We're Solving

You have:
- **Video** with 3D pose tracking (e.g., DeepLabCut, SLEAP, OpenPose)
- **EEG** data from two headbands
- **Question**: How do body movements or behaviors relate to brain activity?

**Examples**:
- Does reaching for an object change alpha waves (8-13 Hz)?
- When the subject is standing vs. sitting, what happens to theta power (4-8 Hz)?
- Does hand velocity correlate with motor cortex activity?

---

## Part 1: Understanding Your Data Files

### 3D Pose Data (typical format)

From DeepLabCut or similar tools:

```csv
frame,timestamp,nose_x,nose_y,nose_z,left_hand_x,left_hand_y,left_hand_z,...
0,0.0000,320.5,240.2,0.0,150.3,180.7,0.0,...
1,0.0333,321.2,240.8,0.0,151.1,181.2,0.0,...
2,0.0667,322.0,241.5,0.0,152.3,182.0,0.0,...
```

**Key columns**:
- `frame`: Video frame number (0, 1, 2, ...)
- `timestamp`: Time in VIDEO (seconds, at 30 fps = 0.033s per frame)
- Body part coordinates: x, y, z positions

### Behavioral State Data (typical format)

From manual annotation or automated classification:

```csv
start_frame,end_frame,start_time,end_time,behavior
45,120,1.5,4.0,sitting
121,240,4.033,8.0,standing
241,360,8.033,12.0,reaching
```

**Key columns**:
- `start_time`, `end_time`: Time in VIDEO (seconds)
- `behavior`: State label (sitting, standing, reaching, etc.)

### EEG Data (from your sync)

You have **two** EEG files already synced:
- `TEc_OpenBCI-RAW-2025-11-14_10-40-39.txt` (EEG A)
- `TEp_OpenBCI-RAW-2025-11-14_10-40-38.txt` (EEG B)

Each sampled at **250 Hz** with **8 channels** per headband.

---

## Part 2: Step-by-Step Walkthrough

### Example 1: Align Pose Data with EEG

**Goal**: For each pose frame, get corresponding EEG data

**Step 1**: Load your pose data

```python
import pandas as pd
import numpy as np
import json

# Load pose tracking results
pose_data = pd.read_csv('pose_tracking_results.csv')

print(f"Loaded {len(pose_data)} pose frames")
print(f"Columns: {list(pose_data.columns)}")
```

**Step 2**: Load sync offsets

```python
# Load sync results
with open('sync_results.json') as f:
    sync = json.load(f)

# Get the offset from Video A to EEG A
video_to_eeg_offset = sync['eeg_to_video_a']['offset']  # 7.954 seconds

print(f"Video to EEG offset: {video_to_eeg_offset:.3f}s")
```

**Step 3**: Add EEG timestamp column to pose data

```python
# Convert video timestamps to EEG timestamps
# Formula: eeg_time = video_time - offset
pose_data['eeg_timestamp'] = pose_data['timestamp'] - video_to_eeg_offset

print("Added EEG timestamps to pose data:")
print(pose_data[['frame', 'timestamp', 'eeg_timestamp']].head())
```

Output:
```
   frame  timestamp  eeg_timestamp
0      0      0.000        -7.954
1      1      0.033        -7.921
2      2      0.067        -7.887
```

**Step 4**: Load EEG data

```python
# Load EEG data
eeg_data = pd.read_csv('TEc_OpenBCI-RAW-2025-11-14_10-40-39.txt',
                       skiprows=5,  # Skip header
                       comment='%')

# Add time column (250 Hz sampling rate)
eeg_data['time'] = np.arange(len(eeg_data)) / 250.0

print(f"Loaded {len(eeg_data)} EEG samples ({len(eeg_data)/250:.1f} seconds)")
```

**Step 5**: Match each pose frame to nearest EEG sample

```python
def find_nearest_eeg_sample(eeg_time, eeg_data):
    """Find EEG sample closest to given time"""
    idx = (np.abs(eeg_data['time'] - eeg_time)).argmin()
    return idx

# For each pose frame, find corresponding EEG
pose_data['eeg_sample_idx'] = pose_data['eeg_timestamp'].apply(
    lambda t: find_nearest_eeg_sample(t, eeg_data)
)

# Extract EEG channels at each pose timepoint
for ch in range(8):
    col_name = f'EXG Channel {ch}'  # EEG column name
    pose_data[f'eeg_ch{ch}'] = pose_data['eeg_sample_idx'].apply(
        lambda idx: eeg_data.iloc[idx][col_name] if 0 <= idx < len(eeg_data) else np.nan
    )

print("Merged pose + EEG:")
print(pose_data[['frame', 'nose_x', 'eeg_ch0', 'eeg_ch1']].head())
```

**Step 6**: Calculate velocities and correlations

```python
# Calculate hand velocity from pose
pose_data['hand_velocity'] = np.sqrt(
    pose_data['left_hand_x'].diff()**2 +
    pose_data['left_hand_y'].diff()**2
) / pose_data['timestamp'].diff()

# Correlate hand velocity with EEG channel 3 (e.g., motor cortex)
correlation = pose_data['hand_velocity'].corr(pose_data['eeg_ch3'])
print(f"Hand velocity ↔ EEG Ch3 correlation: {correlation:.3f}")

# Plot
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.plot(pose_data['timestamp'], pose_data['hand_velocity'])
ax1.set_ylabel('Hand Velocity (pixels/s)')
ax1.set_title('Movement & EEG Over Time')

ax2.plot(pose_data['timestamp'], pose_data['eeg_ch3'])
ax2.set_ylabel('EEG Ch3 (µV)')
ax2.set_xlabel('Video Time (seconds)')

plt.tight_layout()
plt.savefig('pose_eeg_correlation.png', dpi=150)
plt.show()
```

---

### Example 2: Behavioral States → EEG Frequency Bands

**Goal**: Compare EEG frequencies during different behaviors (sitting vs. reaching)

**Step 1**: Load behavioral annotations

```python
# Load behavioral state labels
behavior_data = pd.read_csv('behavioral_states.csv')

print(behavior_data)
#    start_time  end_time   behavior
# 0        1.5       4.0    sitting
# 1      4.033       8.0   standing
# 2      8.033      12.0   reaching
```

**Step 2**: Convert video times to EEG times

```python
# Apply sync offset
behavior_data['eeg_start'] = behavior_data['start_time'] - video_to_eeg_offset
behavior_data['eeg_end'] = behavior_data['end_time'] - video_to_eeg_offset

print(behavior_data[['behavior', 'eeg_start', 'eeg_end']])
```

**Step 3**: Extract EEG for each behavioral state

```python
from scipy import signal

def extract_eeg_for_behavior(behavior_row, eeg_data, sample_rate=250):
    """Extract EEG segment for a behavioral epoch"""
    start_sample = int(behavior_row['eeg_start'] * sample_rate)
    end_sample = int(behavior_row['eeg_end'] * sample_rate)

    # Get EEG segment
    segment = eeg_data.iloc[start_sample:end_sample]
    return segment

# Example: Get EEG during "sitting"
sitting_epochs = behavior_data[behavior_data['behavior'] == 'sitting']
sitting_eeg = extract_eeg_for_behavior(sitting_epochs.iloc[0], eeg_data)

print(f"Extracted {len(sitting_eeg)} samples during sitting")
```

**Step 4**: Calculate EEG frequency power (spectral analysis)

```python
def calculate_frequency_bands(eeg_segment, channel=0, sample_rate=250):
    """
    Calculate power in different frequency bands

    Bands:
    - Delta: 0.5-4 Hz (deep sleep)
    - Theta: 4-8 Hz (drowsiness, meditation)
    - Alpha: 8-13 Hz (relaxed, eyes closed)
    - Beta: 13-30 Hz (active thinking, focus)
    - Gamma: 30-50 Hz (high cognitive function)
    """
    # Get EEG channel data
    eeg_signal = eeg_segment[f'EXG Channel {channel}'].values

    # Calculate power spectral density
    freqs, psd = signal.welch(eeg_signal, fs=sample_rate, nperseg=512)

    # Calculate power in each band
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    band_power = {}
    for band_name, (low_freq, high_freq) in bands.items():
        # Find frequencies in this band
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        # Calculate mean power in band
        band_power[band_name] = np.mean(psd[idx_band])

    return band_power, freqs, psd

# Calculate for sitting behavior
sitting_power, freqs, psd = calculate_frequency_bands(sitting_eeg, channel=0)
print("EEG power during SITTING:")
for band, power in sitting_power.items():
    print(f"  {band:8s}: {power:.2e}")
```

Output:
```
EEG power during SITTING:
  delta   : 2.34e+02
  theta   : 1.56e+02
  alpha   : 8.92e+01
  beta    : 3.21e+01
  gamma   : 1.12e+01
```

**Step 5**: Compare across behaviors

```python
# Calculate for each behavior
results = []
for _, row in behavior_data.iterrows():
    behavior = row['behavior']
    eeg_segment = extract_eeg_for_behavior(row, eeg_data)

    # Calculate frequency power for channel 0
    band_power, _, _ = calculate_frequency_bands(eeg_segment, channel=0)
    band_power['behavior'] = behavior
    results.append(band_power)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print("\nFrequency power by behavior:")
print(results_df)
```

Output:
```
   behavior        delta       theta       alpha        beta       gamma
0   sitting  2.34e+02    1.56e+02    8.92e+01    3.21e+01   1.12e+01
1  standing  1.89e+02    1.23e+02    7.45e+01    4.56e+01   1.89e+01
2  reaching  1.45e+02    9.87e+01    6.12e+01    6.78e+01   3.45e+01
```

**Step 6**: Visualize differences

```python
import matplotlib.pyplot as plt

# Plot bar chart comparing frequency bands
fig, ax = plt.subplots(figsize=(10, 6))

behaviors = results_df['behavior']
x = np.arange(len(behaviors))
width = 0.15

bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (band, color) in enumerate(zip(bands, colors)):
    values = results_df[band]
    ax.bar(x + i*width, values, width, label=band.capitalize(), color=color)

ax.set_xlabel('Behavior')
ax.set_ylabel('Power (µV²/Hz)')
ax.set_title('EEG Frequency Bands Across Behaviors')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(behaviors)
ax.legend()
ax.set_yscale('log')  # Log scale for better visualization

plt.tight_layout()
plt.savefig('behavior_frequency_comparison.png', dpi=150)
plt.show()
```

---

## Part 3: Advanced Analysis

### Multi-Channel Analysis (Both EEG Headbands)

```python
# You have TWO EEG files - analyze both!

# Load EEG B
eeg_b_data = pd.read_csv('TEp_OpenBCI-RAW-2025-11-14_10-40-38.txt',
                         skiprows=5, comment='%')
eeg_b_data['time'] = np.arange(len(eeg_b_data)) / 250.0

# Apply EEG A ↔ EEG B offset
eeg_b_to_a_offset = sync['eeg_a_to_eeg_b']['offset']  # -0.776
eeg_b_data['time_aligned'] = eeg_b_data['time'] - eeg_b_to_a_offset

# Now both EEGs are on same timeline - compare them!
def compare_eeg_headbands(behavior_row, eeg_a, eeg_b, channel=0):
    """Compare same channel from both headbands during behavior"""
    eeg_a_seg = extract_eeg_for_behavior(behavior_row, eeg_a)
    eeg_b_seg = extract_eeg_for_behavior(behavior_row, eeg_b)

    power_a, _, _ = calculate_frequency_bands(eeg_a_seg, channel)
    power_b, _, _ = calculate_frequency_bands(eeg_b_seg, channel)

    return power_a, power_b

# Compare during reaching
reaching_epoch = behavior_data[behavior_data['behavior'] == 'reaching'].iloc[0]
power_headband_a, power_headband_b = compare_eeg_headbands(
    reaching_epoch, eeg_data, eeg_b_data, channel=0
)

print("Alpha power during REACHING:")
print(f"  Headband A (TEc): {power_headband_a['alpha']:.2f}")
print(f"  Headband B (TEp): {power_headband_b['alpha']:.2f}")
```

### Time-Frequency Analysis (Spectrogram)

**See how frequencies change over time during a behavior**

```python
from scipy import signal

def plot_spectrogram_for_behavior(behavior_row, eeg_data, channel=0):
    """Plot time-frequency spectrogram for a behavioral epoch"""
    eeg_segment = extract_eeg_for_behavior(behavior_row, eeg_data)
    eeg_signal = eeg_segment[f'EXG Channel {channel}'].values

    # Compute spectrogram
    freqs, times, Sxx = signal.spectrogram(
        eeg_signal,
        fs=250,
        nperseg=256,
        noverlap=200
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, freqs, 10 * np.log10(Sxx),
                   shading='gouraud', cmap='viridis')
    plt.ylim(0, 50)  # Focus on 0-50 Hz
    plt.colorbar(label='Power (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.title(f"Spectrogram during {behavior_row['behavior']}")

    # Mark frequency bands
    plt.axhline(8, color='white', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(13, color='white', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(times[-1]*0.95, 10.5, 'Alpha', color='white', ha='right')

    plt.tight_layout()
    plt.savefig(f"spectrogram_{behavior_row['behavior']}.png", dpi=150)
    plt.show()

# Generate spectrogram for reaching behavior
reaching = behavior_data[behavior_data['behavior'] == 'reaching'].iloc[0]
plot_spectrogram_for_behavior(reaching, eeg_data, channel=0)
```

---

## Part 4: Complete Analysis Pipeline

Here's a **copy-paste ready** script for your analysis:

```python
"""
Complete Pipeline: Pose/Behavior → EEG Analysis
"""

import pandas as pd
import numpy as np
import json
from scipy import signal
import matplotlib.pyplot as plt

# ============== CONFIGURATION ==============
POSE_FILE = 'pose_tracking_results.csv'
BEHAVIOR_FILE = 'behavioral_states.csv'
EEG_A_FILE = 'TEc_OpenBCI-RAW-2025-11-14_10-40-39.txt'
EEG_B_FILE = 'TEp_OpenBCI-RAW-2025-11-14_10-40-38.txt'
SYNC_FILE = 'sync_results.json'
SAMPLE_RATE = 250  # Hz

# ============== LOAD DATA ==============
print("Loading data...")

# Load sync offsets
with open(SYNC_FILE) as f:
    sync = json.load(f)
video_to_eeg_offset = sync['eeg_to_video_a']['offset']

# Load pose data
pose_data = pd.read_csv(POSE_FILE)
pose_data['eeg_timestamp'] = pose_data['timestamp'] - video_to_eeg_offset

# Load behavioral states
behavior_data = pd.read_csv(BEHAVIOR_FILE)
behavior_data['eeg_start'] = behavior_data['start_time'] - video_to_eeg_offset
behavior_data['eeg_end'] = behavior_data['end_time'] - video_to_eeg_offset

# Load EEG
eeg_data = pd.read_csv(EEG_A_FILE, skiprows=5, comment='%')
eeg_data['time'] = np.arange(len(eeg_data)) / SAMPLE_RATE

print(f"✓ Loaded {len(pose_data)} pose frames")
print(f"✓ Loaded {len(behavior_data)} behavioral epochs")
print(f"✓ Loaded {len(eeg_data)} EEG samples ({len(eeg_data)/SAMPLE_RATE:.1f}s)")

# ============== ANALYSIS FUNCTIONS ==============

def extract_eeg_segment(start_time, end_time, eeg_df, sr=SAMPLE_RATE):
    """Extract EEG between two timestamps"""
    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)
    return eeg_df.iloc[start_idx:end_idx]

def calculate_band_power(eeg_segment, channel=0, sr=SAMPLE_RATE):
    """Calculate power in frequency bands"""
    signal_data = eeg_segment[f'EXG Channel {channel}'].values
    freqs, psd = signal.welch(signal_data, fs=sr, nperseg=512)

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    result = {}
    for band_name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        result[band_name] = np.mean(psd[idx])

    return result

# ============== RUN ANALYSIS ==============
print("\nAnalyzing EEG during behaviors...")

results = []
for _, row in behavior_data.iterrows():
    print(f"  Processing {row['behavior']}...")

    # Extract EEG for this behavior
    eeg_seg = extract_eeg_segment(row['eeg_start'], row['eeg_end'], eeg_data)

    # Calculate frequency power (average across all 8 channels)
    avg_power = {band: 0 for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']}

    for ch in range(8):
        power = calculate_band_power(eeg_seg, channel=ch)
        for band in avg_power:
            avg_power[band] += power[band]

    # Average across channels
    for band in avg_power:
        avg_power[band] /= 8

    avg_power['behavior'] = row['behavior']
    avg_power['duration'] = row['eeg_end'] - row['eeg_start']
    results.append(avg_power)

# ============== RESULTS ==============
results_df = pd.DataFrame(results)

print("\n" + "="*60)
print("RESULTS: EEG Frequency Power by Behavior")
print("="*60)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('eeg_behavior_analysis.csv', index=False)
print("\n✓ Saved to eeg_behavior_analysis.csv")

# ============== VISUALIZATION ==============
print("\nGenerating plots...")

fig, ax = plt.subplots(figsize=(10, 6))
behaviors = results_df['behavior']
x = np.arange(len(behaviors))
width = 0.15

bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (band, color) in enumerate(zip(bands, colors)):
    values = results_df[band]
    ax.bar(x + i*width, values, width, label=band.capitalize(), color=color)

ax.set_xlabel('Behavior', fontsize=12)
ax.set_ylabel('Average Power (µV²/Hz)', fontsize=12)
ax.set_title('EEG Frequency Bands Across Behaviors', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(behaviors)
ax.legend()
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('eeg_behavior_summary.png', dpi=150, bbox_inches='tight')
print("✓ Saved plot to eeg_behavior_summary.png")

plt.show()

print("\nDone!")
```

---

## Part 5: Interpretation Guide

### What do the frequency bands mean?

| Band | Frequency | Associated With |
|------|-----------|----------------|
| **Delta** | 0.5-4 Hz | Deep sleep, unconscious processes |
| **Theta** | 4-8 Hz | Drowsiness, meditation, memory |
| **Alpha** | 8-13 Hz | Relaxed wakefulness, eyes closed |
| **Beta** | 13-30 Hz | Active thinking, focus, anxiety |
| **Gamma** | 30-50 Hz | High-level cognition, perception |

### Example interpretations:

**If during "reaching" you see**:
- ↑ Beta power: Motor planning and execution
- ↓ Alpha power: Active visual processing
- ↑ Gamma power: Sensorimotor integration

**If during "sitting still" you see**:
- ↑ Alpha power: Relaxed, not focused on external stimuli
- ↓ Beta power: Less active cognitive processing

---

## Part 6: Troubleshooting

### "Times don't align!"

**Check**: Did you apply the sync offset correctly?
```python
# CORRECT:
eeg_time = video_time - video_to_eeg_offset

# WRONG:
eeg_time = video_time + video_to_eeg_offset  # ← backwards!
```

### "Power values are huge!"

EEG power is often very large. Use **log scale** or **normalize**:
```python
# Log scale
ax.set_yscale('log')

# Or normalize by total power
total_power = sum(band_power.values())
normalized_power = {k: v/total_power for k, v in band_power.items()}
```

### "Different behaviors show no difference!"

**Possible reasons**:
1. Epochs are too short (need >2 seconds for good frequency resolution)
2. EEG channels don't cover relevant brain areas
3. Artifact contamination (movement, eye blinks)
4. Need to filter/preprocess EEG first

---

## Next Steps

1. **Preprocess EEG**: Bandpass filter (0.5-50 Hz), remove artifacts
2. **Statistical testing**: Use t-tests or ANOVA to compare behaviors
3. **Machine learning**: Train classifier to predict behavior from EEG
4. **Causality**: Use Granger causality to see if movement predicts EEG changes

---

## Quick Reference

```python
# Sync conversion formulas
eeg_time = video_time - 7.954  # Your offset
video_time = eeg_time + 7.954

# Extract EEG for video segment
segment = eeg_data[(eeg_data['time'] >= start) & (eeg_data['time'] <= end)]

# Calculate alpha power
freqs, psd = signal.welch(eeg_signal, fs=250, nperseg=512)
alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
```

**Save this tutorial and run the complete pipeline script to get started!**
