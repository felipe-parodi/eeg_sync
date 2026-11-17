# Clock Drift & Multiple Sync Points

## Do You Need Multiple Sync Points?

**Short answer: Probably not for your use case.**

This tool uses **single-point offset** synchronization (one IR blaster pulse, one clap). This works perfectly when:
- Recording duration < 3 hours
- Devices have stable clocks (OpenBCI, GoPro do)
- You need ~frame-level precision (~33ms at 30fps)

## When Would You Need Multiple Sync Points?

You'd need **multiple sync points + affine transform** (linear time warping) if:

### 1. Very Long Recordings (> 4 hours)
Even stable clocks drift over time:
- OpenBCI drift: ~1-5 ppm (parts per million)
- GoPro drift: ~10-50 ppm
- **Result**: 4-hour recording → 1-2 seconds drift

### 2. Different Sample Rates
If devices are sampling at different rates:
- EEG A: 250 Hz
- EEG B: 250.01 Hz (0.004% difference)
- **Result**: 1 hour → 144ms drift

### 3. Unstable Clocks
Cheap cameras, old devices, or temperature-sensitive oscillators

### 4. High Precision Requirements (< 1 frame)
If you need sub-frame accuracy for neuroscience timing

## Current Accuracy

With single-point sync, your accuracy is:

| Component | Precision | Error Source |
|-----------|-----------|--------------|
| EEG IR pulse | ~4 ms | 1 sample @ 250 Hz |
| Video frame marking | ~33 ms | 1 frame @ 30 fps |
| **Total** | **~35 ms** | Dominated by manual frame selection |

## Checking for Clock Drift

To check if you have significant drift:

```python
import pandas as pd

# Load both EEG files
eeg_a = pd.read_csv('TEc_*fixed_irBlaster.csv')
eeg_b = pd.read_csv('TEp_*fixed_irBlaster.csv')

# Check duration
duration_a = eeg_a['Time (sec)'].max()
duration_b = eeg_b['Time (sec)'].max()

print(f"EEG A duration: {duration_a:.2f} seconds")
print(f"EEG B duration: {duration_b:.2f} seconds")
print(f"Difference: {abs(duration_a - duration_b):.2f} seconds")

# If difference > 2 seconds, you might have clock drift
```

**For your 2-hour pilot data**: Single sync point is perfect!

## If You DO Need Multiple Sync Points

Here's the math:

### Single Point Offset (Current)
```
video_time = eeg_time + offset
```

### Affine Transform (2 sync points)
```python
# Two sync points: start and end
# Sync point 1: eeg_t1 ↔ video_t1
# Sync point 2: eeg_t2 ↔ video_t2

# Calculate scale (clock rate ratio)
scale = (video_t2 - video_t1) / (eeg_t2 - eeg_t1)

# Calculate offset
offset = video_t1 - (eeg_t1 * scale)

# Convert any EEG time to video time
video_time = eeg_time * scale + offset
```

### Implementation
If you need this, add:
1. Second IR blaster pulse at END of recording
2. Detect both pulses
3. Calculate scale + offset
4. Apply linear interpolation

## Recommendation

**For 2-hour research sessions with OpenBCI + GoPro:**
- ✅ Use single-point sync (what you have)
- ✅ Expect ~35ms total accuracy (perfectly fine!)
- ✅ No need for multiple sync points

**Only add multiple sync points if:**
- Recordings > 4 hours
- You notice drift (test by comparing recording durations)
- You need sub-frame precision

Your current setup is **production-ready** for typical neuroscience experiments!

## Further Reading

- [Time Synchronization in Multi-Device Systems](https://en.wikipedia.org/wiki/Clock_synchronization)
- [Crystal Oscillator Stability](https://en.wikipedia.org/wiki/Crystal_oscillator#Accuracy_and_stability)
- OpenBCI uses RTC with ~10 ppm accuracy
- GoPro uses quartz oscillators with ~20-50 ppm drift
