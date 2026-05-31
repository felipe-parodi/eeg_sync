"""Sample, deduplicate, and split frames from the compressed session videos.

Produces a stratified, participant-disjoint set of candidate training frames
plus a ``frame_manifest.csv`` describing each kept frame. Frames are sampled
evenly across each clip, near-duplicates are dropped with a difference hash,
and participants are split into train/val/test so no participant appears in
more than one split (which would leak identity into the validation metrics).
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = REPO_ROOT / "compressed_video"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pose_training" / "data"
VALID_ANGLES = {"SHORT", "TALL"}
HASH_SIZE = 8


@dataclass(frozen=True)
class VideoMeta:
    """Participant and camera angle parsed from a clip filename."""

    participant: str
    angle: str


@dataclass
class SamplingConfig:
    """Configuration for the frame-sampling stage."""

    video_dir: Path
    output_dir: Path
    sample_fps: float = 0.2
    max_frames_per_video: int = 150
    margin_frac: float = 0.03
    dedup_threshold: int = 6
    n_test_participants: int = 3
    n_val_participants: int = 2
    jpeg_quality: int = 2
    workers: int = 8
    seed: int = 1234
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    max_videos: Optional[int] = None


def parse_video_name(filename: str) -> VideoMeta:
    """Parse ``<participant>_FULL_<ANGLE>_compressed.mp4`` into its parts.

    Args:
        filename: Clip filename (basename or full path string).

    Returns:
        VideoMeta with the participant id and camera angle.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3 or parts[1].upper() != "FULL":
        raise ValueError(f"Unexpected clip filename: {filename!r}")
    angle = parts[2].upper()
    if angle not in VALID_ANGLES:
        raise ValueError(f"Unknown camera angle in {filename!r}: {angle}")
    return VideoMeta(participant=parts[0], angle=angle)


def evenly_spaced_timestamps(
    duration_s: float, count: int, margin_frac: float = 0.03
) -> List[float]:
    """Return ``count`` timestamps spread evenly across a clip, inside margins.

    Args:
        duration_s: Clip duration in seconds.
        count: Number of timestamps to return.
        margin_frac: Fraction of the duration to skip at the head and tail.

    Returns:
        Ascending list of timestamps in seconds.
    """
    if count <= 0 or duration_s <= 0:
        return []
    if count == 1:
        return [duration_s / 2.0]
    start = duration_s * margin_frac
    end = duration_s * (1.0 - margin_frac)
    return [float(t) for t in np.linspace(start, end, count)]


def dhash(image: np.ndarray, hash_size: int = HASH_SIZE) -> int:
    """Compute a difference hash of an image as a 64-bit integer.

    Args:
        image: BGR or grayscale uint8 image array.
        hash_size: Output hash side length; the hash has hash_size**2 bits.

    Returns:
        Integer hash; similar images produce small Hamming distances.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Return the Hamming distance (number of differing bits) of two hashes."""
    return bin(a ^ b).count("1")


def dedup_indices(hashes: Sequence[int], threshold: int) -> List[int]:
    """Greedily keep frames whose hash is far from every already-kept frame.

    Args:
        hashes: Per-frame difference hashes in frame order.
        threshold: Maximum Hamming distance considered a near-duplicate.

    Returns:
        Sorted indices of frames to keep.
    """
    kept: List[int] = []
    kept_hashes: List[int] = []
    for index, value in enumerate(hashes):
        if any(hamming(value, other) <= threshold for other in kept_hashes):
            continue
        kept.append(index)
        kept_hashes.append(value)
    return kept


def assign_splits(
    participants: Sequence[str], n_test: int, n_val: int, seed: int
) -> Dict[str, str]:
    """Assign each participant to exactly one of train/val/test.

    Args:
        participants: Participant ids (duplicates allowed; deduped internally).
        n_test: Number of participants held out entirely for test.
        n_val: Number of participants used for validation.
        seed: RNG seed for a reproducible shuffle.

    Returns:
        Mapping participant id -> split name.

    Raises:
        ValueError: If there are not enough participants for the requested
            test and validation counts (at least one must remain for train).
    """
    unique = sorted(set(participants))
    if n_test + n_val >= len(unique):
        raise ValueError(
            f"{len(unique)} participants cannot fill n_test={n_test} + "
            f"n_val={n_val} and still leave any for training"
        )
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    splits: Dict[str, str] = {}
    for participant in shuffled[:n_test]:
        splits[participant] = "test"
    for participant in shuffled[n_test : n_test + n_val]:
        splits[participant] = "val"
    for participant in shuffled[n_test + n_val :]:
        splits[participant] = "train"
    return splits


def ffprobe_duration(video: Path, ffprobe_bin: str = "ffprobe") -> float:
    """Return a clip's duration in seconds via ffprobe."""
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(proc.stdout.strip())


def extract_frame(
    video: Path,
    timestamp_s: float,
    out_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    jpeg_quality: int = 2,
) -> bool:
    """Extract a single frame at a timestamp using fast input seeking.

    Args:
        video: Source clip.
        timestamp_s: Seek time in seconds.
        out_path: Destination jpg path.
        ffmpeg_bin: ffmpeg executable.
        jpeg_quality: ffmpeg ``-q:v`` value (lower is higher quality).

    Returns:
        True if the frame file was written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp_s:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            str(jpeg_quality),
            "-y",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return out_path.exists()


def _sample_one_video(
    video: Path, meta: VideoMeta, split: str, config: SamplingConfig
) -> List[Dict[str, object]]:
    """Sample, extract, and dedup frames for one clip. Returns manifest rows."""
    duration = ffprobe_duration(video, config.ffprobe_bin)
    count = min(
        config.max_frames_per_video,
        max(1, int(duration * config.sample_fps)),
    )
    timestamps = evenly_spaced_timestamps(duration, count, config.margin_frac)
    stream_dir = config.output_dir / "frames" / f"{meta.participant}_{meta.angle}"

    extracted: List[Dict[str, object]] = []
    for index, timestamp in enumerate(timestamps):
        out_path = stream_dir / f"{meta.participant}_{meta.angle}_{index:04d}.jpg"
        if not extract_frame(
            video, timestamp, out_path, config.ffmpeg_bin, config.jpeg_quality
        ):
            continue
        image = cv2.imread(str(out_path))
        if image is None:
            out_path.unlink(missing_ok=True)
            continue
        extracted.append(
            {
                "frame_path": out_path,
                "participant": meta.participant,
                "angle": meta.angle,
                "split": split,
                "source_video": video.name,
                "source_timestamp_s": round(float(timestamp), 3),
                "dhash": dhash(image),
            }
        )

    keep = set(
        dedup_indices([row["dhash"] for row in extracted], config.dedup_threshold)
    )
    rows: List[Dict[str, object]] = []
    for index, row in enumerate(extracted):
        if index in keep:
            rows.append(row)
        else:
            Path(row["frame_path"]).unlink(missing_ok=True)
    return rows


def sample_dataset(config: SamplingConfig) -> Path:
    """Run the full sampling stage and write ``frame_manifest.csv``.

    Args:
        config: Sampling configuration.

    Returns:
        Path to the written manifest CSV.

    Raises:
        FileNotFoundError: If no clips are found in the video directory.
    """
    videos = sorted(config.video_dir.glob("*.mp4"))
    if config.max_videos is not None:
        videos = videos[: config.max_videos]
    if not videos:
        raise FileNotFoundError(f"No .mp4 clips found in {config.video_dir}")

    metas = {video: parse_video_name(video.name) for video in videos}
    splits = assign_splits(
        [meta.participant for meta in metas.values()],
        n_test=config.n_test_participants,
        n_val=config.n_val_participants,
        seed=config.seed,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {
            pool.submit(
                _sample_one_video,
                video,
                meta,
                splits[meta.participant],
                config,
            ): video
            for video, meta in metas.items()
        }
        for future in as_completed(futures):
            all_rows.extend(future.result())

    all_rows.sort(key=lambda row: (str(row["source_video"]), str(row["frame_path"])))
    manifest_path = config.output_dir / "frame_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_path",
                "participant",
                "angle",
                "split",
                "source_video",
                "source_timestamp_s",
                "dhash",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow({**row, "frame_path": str(row["frame_path"])})
    return manifest_path


def summarize_manifest(manifest_path: Path) -> str:
    """Return a short per-split / per-angle summary of a manifest."""
    with manifest_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_split: Dict[str, int] = {}
    by_angle: Dict[str, int] = {}
    participants_by_split: Dict[str, set] = {}
    for row in rows:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
        by_angle[row["angle"]] = by_angle.get(row["angle"], 0) + 1
        participants_by_split.setdefault(row["split"], set()).add(row["participant"])
    lines = [f"Total kept frames: {len(rows)}"]
    for split in ("train", "val", "test"):
        n_part = len(participants_by_split.get(split, set()))
        lines.append(
            f"  {split}: {by_split.get(split, 0)} frames, {n_part} participant(s)"
        )
    lines.append(f"  by angle: {by_angle}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample and deduplicate training frames from session clips."
    )
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), type=str)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    parser.add_argument("--sample-fps", default=0.2, type=float)
    parser.add_argument("--max-frames-per-video", default=150, type=int)
    parser.add_argument("--dedup-threshold", default=6, type=int)
    parser.add_argument("--n-test-participants", default=3, type=int)
    parser.add_argument("--n-val-participants", default=2, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--max-videos", default=None, type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = SamplingConfig(
        video_dir=Path(args.video_dir),
        output_dir=Path(args.output_dir),
        sample_fps=args.sample_fps,
        max_frames_per_video=args.max_frames_per_video,
        dedup_threshold=args.dedup_threshold,
        n_test_participants=args.n_test_participants,
        n_val_participants=args.n_val_participants,
        workers=args.workers,
        seed=args.seed,
        max_videos=args.max_videos,
    )
    manifest_path = sample_dataset(config)
    print(f"Wrote manifest: {manifest_path}")
    print(summarize_manifest(manifest_path))


if __name__ == "__main__":
    main()
