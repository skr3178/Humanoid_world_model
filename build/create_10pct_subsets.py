"""Create 10% subsets of train/val/test v2.0 datasets.

This script creates proportional 10% subsets by sampling frames/clips
across all shards while preserving the shard structure.
"""

import json
import math
import random
from pathlib import Path
import numpy as np


# v2.0 format constants
FRAMES_PER_CLIP = 17
NUM_FACTORED_TOKENS = 3
SPATIAL_SIZE = 32
ACTION_DIM = 25


def infer_metadata_from_files(source_dir):
    """Infer metadata from files when metadata.json doesn't exist (e.g., test set)."""
    source_dir = Path(source_dir)
    
    # Find all video files
    video_files = sorted(source_dir.glob("videos/video_*.bin"))
    if not video_files:
        video_files = sorted(source_dir.glob("video_*.bin"))
    
    if not video_files:
        return None
    
    num_shards = len(video_files)
    total_frames = 0
    
    # Infer structure from first few files to get average
    sample_size = min(10, num_shards)
    frames_per_shard_list = []
    
    states_files = sorted(source_dir.glob("robot_states/states_*.bin"))
    if not states_files:
        states_files = sorted(source_dir.glob("states_*.bin"))
    
    for i in range(sample_size):
        try:
            if states_files and i < len(states_files):
                states = np.memmap(states_files[i], dtype=np.float32, mode="r")
                num_frames = states.size // ACTION_DIM
                frames_per_shard_list.append(num_frames)
        except Exception:
            continue
    
    if frames_per_shard_list:
        avg_frames_per_shard = sum(frames_per_shard_list) // len(frames_per_shard_list)
        total_frames = avg_frames_per_shard * num_shards
    else:
        # Fallback: estimate from video files
        try:
            first_video = np.memmap(video_files[0], dtype=np.int32, mode="r")
            video_size = first_video.size
            num_clips_per_shard = video_size // (NUM_FACTORED_TOKENS * SPATIAL_SIZE * SPATIAL_SIZE)
            avg_frames_per_shard = num_clips_per_shard * FRAMES_PER_CLIP
            total_frames = avg_frames_per_shard * num_shards
        except Exception:
            return None
    
    return {
        "num_shards": num_shards,
        "num_images": total_frames,
        "hz": 30,  # Default
    }


def create_10pct_subset(source_dir, target_dir, subset_fraction=0.1, seed=42):
    """Create a 10% subset of the v2.0 dataset.
    
    Args:
        source_dir: Source dataset directory (v2.0 format)
        target_dir: Target directory for subset
        subset_fraction: Fraction of data to extract (default: 0.1 for 10%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Load global metadata (or infer from files)
    metadata_path = source_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        print(f"  metadata.json not found, inferring from files...")
        metadata = infer_metadata_from_files(source_dir)
        if metadata is None:
            raise FileNotFoundError(f"Could not infer metadata from files in {source_dir}")
    
    num_shards = metadata.get("num_shards", 1)
    total_frames = metadata.get("num_images", 0)
    
    print(f"Source: {total_frames} frames across {num_shards} shard(s)")
    print(f"Target: {int(total_frames * subset_fraction)} frames ({subset_fraction*100:.1f}%)")
    
    # Create directory structure
    (target_dir / "videos").mkdir(exist_ok=True)
    (target_dir / "robot_states").mkdir(exist_ok=True)
    (target_dir / "segment_indices").mkdir(exist_ok=True)
    (target_dir / "metadata").mkdir(exist_ok=True)
    
    # Process shards
    total_frames_subset = 0
    total_clips_subset = 0
    shard_data = []
    
    # First pass: collect shard info to calculate proportional sampling
    shard_info_list = []
    for shard_idx in range(num_shards):
        # Try metadata in subdirectory first, then root
        shard_metadata_path = source_dir / "metadata" / f"metadata_{shard_idx}.json"
        if not shard_metadata_path.exists():
            shard_metadata_path = source_dir / f"metadata_{shard_idx}.json"
        
        # Try to load shard metadata, or infer from files
        shard_metadata = {}
        if shard_metadata_path.exists():
            with open(shard_metadata_path) as f:
                shard_metadata = json.load(f)
            num_frames = shard_metadata.get("shard_num_frames", 0)
        else:
            # Infer from files by actually loading them
            video_path = source_dir / "videos" / f"video_{shard_idx}.bin"
            if not video_path.exists():
                video_path = source_dir / f"video_{shard_idx}.bin"
            
            states_path = source_dir / "robot_states" / f"states_{shard_idx}.bin"
            if not states_path.exists():
                states_path = source_dir / f"states_{shard_idx}.bin"
            
            if not video_path.exists() or not states_path.exists():
                print(f"  Shard {shard_idx}: files not found, skipping")
                continue
            
            try:
                # Load states to get actual frame count
                states = np.memmap(states_path, dtype=np.float32, mode="r")
                num_frames = states.size // ACTION_DIM
                shard_metadata = {"shard_num_frames": num_frames}
            except Exception as e:
                print(f"  Shard {shard_idx}: error loading files ({e}), skipping")
                continue
        
        if num_frames == 0:
            print(f"  Shard {shard_idx}: empty, skipping")
            continue
        
        num_clips = math.ceil(num_frames / FRAMES_PER_CLIP)
        shard_info_list.append({
            "shard_idx": shard_idx,
            "num_frames": num_frames,
            "num_clips": num_clips,
            "metadata": shard_metadata,
        })
    
    # Calculate target frames per shard (proportional)
    total_source_frames = sum(info["num_frames"] for info in shard_info_list)
    target_total_frames = int(total_source_frames * subset_fraction)
    
    print(f"\nProcessing {len(shard_info_list)} shard(s)...")
    
    # Second pass: extract data proportionally from each shard
    for shard_info in shard_info_list:
        shard_idx = shard_info["shard_idx"]
        num_frames = shard_info["num_frames"]
        num_clips = shard_info["num_clips"]
        
        # Calculate proportional target for this shard
        shard_fraction = num_frames / total_source_frames if total_source_frames > 0 else 0
        target_frames_for_shard = int(target_total_frames * shard_fraction)
        target_clips_for_shard = math.ceil(target_frames_for_shard / FRAMES_PER_CLIP)
        
        # Ensure we don't exceed available clips
        target_clips_for_shard = min(target_clips_for_shard, num_clips)
        # Ensure at least 1 clip if we have any data
        if target_clips_for_shard == 0 and num_clips > 0:
            target_clips_for_shard = 1
        target_frames_for_shard = min(target_clips_for_shard * FRAMES_PER_CLIP, num_frames)
        
        print(f"  Shard {shard_idx}: {num_frames} frames -> {target_frames_for_shard} frames "
              f"({target_frames_for_shard/num_frames*100:.1f}%)")
        
        # Try subdirectories first, then root directory
        video_path = source_dir / "videos" / f"video_{shard_idx}.bin"
        if not video_path.exists():
            video_path = source_dir / f"video_{shard_idx}.bin"
        
        states_path = source_dir / "robot_states" / f"states_{shard_idx}.bin"
        if not states_path.exists():
            states_path = source_dir / f"states_{shard_idx}.bin"
        
        segment_path = source_dir / "segment_indices" / f"segment_idx_{shard_idx}.bin"
        if not segment_path.exists():
            segment_path = source_dir / f"segment_idx_{shard_idx}.bin"
        
        if not video_path.exists() or not states_path.exists():
            print(f"    Files not found, skipping")
            continue
        
        # Verify file sizes match expected shapes
        video_file_size = video_path.stat().st_size
        expected_video_size = num_clips * NUM_FACTORED_TOKENS * SPATIAL_SIZE * SPATIAL_SIZE * 4
        if video_file_size < expected_video_size:
            # Recalculate num_clips from actual file size
            num_clips = video_file_size // (NUM_FACTORED_TOKENS * SPATIAL_SIZE * SPATIAL_SIZE * 4)
            if num_clips == 0:
                print(f"    Shard {shard_idx}: video file too small, skipping")
                continue
            target_clips_for_shard = min(target_clips_for_shard, num_clips)
            if target_clips_for_shard == 0:
                target_clips_for_shard = 1
        
        # Load source data
        try:
            source_video = np.memmap(
                video_path,
                dtype=np.int32,
                mode="r",
                shape=(num_clips, NUM_FACTORED_TOKENS, SPATIAL_SIZE, SPATIAL_SIZE)
            )
        except ValueError as e:
            print(f"    Shard {shard_idx}: error loading video ({e}), skipping")
            continue
        
        source_states = np.memmap(
            states_path,
            dtype=np.float32,
            mode="r",
            shape=(num_frames, ACTION_DIM)
        )
        
        source_segments = None
        if segment_path.exists():
            source_segments = np.memmap(
                segment_path,
                dtype=np.int32,
                mode="r",
                shape=(num_frames,)
            )
        
        # Sample clips uniformly across the shard
        if target_clips_for_shard >= num_clips:
            # Take all clips
            clip_indices = list(range(num_clips))
        else:
            # Sample uniformly
            clip_indices = sorted(random.sample(range(num_clips), target_clips_for_shard))
        
        # Calculate frame indices for sampled clips
        frame_indices = []
        for clip_idx in clip_indices:
            start_frame = clip_idx * FRAMES_PER_CLIP
            end_frame = min(start_frame + FRAMES_PER_CLIP, num_frames)
            frame_indices.extend(range(start_frame, end_frame))
        
        frame_indices = sorted(set(frame_indices))
        clip_indices_subset = sorted(set(clip_indices))
        
        num_frames_subset = len(frame_indices)
        num_clips_subset = len(clip_indices_subset)
        
        print(f"    Extracting {num_clips_subset} clips ({num_frames_subset} frames)")
        
        # Extract video tokens
        subset_video = np.zeros(
            (num_clips_subset, NUM_FACTORED_TOKENS, SPATIAL_SIZE, SPATIAL_SIZE),
            dtype=np.int32
        )
        for i, clip_idx in enumerate(clip_indices_subset):
            subset_video[i] = source_video[clip_idx]
        
        # Extract states
        subset_states = np.zeros((num_frames_subset, ACTION_DIM), dtype=np.float32)
        for i, frame_idx in enumerate(frame_indices):
            subset_states[i] = source_states[frame_idx]
        
        # Extract segment indices if available
        subset_segments = None
        if source_segments is not None:
            subset_segments = np.zeros(num_frames_subset, dtype=np.int32)
            for i, frame_idx in enumerate(frame_indices):
                subset_segments[i] = source_segments[frame_idx]
        
        # Save subset shard
        target_shard_idx = len(shard_data)
        
        # Save video
        target_video_path = target_dir / "videos" / f"video_{target_shard_idx}.bin"
        with open(target_video_path, "wb") as f:
            subset_video.tofile(f)
        
        # Save states
        target_states_path = target_dir / "robot_states" / f"states_{target_shard_idx}.bin"
        with open(target_states_path, "wb") as f:
            subset_states.tofile(f)
        
        # Save segment indices if available
        if subset_segments is not None:
            target_segment_path = target_dir / "segment_indices" / f"segment_idx_{target_shard_idx}.bin"
            with open(target_segment_path, "wb") as f:
                subset_segments.tofile(f)
        
        # Save shard metadata
        subset_shard_metadata = shard_info["metadata"].copy()
        subset_shard_metadata["shard_num_frames"] = num_frames_subset
        with open(target_dir / "metadata" / f"metadata_{target_shard_idx}.json", "w") as f:
            json.dump(subset_shard_metadata, f, indent=2)
        
        shard_data.append({
            "shard_idx": target_shard_idx,
            "num_frames": num_frames_subset,
            "num_clips": num_clips_subset,
        })
        
        total_frames_subset += num_frames_subset
        total_clips_subset += num_clips_subset
    
    # Save global metadata
    subset_metadata = metadata.copy()
    subset_metadata["num_shards"] = len(shard_data)
    subset_metadata["num_images"] = total_frames_subset
    with open(target_dir / "metadata.json", "w") as f:
        json.dump(subset_metadata, f, indent=2)
    
    print(f"\nDone: {target_dir}")
    print(f"  Total: {total_frames_subset} frames, {total_clips_subset} clips across {len(shard_data)} shard(s)")
    print(f"  Original: {total_frames} frames")
    print(f"  Percentage: {total_frames_subset/total_frames*100:.2f}%" if total_frames > 0 else "  Percentage: N/A")


if __name__ == "__main__":
    base_dir = Path("/media/skr/storage/robot_world/humanoid_wm/1xgpt/data")
    
    print("=" * 70)
    print("Creating 10% Subsets of v2.0 Datasets")
    print("=" * 70)
    
    # Train subset
    print("\n" + "=" * 70)
    print("Creating Training Subset (10%)...")
    print("=" * 70)
    create_10pct_subset(
        base_dir / "train_v2.0",
        base_dir / "train_v2.0_10pct",
        subset_fraction=0.1,
        seed=42
    )
    
    # Val subset
    print("\n" + "=" * 70)
    print("Creating Validation Subset (10%)...")
    print("=" * 70)
    create_10pct_subset(
        base_dir / "val_v2.0",
        base_dir / "val_v2.0_10pct",
        subset_fraction=0.1,
        seed=42
    )
    
    # Test subset (if exists)
    test_source = base_dir / "test_v2.0"
    if test_source.exists():
        print("\n" + "=" * 70)
        print("Creating Test Subset (10%)...")
        print("=" * 70)
        try:
            create_10pct_subset(
                test_source,
                base_dir / "test_v2.0_10pct",
                subset_fraction=0.1,
                seed=42
            )
        except Exception as e:
            print(f"  Error creating test subset: {e}")
            print("  Skipping test subset...")
    else:
        print("\n" + "=" * 70)
        print("Test dataset not found, skipping...")
        print("=" * 70)
    
    print("\n" + "=" * 70)
    print("10% subsets created successfully!")
    print("=" * 70)
    print(f"\nOutput directories:")
    print(f"  - Train: {base_dir / 'train_v2.0_10pct'}")
    print(f"  - Val: {base_dir / 'val_v2.0_10pct'}")
    if test_source.exists() and (base_dir / "test_v2.0_10pct" / "metadata.json").exists():
        print(f"  - Test: {base_dir / 'test_v2.0_10pct'}")
