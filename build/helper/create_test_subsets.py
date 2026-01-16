"""Create small train/val subsets for quick testing (v2.0 format).

v2.0 format structure:
- Sharded data with multiple shards
- Each shard: videos/video_{shard_idx}.bin, robot_states/states_{shard_idx}.bin
- Metadata: metadata.json (global) and metadata/metadata_{shard_idx}.json (per-shard)
"""

import json
import math
from pathlib import Path
import numpy as np


# v2.0 format constants
FRAMES_PER_CLIP = 17
NUM_FACTORED_TOKENS = 3
SPATIAL_SIZE = 32
ACTION_DIM = 25


def create_subset_v2(source_dir, target_dir, num_sequences=100, max_shards=2):
    """Create a small subset of the v2.0 dataset for testing.
    
    Args:
        source_dir: Source dataset directory (v2.0 format)
        target_dir: Target directory for subset
        num_sequences: Number of sequences to extract
        max_shards: Maximum number of shards to process
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Load global metadata
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {source_dir}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    num_shards = metadata.get("num_shards", 1)
    num_shards = min(num_shards, max_shards)
    
    print(f"Source: {metadata.get('num_images', 0)} frames across {metadata.get('num_shards', 1)} shards")
    print(f"Processing first {num_shards} shard(s)...")
    
    # Create directory structure
    (target_dir / "videos").mkdir(exist_ok=True)
    (target_dir / "robot_states").mkdir(exist_ok=True)
    (target_dir / "segment_indices").mkdir(exist_ok=True)
    (target_dir / "metadata").mkdir(exist_ok=True)
    
    # Process shards
    total_frames = 0
    total_clips = 0
    sequences_collected = 0
    shard_data = []
    
    for shard_idx in range(num_shards):
        # Try metadata in subdirectory first, then root
        shard_metadata_path = source_dir / "metadata" / f"metadata_{shard_idx}.json"
        if not shard_metadata_path.exists():
            shard_metadata_path = source_dir / f"metadata_{shard_idx}.json"
        
        if not shard_metadata_path.exists():
            print(f"  Shard {shard_idx}: metadata not found, skipping")
            continue
        
        with open(shard_metadata_path) as f:
            shard_metadata = json.load(f)
        
        num_frames = shard_metadata.get("shard_num_frames", 0)
        if num_frames == 0:
            # Try to infer from global metadata if available
            if metadata.get("num_images", 0) > 0 and num_shards == 1:
                num_frames = metadata["num_images"]
            else:
                print(f"  Shard {shard_idx}: empty, skipping")
                continue
        
        num_clips = math.ceil(num_frames / FRAMES_PER_CLIP)
        
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
            print(f"  Shard {shard_idx}: files not found, skipping")
            continue
        
        print(f"  Shard {shard_idx}: {num_frames} frames, {num_clips} clips")
        
        # Load video tokens: [num_clips, 3, 32, 32]
        source_video = np.memmap(
            video_path,
            dtype=np.int32,
            mode="r",
            shape=(num_clips, NUM_FACTORED_TOKENS, SPATIAL_SIZE, SPATIAL_SIZE)
        )
        
        # Load states: [num_frames, 25]
        source_states = np.memmap(
            states_path,
            dtype=np.float32,
            mode="r",
            shape=(num_frames, ACTION_DIM)
        )
        
        # Load segment indices if available
        source_segments = None
        if segment_path.exists():
            source_segments = np.memmap(
                segment_path,
                dtype=np.int32,
                mode="r",
                shape=(num_frames,)
            )
        
        # Calculate how many sequences we can extract from this shard
        # Each sequence needs ~17 frames (1 clip) for past+future
        sequences_from_shard = min(
            num_sequences - sequences_collected,
            num_clips - 1  # Need at least 1 clip for past
        )
        
        if sequences_from_shard <= 0:
            break
        
        # Extract sequences (every 10 clips to avoid overlap)
        stride = max(1, num_clips // sequences_from_shard)
        clip_indices = list(range(0, num_clips - 1, stride))[:sequences_from_shard]
        
        # Calculate frame ranges for these clips
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
        subset_shard_metadata = shard_metadata.copy()
        subset_shard_metadata["shard_num_frames"] = num_frames_subset
        with open(target_dir / "metadata" / f"metadata_{target_shard_idx}.json", "w") as f:
            json.dump(subset_shard_metadata, f, indent=2)
        
        shard_data.append({
            "shard_idx": target_shard_idx,
            "num_frames": num_frames_subset,
            "num_clips": num_clips_subset,
        })
        
        total_frames += num_frames_subset
        total_clips += num_clips_subset
        sequences_collected += num_clips_subset
        
        if sequences_collected >= num_sequences:
            break
    
    # Save global metadata
    subset_metadata = metadata.copy()
    subset_metadata["num_shards"] = len(shard_data)
    subset_metadata["num_images"] = total_frames
    with open(target_dir / "metadata.json", "w") as f:
        json.dump(subset_metadata, f, indent=2)
    
    print(f"\nDone: {target_dir}")
    print(f"  Total: {total_frames} frames, {total_clips} clips across {len(shard_data)} shard(s)")
    print(f"  Sequences: ~{sequences_collected} sequences")


if __name__ == "__main__":
    print("=" * 60)
    print("Creating v2.0 Test Subsets")
    print("=" * 60)
    
    print("\nCreating Training Subset...")
    create_subset_v2(
        "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0",
        "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0_test",
        num_sequences=100,
        max_shards=2
    )
    
    print("\n" + "=" * 60)
    print("Creating Validation Subset...")
    create_subset_v2(
        "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0",
        "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0_test",
        num_sequences=20,
        max_shards=1
    )
    
    print("\n" + "=" * 60)
    print("Test subsets created!")
    print("=" * 60)
