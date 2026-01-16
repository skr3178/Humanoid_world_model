#!/usr/bin/env python3
"""
Inspect the shapes of video and action data in train_v2.0 dataset
"""
import numpy as np
import json
from pathlib import Path

def inspect_dataset_shapes(data_dir):
    """Inspect shapes of video and action data"""
    data_dir = Path(data_dir)
    
    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    print("=" * 80)
    print("Dataset Metadata:")
    print("=" * 80)
    print(f"Number of shards: {metadata['num_shards']}")
    print(f"Total number of images (frames): {metadata['num_images']}")
    print(f"Hz (frame rate): {metadata['hz']}")
    print()
    
    # Inspect first shard
    shard_idx = 0
    video_path = data_dir / "videos" / f"video_{shard_idx}.bin"
    states_path = data_dir / "robot_states" / f"states_{shard_idx}.bin"
    segment_path = data_dir / "segment_indices" / f"segment_idx_{shard_idx}.bin"
    
    print("=" * 80)
    print(f"Inspecting Shard {shard_idx}:")
    print("=" * 80)
    
    # Load video tokens (int32, shape: [num_clips, 3, 32, 32])
    video_data = np.memmap(video_path, dtype=np.int32, mode='r')
    video_size = video_data.size
    # Expected shape: [num_clips, 3, 32, 32] = num_clips * 3 * 32 * 32
    num_clips_shard = video_size // (3 * 32 * 32)
    video_shape = (num_clips_shard, 3, 32, 32)
    video_data_reshaped = video_data.reshape(video_shape)
    
    print(f"\nVideo Tokens:")
    print(f"  File: {video_path}")
    print(f"  Dtype: {video_data.dtype}")
    print(f"  Raw size: {video_size} elements")
    print(f"  Reshaped shape: {video_shape}")
    print(f"  Number of clips in shard: {num_clips_shard}")
    print(f"  Frames per clip: 17 (temporally compressed)")
    print(f"  Total frames in shard: {num_clips_shard * 17}")
    print(f"  Sample values (first clip, first factor, first 5x5):")
    print(f"    {video_data_reshaped[0, 0, :5, :5]}")
    
    # Load robot states (float32, shape: [num_frames, 25])
    states_data = np.memmap(states_path, dtype=np.float32, mode='r')
    states_size = states_data.size
    # Expected shape: [num_frames, 25]
    num_frames_shard = states_size // 25
    states_shape = (num_frames_shard, 25)
    states_data_reshaped = states_data.reshape(states_shape)
    
    print(f"\nRobot States (Actions):")
    print(f"  File: {states_path}")
    print(f"  Dtype: {states_data.dtype}")
    print(f"  Raw size: {states_size} elements")
    print(f"  Reshaped shape: {states_shape}")
    print(f"  Number of frames in shard: {num_frames_shard}")
    print(f"  Action dimension: 25")
    print(f"  Sample values (first frame):")
    print(f"    {states_data_reshaped[0]}")
    
    # Load segment indices (int32, shape: [num_frames])
    segment_data = np.memmap(segment_path, dtype=np.int32, mode='r')
    segment_size = segment_data.size
    
    print(f"\nSegment Indices:")
    print(f"  File: {segment_path}")
    print(f"  Dtype: {segment_data.dtype}")
    print(f"  Shape: ({segment_size},)")
    print(f"  Number of frames: {segment_size}")
    print(f"  Unique segments in shard: {len(np.unique(segment_data))}")
    print(f"  Sample values (first 20):")
    print(f"    {segment_data[:20]}")
    
    # Relationship between clips and frames
    print(f"\n" + "=" * 80)
    print("Relationship Analysis:")
    print("=" * 80)
    print(f"  Video clips in shard: {num_clips_shard}")
    print(f"  Frames represented by clips: {num_clips_shard * 17}")
    print(f"  Actual frames in states: {num_frames_shard}")
    print(f"  Actual frames in segments: {segment_size}")
    print(f"  Ratio (frames/clips): {num_frames_shard / num_clips_shard:.2f}")
    
    # Check a few more shards to see if they're consistent
    print(f"\n" + "=" * 80)
    print("Checking Additional Shards (1, 2, 3):")
    print("=" * 80)
    for idx in [1, 2, 3]:
        if idx >= metadata['num_shards']:
            break
        video_path_check = data_dir / "videos" / f"video_{idx}.bin"
        states_path_check = data_dir / "robot_states" / f"states_{idx}.bin"
        
        if video_path_check.exists() and states_path_check.exists():
            video_check = np.memmap(video_path_check, dtype=np.int32, mode='r')
            states_check = np.memmap(states_path_check, dtype=np.float32, mode='r')
            
            num_clips_check = video_check.size // (3 * 32 * 32)
            num_frames_check = states_check.size // 25
            
            print(f"  Shard {idx}: {num_clips_check} clips, {num_frames_check} frames")
    
    print(f"\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Video shape: [num_clips, 3, 32, 32] (int32)")
    print(f"  - Each clip represents 17 temporally-compressed frames")
    print(f"  - 3 factorized tokens per spatial position")
    print(f"  - Spatial resolution: 32x32")
    print(f"\nAction/State shape: [num_frames, 25] (float32)")
    print(f"  - 25-dimensional action vector per frame")
    print(f"  - Indices 0-20: Joint positions (21 dims)")
    print(f"  - Index 21: Left hand closure")
    print(f"  - Index 22: Right hand closure")
    print(f"  - Index 23: Linear Velocity")
    print(f"  - Index 24: Angular Velocity")
    print(f"\nSegment indices shape: [num_frames] (int32)")
    print(f"  - One segment ID per frame")
    print("=" * 80)

if __name__ == "__main__":
    data_dir = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
    inspect_dataset_shapes(data_dir)
