#!/usr/bin/env python3
"""
Inspect a sample from the dataset and show dimensions at every step of the collator.
"""

import sys
from pathlib import Path

# Add build directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from data.dataset import HumanoidWorldModelDataset
from data.collator_with_dimensions import MaskedHWMCollator
from masked_hwm.config import MaskedHWMConfig

def main():
    # Dataset path - try test_v2.0 first as requested
    test_dir = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/test_v2.0"
    train_dir = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
    
    # Check which dataset to use
    from pathlib import Path
    use_test = Path(test_dir).exists() and (Path(test_dir) / "metadata.json").exists()
    
    if use_test:
        data_dir = test_dir
        print("Using test_v2.0 dataset (as requested)")
        # For test_v2.0, use smaller frame counts since shards are small
        # Each clip = 17 frames, so use 1 clip for past and 1 for future
        num_past_frames = 17
        num_future_frames = 17
    else:
        data_dir = train_dir
        print("Using train_v2.0 dataset (test_v2.0 not available or has issues)")
        num_past_frames = None  # Use config default
        num_future_frames = None  # Use config default
    
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    print(f"Data directory: {data_dir}")
    
    # Load config
    config = MaskedHWMConfig()
    
    # Create dataset
    dataset_kwargs = {
        "data_dir": data_dir,
        "filter_interrupts": True,
        "filter_overlaps": False,
        "max_shards": 1,  # Only load first shard for quick inspection
    }
    
    if num_past_frames is not None:
        dataset_kwargs["num_past_frames"] = num_past_frames
        dataset_kwargs["num_future_frames"] = num_future_frames
    else:
        dataset_kwargs["num_past_frames"] = config.num_past_frames
        dataset_kwargs["num_future_frames"] = config.num_future_frames
    
    try:
        dataset = HumanoidWorldModelDataset(**dataset_kwargs)
    except (ValueError, IndexError) as e:
        if use_test:
            print(f"\nWarning: test_v2.0 dataset has insufficient data: {e}")
            print("Falling back to train_v2.0 dataset...")
            data_dir = train_dir
            dataset_kwargs["data_dir"] = data_dir
            dataset_kwargs["num_past_frames"] = config.num_past_frames
            dataset_kwargs["num_future_frames"] = config.num_future_frames
            dataset = HumanoidWorldModelDataset(**dataset_kwargs)
        else:
            raise
    
    print(f"\nDataset loaded: {len(dataset)} samples available")
    
    if len(dataset) == 0 and use_test:
        print("\nWarning: test_v2.0 has no valid samples (shards too small).")
        print("Falling back to train_v2.0 dataset...")
        data_dir = train_dir
        dataset_kwargs["data_dir"] = data_dir
        dataset_kwargs["num_past_frames"] = config.num_past_frames
        dataset_kwargs["num_future_frames"] = config.num_future_frames
        dataset = HumanoidWorldModelDataset(**dataset_kwargs)
        print(f"Dataset loaded: {len(dataset)} samples available from train_v2.0")
    
    # Get a sample
    print("\n" + "="*80)
    print("LOADING SAMPLE FROM DATASET")
    print("="*80)
    sample_idx = 0
    sample = dataset[sample_idx]
    
    print(f"\nSample {sample_idx} from dataset:")
    print(f"  video_past: {sample['video_past'].shape}")
    print(f"  video_future: {sample['video_future'].shape}")
    print(f"  actions_past: {sample['actions_past'].shape}")
    print(f"  actions_future: {sample['actions_future'].shape}")
    
    # Show detailed dimensions
    print(f"\nDetailed dimensions:")
    print(f"  Video past:")
    print(f"    Shape: {sample['video_past'].shape}")
    print(f"    Interpretation: (num_past_clips={sample['video_past'].shape[0]}, num_factors={sample['video_past'].shape[1]}, H={sample['video_past'].shape[2]}, W={sample['video_past'].shape[3]})")
    print(f"    Total tokens: {sample['video_past'].numel()}")
    
    print(f"  Video future:")
    print(f"    Shape: {sample['video_future'].shape}")
    print(f"    Interpretation: (num_future_clips={sample['video_future'].shape[0]}, num_factors={sample['video_future'].shape[1]}, H={sample['video_future'].shape[2]}, W={sample['video_future'].shape[3]})")
    print(f"    Total tokens: {sample['video_future'].numel()}")
    
    print(f"  Actions past:")
    print(f"    Shape: {sample['actions_past'].shape}")
    print(f"    Interpretation: (T_p_frames={sample['actions_past'].shape[0]}, action_dim={sample['actions_past'].shape[1]})")
    
    print(f"  Actions future:")
    print(f"    Shape: {sample['actions_future'].shape}")
    print(f"    Interpretation: (T_f_frames={sample['actions_future'].shape[0]}, action_dim={sample['actions_future'].shape[1]})")
    
    # Create collator with dimension printing
    print("\n" + "="*80)
    print("PROCESSING SAMPLE THROUGH COLLATOR")
    print("="*80)
    collator = MaskedHWMCollator(config)
    
    # Process sample (wrap in list to create batch of size 1)
    batch = collator([sample])
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nOriginal sample dimensions:")
    print(f"  video_past: {sample['video_past'].shape}")
    print(f"  video_future: {sample['video_future'].shape}")
    print(f"  actions_past: {sample['actions_past'].shape}")
    print(f"  actions_future: {sample['actions_future'].shape}")
    
    print(f"\nCollator output dimensions:")
    print(f"  video_past: {batch['video_past'].shape}")
    print(f"  video_future: {batch['video_future'].shape}")
    print(f"  video_future_labels: {batch['video_future_labels'].shape}")
    print(f"  mask: {batch['mask'].shape}")
    print(f"  actions_past: {batch['actions_past'].shape}")
    print(f"  actions_future: {batch['actions_future'].shape}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)

if __name__ == "__main__":
    main()
