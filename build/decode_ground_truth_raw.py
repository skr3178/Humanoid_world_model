#!/usr/bin/env python3
"""Decode ground truth video directly from dataset without any processing."""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add build directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import HumanoidWorldModelDataset
from masked_hwm.config_test import MaskedHWMTestConfig

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("WARNING: imageio not available")


def decode_ground_truth_raw(
    data_dir,
    output_dir,
    tokenizer_path,
    num_samples=3,
    device="cuda"
):
    """Decode ground truth videos directly from dataset without any processing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = MaskedHWMTestConfig()
    
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = HumanoidWorldModelDataset(
        data_dir=data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1
    )
    
    if len(dataset) == 0:
        print(f"ERROR: Dataset is empty")
        return
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Load decoder
    try:
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer
        from cosmos_tokenizer.utils import tensor2numpy
    except ImportError:
        print("ERROR: cosmos_tokenizer not found. Cannot decode video.")
        return
    
    decoder = CausalVideoTokenizer(
        checkpoint_dec=f"{tokenizer_path}/decoder.jit",
        device=device,
        dtype="bfloat16"
    )
    
    print(f"\nDecoding {min(num_samples, len(dataset))} ground truth samples...")
    
    for sample_idx in range(min(num_samples, len(dataset))):
        print(f"\nSample {sample_idx + 1}:")
        
        # Get sample directly from dataset
        sample = dataset[sample_idx]
        video_future = sample["video_future"]  # (num_future_clips, 3, 32, 32)
        
        print(f"  Video future shape: {video_future.shape}")
        print(f"  Video future type: {type(video_future)}")
        
        # Decode each clip
        decoded_frames = []
        num_clips = video_future.shape[0]
        
        for clip_idx in range(num_clips):
            # Get tokens for this clip: (3, 32, 32)
            clip_tokens = video_future[clip_idx]  # (3, 32, 32)
            
            # Convert to tensor and add batch dimension
            if isinstance(clip_tokens, np.ndarray):
                clip_tokens = torch.from_numpy(clip_tokens.copy())
            clip_tokens = clip_tokens.unsqueeze(0).to(device)  # (1, 3, 32, 32)
            
            # Decode: (1, 3, 32, 32) -> (1, 3, 17, 256, 256)
            with torch.no_grad():
                decoded_clip = decoder.decode(clip_tokens).float()
                decoded_frames.append(decoded_clip)
        
        # Concatenate all clips: (1, 3, T*17, 256, 256)
        video = torch.cat(decoded_frames, dim=2)
        
        print(f"  Decoded video shape: {video.shape}")
        print(f"  Decoded video range: [{video.min().item():.3f}, {video.max().item():.3f}]")
        
        # Use official tensor2numpy for conversion - NO channel swapping
        video_np = tensor2numpy(video)  # (1, T, H, W, 3) in [0, 255] uint8
        video_np = video_np[0]  # (T, H, W, 3) - remove batch dim
        
        print(f"  Final numpy shape: {video_np.shape}")
        print(f"  Final numpy range: [{video_np.min()}, {video_np.max()}]")
        print(f"  Final numpy dtype: {video_np.dtype}")
        
        # Save video - NO processing, NO channel swapping
        output_path = output_dir / f"sample_{sample_idx}_ground_truth_raw.mp4"
        
        if HAS_IMAGEIO:
            imageio.mimsave(
                str(output_path),
                video_np,
                fps=30,
                codec='libx264',
                quality=8
            )
            print(f"  Saved raw ground truth video: {output_path}")
        else:
            print(f"  ERROR: Cannot save video - imageio not available")
    
    print(f"\n✓ Complete! Raw ground truth videos saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode ground truth videos directly from dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./ground_truth_raw",
                       help="Output directory for videos")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                       help="Path to Cosmos tokenizer directory")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to decode")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    decode_ground_truth_raw(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_dir,
        num_samples=args.num_samples,
        device=args.device
    )
