#!/usr/bin/env python3
"""Test if ground truth latents decode correctly to verify our conversion is working."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from flow_hwm.dataset_latent import FlowHWMDataset
from flow_hwm.config import FlowHWMConfigMedium
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.utils import tensor2numpy
import imageio

def latents_to_factorized_tokens(latents: torch.Tensor) -> torch.Tensor:
    """Convert continuous latents to factorized token format."""
    B, C, T, H, W = latents.shape
    
    # Clip to [-1, 1]
    latents = torch.clamp(latents, -1.0, 1.0)
    
    # Take first 3 channels
    if C >= 3:
        latents_3ch = latents[:, :3, :, :, :]
    else:
        latents_3ch = latents.repeat(1, (3 // C + 1), 1, 1, 1)[:, :3, :, :, :]
    
    # Reverse normalization
    latents_normalized = (latents_3ch + 1.0) / 2.0
    latents_normalized = torch.clamp(latents_normalized, 0.0, 1.0)
    
    # Quantize
    tokens_float = latents_normalized * 65535.0
    tokens = tokens_float.round().long()
    tokens = torch.clamp(tokens, 0, 65535)
    
    return tokens

def decode_latents_to_video(latents: torch.Tensor, decoder, device: str = "cuda") -> np.ndarray:
    """Decode continuous latents to video frames."""
    B, C, T, H, W = latents.shape
    
    tokens = latents_to_factorized_tokens(latents)
    
    decoded_frames = []
    for t in range(T):
        clip_tokens = tokens[:, :, t, :, :].to(device)
        with torch.no_grad():
            decoded_clip = decoder.decode(clip_tokens).float()
            decoded_frames.append(decoded_clip)
    
    video = torch.cat(decoded_frames, dim=2)
    video_np = tensor2numpy(video)
    
    return video_np

def main():
    data_dir = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
    tokenizer_dir = "/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = FlowHWMConfigMedium()
    
    # Load dataset
    print("Loading dataset...")
    dataset = FlowHWMDataset(
        data_dir=data_dir,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        latent_dim=config.latent_dim,
        max_shards=1
    )
    
    # Load decoder
    print("Loading decoder...")
    decoder = CausalVideoTokenizer(
        checkpoint_dec=f"{tokenizer_dir}/decoder.jit",
        device=device,
        dtype="bfloat16"
    )
    
    # Get a sample
    sample = dataset[0]
    latent_future_gt = sample["latent_future"].unsqueeze(0).to(device)
    
    print(f"\nGround truth latent shape: {latent_future_gt.shape}")
    print(f"Range: [{latent_future_gt.min().item():.6f}, {latent_future_gt.max().item():.6f}]")
    
    # Decode ground truth
    print("\nDecoding ground truth latents...")
    gt_video_np = decode_latents_to_video(latent_future_gt, decoder, device)
    gt_video = gt_video_np[0]
    
    print(f"Decoded video shape: {gt_video.shape}")
    print(f"Video range: [{gt_video.min()}, {gt_video.max()}]")
    print(f"Video dtype: {gt_video.dtype}")
    
    # Save video
    output_path = "/media/skr/storage/robot_world/humanoid_wm/test_gt_decoding.mp4"
    imageio.mimsave(
        str(output_path),
        gt_video,
        fps=30,
        codec='libx264',
        quality=8
    )
    print(f"\nSaved ground truth video to: {output_path}")
    print("If this video looks correct, then the conversion function is working.")
    print("If this video is also pixelated/random, then the conversion is the problem.")

if __name__ == "__main__":
    main()
