#!/usr/bin/env python3
"""Compare predicted latents directly to ground truth to see if model is learning."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from flow_hwm.config import FlowHWMConfigMedium
from flow_hwm.model import FlowHWM, create_flow_hwm
from flow_hwm.inference import load_model_from_checkpoint, generate_video_latents
from flow_hwm.dataset_latent import FlowHWMDataset

def main():
    checkpoint_path = "/media/skr/storage/robot_world/humanoid_wm/checkpoints_flow_hwm_medium/model-60000.pt"
    data_dir = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = FlowHWMConfigMedium()
    
    # Load model
    print("Loading model...")
    model = load_model_from_checkpoint(checkpoint_path, config, device)
    
    # Load dataset
    print("Loading dataset...")
    dataset = FlowHWMDataset(
        data_dir=data_dir,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        latent_dim=config.latent_dim,
        max_shards=1
    )
    
    # Get a sample
    sample = dataset[0]
    v_p = sample["latent_past"].unsqueeze(0).to(device)
    a_p = sample["actions_past"].unsqueeze(0).to(device)
    a_f = sample["actions_future"].unsqueeze(0).to(device)
    latent_future_gt = sample["latent_future"].unsqueeze(0).to(device)
    
    print("\n" + "="*60)
    print("GROUND TRUTH LATENT")
    print("="*60)
    print(f"Shape: {latent_future_gt.shape}")
    print(f"Min: {latent_future_gt.min().item():.6f}")
    print(f"Max: {latent_future_gt.max().item():.6f}")
    print(f"Mean: {latent_future_gt.mean().item():.6f}")
    print(f"Std: {latent_future_gt.std().item():.6f}")
    
    # Generate prediction
    print("\n" + "="*60)
    print("GENERATING PREDICTION...")
    print("="*60)
    with torch.no_grad():
        predicted_latents = generate_video_latents(
            model, v_p, a_p, a_f,
            num_steps=50,
            cfg_scale=1.5,
            verbose=False,
        )
    
    print("\n" + "="*60)
    print("PREDICTED LATENT")
    print("="*60)
    print(f"Shape: {predicted_latents.shape}")
    print(f"Min: {predicted_latents.min().item():.6f}")
    print(f"Max: {predicted_latents.max().item():.6f}")
    print(f"Mean: {predicted_latents.mean().item():.6f}")
    print(f"Std: {predicted_latents.std().item():.6f}")
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    # MSE between predicted and ground truth
    mse = torch.nn.functional.mse_loss(predicted_latents, latent_future_gt)
    print(f"MSE between predicted and GT: {mse.item():.6f}")
    
    # Per-channel MSE
    print("\nPer-channel MSE (first 5 channels):")
    for c in range(min(5, predicted_latents.shape[1])):
        ch_mse = torch.nn.functional.mse_loss(
            predicted_latents[0, c], 
            latent_future_gt[0, c]
        )
        print(f"  Channel {c}: {ch_mse.item():.6f}")
    
    # Check if they're similar at all
    print("\n" + "="*60)
    print("SIMILARITY ANALYSIS")
    print("="*60)
    
    # Clip predicted to [-1, 1] and compare again
    predicted_clipped = torch.clamp(predicted_latents, -1.0, 1.0)
    mse_clipped = torch.nn.functional.mse_loss(predicted_clipped, latent_future_gt)
    print(f"MSE after clipping predicted to [-1, 1]: {mse_clipped.item():.6f}")
    
    # Check correlation
    pred_flat = predicted_latents.flatten()
    gt_flat = latent_future_gt.flatten()
    correlation = torch.corrcoef(torch.stack([pred_flat, gt_flat]))[0, 1]
    print(f"Correlation between predicted and GT: {correlation.item():.6f}")
    
    # Check if predicted is just noise
    pred_std = predicted_latents.std().item()
    gt_std = latent_future_gt.std().item()
    print(f"\nPredicted std: {pred_std:.6f}")
    print(f"GT std: {gt_std:.6f}")
    print(f"Ratio: {pred_std / gt_std:.6f}")
    
    if mse.item() > 0.5:
        print("\n⚠️  WARNING: Very high MSE! Model predictions are very different from ground truth.")
        print("   This suggests the model is not learning correctly.")
    
    if abs(correlation.item()) < 0.1:
        print("\n⚠️  WARNING: Very low correlation! Predictions are uncorrelated with ground truth.")
        print("   This suggests the model is generating random/noise-like outputs.")

if __name__ == "__main__":
    main()
