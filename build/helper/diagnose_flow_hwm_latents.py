#!/usr/bin/env python3
"""Diagnostic script to check latent ranges and conversion issues."""

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
    print("GROUND TRUTH LATENT STATISTICS")
    print("="*60)
    print(f"Shape: {latent_future_gt.shape}")
    print(f"Min: {latent_future_gt.min().item():.6f}")
    print(f"Max: {latent_future_gt.max().item():.6f}")
    print(f"Mean: {latent_future_gt.mean().item():.6f}")
    print(f"Std: {latent_future_gt.std().item():.6f}")
    print(f"Range: [{latent_future_gt.min().item():.6f}, {latent_future_gt.max().item():.6f}]")
    
    # Check per-channel statistics
    print("\nPer-channel statistics (first 5 channels):")
    for c in range(min(5, latent_future_gt.shape[1])):
        ch = latent_future_gt[0, c]
        print(f"  Channel {c}: min={ch.min().item():.6f}, max={ch.max().item():.6f}, mean={ch.mean().item():.6f}, std={ch.std().item():.6f}")
    
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
    print("PREDICTED LATENT STATISTICS")
    print("="*60)
    print(f"Shape: {predicted_latents.shape}")
    print(f"Min: {predicted_latents.min().item():.6f}")
    print(f"Max: {predicted_latents.max().item():.6f}")
    print(f"Mean: {predicted_latents.mean().item():.6f}")
    print(f"Std: {predicted_latents.std().item():.6f}")
    print(f"Range: [{predicted_latents.min().item():.6f}, {predicted_latents.max().item():.6f}]")
    
    # Check per-channel statistics
    print("\nPer-channel statistics (first 5 channels):")
    for c in range(min(5, predicted_latents.shape[1])):
        ch = predicted_latents[0, c]
        print(f"  Channel {c}: min={ch.min().item():.6f}, max={ch.max().item():.6f}, mean={ch.mean().item():.6f}, std={ch.std().item():.6f}")
    
    # Compare ranges
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"GT range: [{latent_future_gt.min().item():.6f}, {latent_future_gt.max().item():.6f}]")
    print(f"Pred range: [{predicted_latents.min().item():.6f}, {predicted_latents.max().item():.6f}]")
    print(f"GT should be in [-1, 1] (from normalized tokens)")
    print(f"Pred is in range: [{predicted_latents.min().item():.6f}, {predicted_latents.max().item():.6f}]")
    
    # Check if we need to normalize/clip
    print("\n" + "="*60)
    print("CONVERSION ANALYSIS")
    print("="*60)
    
    # Try current conversion
    latents_3ch_gt = latent_future_gt[:, :3, :, :, :]
    latents_normalized_gt = (latents_3ch_gt + 1.0) / 2.0
    tokens_gt = (latents_normalized_gt * 65535.0).round().long().clamp(0, 65535)
    
    latents_3ch_pred = predicted_latents[:, :3, :, :, :]
    latents_normalized_pred = (latents_3ch_pred + 1.0) / 2.0
    tokens_pred = (latents_normalized_pred * 65535.0).round().long().clamp(0, 65535)
    
    print(f"GT tokens range: [{tokens_gt.min().item()}, {tokens_gt.max().item()}]")
    print(f"Pred tokens range: [{tokens_pred.min().item()}, {tokens_pred.max().item()}]")
    
    # Check if pred latents are outside [-1, 1]
    out_of_range = (predicted_latents < -1.0).any() or (predicted_latents > 1.0).any()
    print(f"\nPredicted latents outside [-1, 1]: {out_of_range}")
    if out_of_range:
        print(f"  Values < -1: {(predicted_latents < -1.0).sum().item()} elements")
        print(f"  Values > 1: {(predicted_latents > 1.0).sum().item()} elements")
        print(f"  Min value: {predicted_latents.min().item():.6f}")
        print(f"  Max value: {predicted_latents.max().item():.6f}")
        print("\n  SOLUTION: Need to clip or normalize predicted latents before conversion!")

if __name__ == "__main__":
    main()
