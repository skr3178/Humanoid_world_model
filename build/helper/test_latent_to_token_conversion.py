#!/usr/bin/env python3
"""Test different approaches for converting latents back to tokens.

This script tests multiple methods for reversing the token-to-latent conversion
and measures reconstruction errors to find the best approach.
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flow_hwm.dataset_latent import FlowHWMDataset

# Import base dataset to get original tokens
sys.path.insert(0, "/media/skr/storage/robot_world/humanoid_wm/build")
from data.dataset import HumanoidWorldModelDataset


def latents_to_tokens_method1_first3ch(latents: torch.Tensor) -> torch.Tensor:
    """Method 1: Simply take first 3 channels (current simple approach)."""
    B, C, T, H, W = latents.shape
    latents = torch.clamp(latents, -1.0, 1.0)
    
    if C >= 3:
        latents_3ch = latents[:, :3, :, :, :]
    else:
        latents_3ch = latents.repeat(1, (3 // C + 1), 1, 1, 1)[:, :3, :, :, :]
    
    latents_normalized = (latents_3ch + 1.0) / 2.0
    latents_normalized = torch.clamp(latents_normalized, 0.0, 1.0)
    tokens_float = latents_normalized * 65535.0
    tokens = tokens_float.round().long()
    return torch.clamp(tokens, 0, 65535)


def latents_to_tokens_method2_channel_selection(latents: torch.Tensor) -> torch.Tensor:
    """Method 2: Select channels at indices 0, 5, 10 (one from each group)."""
    B, C, T, H, W = latents.shape
    latents = torch.clamp(latents, -1.0, 1.0)
    
    if C >= 15:
        # For latent_dim=16: take channels 0, 5, 10
        latents_3ch = torch.stack([
            latents[:, 0, :, :, :],  # Factor 0 (from group 0-4)
            latents[:, 5, :, :, :],  # Factor 1 (from group 5-9)
            latents[:, 10, :, :, :], # Factor 2 (from group 10-14)
        ], dim=1)
    elif C >= 3:
        # Fallback: take evenly spaced channels
        indices = [0, C // 3, 2 * C // 3]
        latents_3ch = torch.stack([latents[:, i, :, :, :] for i in indices], dim=1)
    else:
        latents_3ch = latents.repeat(1, (3 // C + 1), 1, 1, 1)[:, :3, :, :, :]
    
    latents_normalized = (latents_3ch + 1.0) / 2.0
    latents_normalized = torch.clamp(latents_normalized, 0.0, 1.0)
    tokens_float = latents_normalized * 65535.0
    tokens = tokens_float.round().long()
    return torch.clamp(tokens, 0, 65535)


def latents_to_tokens_method3_averaging(latents: torch.Tensor) -> torch.Tensor:
    """Method 3: Average each group of channels (documented approach)."""
    B, C, T, H, W = latents.shape
    latents = torch.clamp(latents, -1.0, 1.0)
    
    if C >= 15:
        # For latent_dim=16: average groups [0-4], [5-9], [10-14]
        factor0 = latents[:, 0:5, :, :, :].mean(dim=1, keepdim=True)
        factor1 = latents[:, 5:10, :, :, :].mean(dim=1, keepdim=True)
        factor2 = latents[:, 10:15, :, :, :].mean(dim=1, keepdim=True)
        latents_3ch = torch.cat([factor0, factor1, factor2], dim=1)
    elif C >= 3:
        # For other sizes, average evenly
        repeats = C // 3
        remainder = C % 3
        factors = []
        for i in range(3):
            start_idx = i * repeats
            end_idx = (i + 1) * repeats
            if i == 2 and remainder > 0:
                # Last factor gets remainder channels too
                end_idx += remainder
            factor = latents[:, start_idx:end_idx, :, :, :].mean(dim=1, keepdim=True)
            factors.append(factor)
        latents_3ch = torch.cat(factors, dim=1)
    else:
        latents_3ch = latents.repeat(1, (3 // C + 1), 1, 1, 1)[:, :3, :, :, :]
    
    latents_normalized = (latents_3ch + 1.0) / 2.0
    latents_normalized = torch.clamp(latents_normalized, 0.0, 1.0)
    tokens_float = latents_normalized * 65535.0
    tokens = tokens_float.round().long()
    return torch.clamp(tokens, 0, 65535)


def latents_to_tokens_method4_median(latents: torch.Tensor) -> torch.Tensor:
    """Method 4: Use median instead of mean for each group (more robust to outliers)."""
    B, C, T, H, W = latents.shape
    latents = torch.clamp(latents, -1.0, 1.0)
    
    if C >= 15:
        # For latent_dim=16: median of groups [0-4], [5-9], [10-14]
        factor0 = latents[:, 0:5, :, :, :].median(dim=1, keepdim=True)[0]
        factor1 = latents[:, 5:10, :, :, :].median(dim=1, keepdim=True)[0]
        factor2 = latents[:, 10:15, :, :, :].median(dim=1, keepdim=True)[0]
        latents_3ch = torch.cat([factor0, factor1, factor2], dim=1)
    elif C >= 3:
        repeats = C // 3
        remainder = C % 3
        factors = []
        for i in range(3):
            start_idx = i * repeats
            end_idx = (i + 1) * repeats
            if i == 2 and remainder > 0:
                end_idx += remainder
            factor = latents[:, start_idx:end_idx, :, :, :].median(dim=1, keepdim=True)[0]
            factors.append(factor)
        latents_3ch = torch.cat(factors, dim=1)
    else:
        latents_3ch = latents.repeat(1, (3 // C + 1), 1, 1, 1)[:, :3, :, :, :]
    
    latents_normalized = (latents_3ch + 1.0) / 2.0
    latents_normalized = torch.clamp(latents_normalized, 0.0, 1.0)
    tokens_float = latents_normalized * 65535.0
    tokens = tokens_float.round().long()
    return torch.clamp(tokens, 0, 65535)


def compute_reconstruction_error(gt_tokens: torch.Tensor, recon_tokens: torch.Tensor) -> dict:
    """Compute various error metrics between GT and reconstructed tokens."""
    errors = {
        'mean_abs_error': float((gt_tokens - recon_tokens).abs().float().mean().item()),
        'max_abs_error': float((gt_tokens - recon_tokens).abs().float().max().item()),
        'mean_squared_error': float(((gt_tokens - recon_tokens).float() ** 2).mean().item()),
        'per_factor_errors': [],
    }
    
    # Per-factor errors
    for factor_idx in range(3):
        factor_error = (gt_tokens[:, factor_idx, :, :, :] - recon_tokens[:, factor_idx, :, :, :]).abs().float()
        errors['per_factor_errors'].append({
            'factor': factor_idx,
            'mean': float(factor_error.mean().item()),
            'max': float(factor_error.max().item()),
        })
    
    return errors


def test_conversion_methods(data_dir: str, num_samples: int = 10, latent_dim: int = 16):
    """Test all conversion methods on dataset samples."""
    
    print("=" * 80)
    print("Testing Latent-to-Token Conversion Methods")
    print("=" * 80)
    
    # Load datasets
    print(f"\nLoading datasets from {data_dir}...")
    base_dataset = HumanoidWorldModelDataset(
        data_dir=data_dir,
        num_past_frames=2,
        num_future_frames=1,
        use_factored_tokens=True,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1,
    )
    
    flow_dataset = FlowHWMDataset(
        data_dir=data_dir,
        num_past_clips=2,
        num_future_clips=1,
        latent_dim=latent_dim,
        max_shards=1,
    )
    
    if len(base_dataset) == 0:
        print("ERROR: Dataset is empty")
        return
    
    num_samples = min(num_samples, len(base_dataset))
    print(f"Testing on {num_samples} samples...")
    
    methods = {
        'Method 1 (First 3 channels)': latents_to_tokens_method1_first3ch,
        'Method 2 (Channel selection 0,5,10)': latents_to_tokens_method2_channel_selection,
        'Method 3 (Averaging groups)': latents_to_tokens_method3_averaging,
        'Method 4 (Median groups)': latents_to_tokens_method4_median,
    }
    
    all_results = {name: [] for name in methods.keys()}
    
    for sample_idx in range(num_samples):
        print(f"\n{'=' * 60}")
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print(f"{'=' * 60}")
        
        # Get ground truth tokens
        base_sample = base_dataset[sample_idx]
        gt_tokens = base_sample['video_future']  # (T, 3, H, W)
        gt_tokens = gt_tokens.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, 3, T, H, W)
        
        # Get latents from flow dataset
        flow_sample = flow_dataset[sample_idx]
        latents = flow_sample['latent_future'].unsqueeze(0)  # (1, C, T, H, W)
        
        print(f"GT tokens shape: {gt_tokens.shape}, range: [{gt_tokens.min()}, {gt_tokens.max()}]")
        print(f"Latents shape: {latents.shape}, range: [{latents.min():.4f}, {latents.max():.4f}]")
        
        # Test each method
        for method_name, method_func in methods.items():
            recon_tokens = method_func(latents)
            errors = compute_reconstruction_error(gt_tokens, recon_tokens)
            all_results[method_name].append(errors)
            
            print(f"\n{method_name}:")
            print(f"  Reconstructed tokens range: [{recon_tokens.min()}, {recon_tokens.max()}]")
            print(f"  Mean absolute error: {errors['mean_abs_error']:.2f}")
            print(f"  Max absolute error: {errors['max_abs_error']:.2f}")
            print(f"  Mean squared error: {errors['mean_squared_error']:.2f}")
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (averaged over all samples)")
    print("=" * 80)
    
    for method_name, results in all_results.items():
        mean_mae = np.mean([r['mean_abs_error'] for r in results])
        mean_max = np.mean([r['max_abs_error'] for r in results])
        mean_mse = np.mean([r['mean_squared_error'] for r in results])
        
        print(f"\n{method_name}:")
        print(f"  Mean Absolute Error: {mean_mae:.2f}")
        print(f"  Mean Max Error: {mean_max:.2f}")
        print(f"  Mean Squared Error: {mean_mse:.2f}")
        
        # Per-factor breakdown
        for factor_idx in range(3):
            factor_errors = [r['per_factor_errors'][factor_idx]['mean'] for r in results]
            print(f"    Factor {factor_idx} mean error: {np.mean(factor_errors):.2f}")
    
    # Find best method
    print("\n" + "=" * 80)
    print("BEST METHOD (lowest mean absolute error):")
    print("=" * 80)
    
    best_method = None
    best_error = float('inf')
    for method_name, results in all_results.items():
        mean_mae = np.mean([r['mean_abs_error'] for r in results])
        if mean_mae < best_error:
            best_error = mean_mae
            best_method = method_name
    
    print(f"{best_method}: {best_error:.2f} mean absolute error")
    print("\nNote: Even the best method may have high errors due to the lossy nature")
    print("      of the token-to-latent conversion. Consider using original tokens")
    print("      for ground truth decoding instead of converting latents back.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test latent-to-token conversion methods")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to test",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=16,
        help="Latent dimension",
    )
    
    args = parser.parse_args()
    test_conversion_methods(args.data_dir, args.num_samples, args.latent_dim)
