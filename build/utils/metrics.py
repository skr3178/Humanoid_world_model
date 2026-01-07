"""Metrics for evaluation."""

import torch
import numpy as np
from typing import Optional


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted video in range [0, 1]
        target: Target video in range [0, 1]
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Compute Structural Similarity Index.
    
    Args:
        pred: Predicted video
        target: Target video
        window_size: Size of the Gaussian window
        
    Returns:
        SSIM value
    """
    # Simplified SSIM computation
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_pred = pred.mean()
    mu_target = target.mean()
    sigma_pred = pred.var()
    sigma_target = target.var()
    sigma_pred_target = ((pred - mu_pred) * (target - mu_target)).mean()
    
    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    
    return ssim.item()
