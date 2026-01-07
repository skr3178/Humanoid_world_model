"""PSNR (Peak Signal-to-Noise Ratio) metric for video generation evaluation"""

import torch
import torch.nn.functional as F
import math
from typing import Union, Tuple


def psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Calculate PSNR between two images/videos.
    
    Args:
        img1: First image/video tensor of shape (..., C, H, W) or (..., H, W, C)
        img2: Second image/video tensor, same shape as img1
        max_val: Maximum pixel value (1.0 for normalized [0,1], 255.0 for uint8)
        reduction: 'mean' for average PSNR, 'none' for per-sample PSNR
    
    Returns:
        PSNR value(s) in dB
    """
    # Ensure tensors are on same device and have same shape
    img1 = img1.to(img2.device)
    
    # Handle different input formats
    if img1.shape != img2.shape:
        # Try to match shapes by resizing
        if img1.dim() == 4 and img2.dim() == 4:
            # (B, C, H, W) format
            img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
        elif img1.dim() == 5 and img2.dim() == 5:
            # (B, T, C, H, W) format
            img2 = F.interpolate(
                img2.view(-1, *img2.shape[2:]),
                size=img1.shape[3:],
                mode='bilinear',
                align_corners=False
            ).view(*img2.shape[:2], *img1.shape[2:])
    
    # Normalize to [0, 1] if needed
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # Calculate MSE
    mse = torch.mean((img1 - img2) ** 2, dim=(-3, -2, -1))  # Average over C, H, W
    
    # Handle case where images are identical (MSE = 0)
    mse = torch.clamp(mse, min=1e-10)
    
    # Calculate PSNR
    psnr_val = 20 * torch.log10(torch.tensor(max_val, device=mse.device)) - 10 * torch.log10(mse)
    
    if reduction == 'mean':
        return psnr_val.mean()
    elif reduction == 'none':
        return psnr_val
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def psnr_per_frame(
    video1: torch.Tensor,
    video2: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Calculate PSNR for each frame in a video sequence.
    
    Args:
        video1: First video tensor of shape (T, C, H, W) or (B, T, C, H, W)
        video2: Second video tensor, same shape as video1
        max_val: Maximum pixel value
    
    Returns:
        PSNR values per frame of shape (T,) or (B, T)
    """
    if video1.dim() == 4:
        # (T, C, H, W) - single video
        T = video1.shape[0]
        psnr_values = []
        for t in range(T):
            psnr_t = psnr(video1[t:t+1], video2[t:t+1], max_val=max_val, reduction='none')
            psnr_values.append(psnr_t)
        return torch.stack(psnr_values)
    elif video1.dim() == 5:
        # (B, T, C, H, W) - batch of videos
        B, T = video1.shape[:2]
        psnr_values = []
        for b in range(B):
            for t in range(T):
                psnr_bt = psnr(
                    video1[b:b+1, t:t+1],
                    video2[b:b+1, t:t+1],
                    max_val=max_val,
                    reduction='none'
                )
                psnr_values.append(psnr_bt)
        return torch.stack(psnr_values).view(B, T)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {video1.dim()}D")
