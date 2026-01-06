"""Data collator with masking for Masked-HWM (v2.0 format)."""

import math
import random
import torch
from typing import Dict, List, Union
from dataclasses import dataclass

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from masked_hwm.config import MaskedHWMConfig


def cosine_schedule(u: torch.Tensor) -> torch.Tensor:
    """Cosine schedule for masking.
    
    Args:
        u: Values in [0, 1]
        
    Returns:
        Cosine schedule values
    """
    return torch.cos(u * math.pi / 2)


class MaskedHWMCollator:
    """Data collator for Masked-HWM training (v2.0 format).
    
    Handles factorized video tokens and applies masking to future clips.
    """
    
    def __init__(self, config: MaskedHWMConfig):
        """Initialize collator.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.mask_token_id = config.mask_token_id
        self.num_factors = config.num_factored_vocabs
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate and mask a batch of samples.
        
        v2.0 format:
        - video_past: (num_clips, num_factors, H, W)
        - video_future: (num_clips, num_factors, H, W)
        - actions_past: (T_p_frames, action_dim)
        - actions_future: (T_f_frames, action_dim)
        
        Args:
            features: List of samples from dataset
            
        Returns:
            Dictionary with batched tensors ready for model
        """
        batch_size = len(features)
        H = W = self.config.spatial_size
        
        # Stack batch - need to handle variable clip counts
        video_past = torch.stack([f["video_past"] for f in features])  # (B, num_clips, F, H, W)
        video_future = torch.stack([f["video_future"] for f in features])  # (B, num_clips, F, H, W)
        actions_past = torch.stack([f["actions_past"] for f in features])  # (B, T_p_frames, action_dim)
        actions_future = torch.stack([f["actions_future"] for f in features])  # (B, T_f_frames, action_dim)
        
        # Get dimensions
        T_f_clips = video_future.shape[1]
        num_factors = video_future.shape[2]
        
        # Create labels (ground truth)
        labels = video_future.clone()
        
        # Transpose video for model: (B, num_clips, F, H, W) -> (B, F, num_clips, H, W)
        video_past_model = video_past.permute(0, 2, 1, 3, 4)
        video_future_input = video_future.permute(0, 2, 1, 3, 4).clone()
        video_future_labels = labels.permute(0, 2, 1, 3, 4)
        
        # Apply corruption (random token replacement)
        if self.config.max_corrupt_rate > 0:
            corrupt_rate = random.uniform(0, self.config.max_corrupt_rate)
            corrupt_mask = torch.rand(batch_size, T_f_clips, H, W) < corrupt_rate
            
            for f in range(num_factors):
                random_tokens = torch.randint(
                    0, self.config.vocab_size,
                    size=(batch_size, T_f_clips, H, W),
                    dtype=video_future_input.dtype
                )
                video_future_input[:, f] = torch.where(
                    corrupt_mask, random_tokens, video_future_input[:, f]
                )
        
        # Create mask for future tokens (per-clip, shared across factors)
        mask = torch.zeros(batch_size, T_f_clips, H, W, dtype=torch.bool)
        
        for t in range(T_f_clips):
            # Random threshold per clip
            r = random.uniform(0, 1)
            threshold = r
            
            # Create per-clip mask
            clip_mask = torch.rand(batch_size, H, W) < threshold
            mask[:, t] = clip_mask
        
        # Ensure at least some tokens are masked
        if mask.sum() == 0:
            for t in range(T_f_clips):
                num_tokens = H * W
                num_to_mask = max(1, num_tokens // 4)  # Mask at least 25%
                flat_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
                for b in range(batch_size):
                    indices = torch.randperm(num_tokens)[:num_to_mask]
                    flat_mask[b, indices] = True
                mask[:, t] = flat_mask.reshape(batch_size, H, W)
        
        # Apply mask to all factors
        for f in range(num_factors):
            video_future_input[:, f][mask] = self.mask_token_id
        
        return {
            "video_past": video_past_model,  # (B, F, T_p_clips, H, W)
            "video_future": video_future_input,  # (B, F, T_f_clips, H, W) - masked
            "video_future_labels": video_future_labels,  # (B, F, T_f_clips, H, W)
            "mask": mask,  # (B, T_f_clips, H, W)
            "actions_past": actions_past,  # (B, T_p_frames, action_dim)
            "actions_future": actions_future,  # (B, T_f_frames, action_dim)
        }
