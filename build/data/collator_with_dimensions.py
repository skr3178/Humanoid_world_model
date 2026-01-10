"""Data collator with masking for Masked-HWM (v2.0 format) - WITH DIMENSION PRINTING.

Paper Section 2.2 (Masked Video Modelling) and Experimental Setup:
- Each training sample: 2 fully unmasked past latents + 1 partially masked future latent
- Corruption: Random token replacements at rate ~ U(0, ρ_max) where ρ_max=0.2
- Masking: Applied ONLY to future latent using cosine schedule from MaskGIT
  - For each frame, sample r ~ U(0,1)
  - Compute threshold γ(r) = cos(r * π/2)
  - For each token, sample p ~ U(0,1), mask if p < γ(r)
"""

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


def cosine_schedule(r: float) -> float:
    """MaskGIT cosine schedule for masking threshold.

    γ(r) = cos(r * π/2)
    - r=0 → γ=1.0 (mask ~100% of tokens)
    - r=1 → γ=0.0 (mask ~0% of tokens)

    This creates a bias toward higher masking rates during training.

    Args:
        r: Random value sampled from U(0,1)

    Returns:
        Masking threshold γ(r) in [0, 1]
    """
    return math.cos(r * math.pi / 2)


class MaskedHWMCollator:
    """Data collator for Masked-HWM training (v2.0 format) - WITH DIMENSION PRINTING.

    Per paper:
    - Corruption: Applied to ALL tokens (past + future) with random replacements
    - Masking: Applied ONLY to future tokens using MaskGIT cosine schedule
    - Past latents remain fully unmasked (but may be corrupted)
    """

    def __init__(self, config: MaskedHWMConfig):
        """Initialize collator.

        Args:
            config: Model configuration
        """
        self.config = config
        # Embeddings use vocab_size + 1 to include mask token
        self.mask_token_id = config.mask_token_id
        self.num_factors = config.num_factored_vocabs
        self.verbose = False  # Set to True for debug output
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate and mask a batch of samples.

        Per paper (Section 2.2 + Experimental Setup):
        - 2 fully unmasked past latents + 1 partially masked future latent
        - Corruption: U(0, ρ_max) where ρ_max=0.2, applied to ALL tokens
        - Masking: MaskGIT cosine schedule, applied ONLY to future tokens

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
        print("\n" + "="*80)
        print("COLLATOR: Starting batch processing")
        print("="*80)
        
        batch_size = len(features)
        H = W = self.config.spatial_size

        # Print input features dimensions
        print(f"\n[STEP 0] Input features from dataset:")
        for i, f in enumerate(features):
            print(f"  Sample {i}:")
            print(f"    video_past: {f['video_past'].shape}")
            print(f"    video_future: {f['video_future'].shape}")
            print(f"    actions_past: {f['actions_past'].shape}")
            print(f"    actions_future: {f['actions_future'].shape}")

        # Stack batch
        video_past = torch.stack([f["video_past"] for f in features])  # (B, T_p_clips, F, H, W)
        video_future = torch.stack([f["video_future"] for f in features])  # (B, T_f_clips, F, H, W)
        actions_past = torch.stack([f["actions_past"] for f in features])  # (B, T_p_frames, action_dim)
        actions_future = torch.stack([f["actions_future"] for f in features])  # (B, T_f_frames, action_dim)

        print(f"\n[STEP 1] After stacking batch (batch_size={batch_size}):")
        print(f"  video_past: {video_past.shape} (B, T_p_clips, F, H, W)")
        print(f"  video_future: {video_future.shape} (B, T_f_clips, F, H, W)")
        print(f"  actions_past: {actions_past.shape} (B, T_p_frames, action_dim)")
        print(f"  actions_future: {actions_future.shape} (B, T_f_frames, action_dim)")

        # Get dimensions
        T_p_clips = video_past.shape[1]
        T_f_clips = video_future.shape[1]
        num_factors = video_future.shape[2]

        print(f"\n[STEP 2] Extracted dimensions:")
        print(f"  T_p_clips (past clips): {T_p_clips}")
        print(f"  T_f_clips (future clips): {T_f_clips}")
        print(f"  num_factors: {num_factors}")
        print(f"  H, W (spatial size): {H}, {W}")

        # Create labels (ground truth for future - BEFORE any corruption/masking)
        video_future_labels = video_future.clone()

        print(f"\n[STEP 3] Created labels (before corruption/masking):")
        print(f"  video_future_labels: {video_future_labels.shape} (B, T_f_clips, F, H, W)")

        # Transpose video for model: (B, T_clips, F, H, W) -> (B, F, T_clips, H, W)
        video_past_input = video_past.permute(0, 2, 1, 3, 4).clone()
        video_future_input = video_future.permute(0, 2, 1, 3, 4).clone()
        video_future_labels = video_future_labels.permute(0, 2, 1, 3, 4)

        print(f"\n[STEP 4] After permute (B, T_clips, F, H, W) -> (B, F, T_clips, H, W):")
        print(f"  video_past_input: {video_past_input.shape} (B, F, T_p_clips, H, W)")
        print(f"  video_future_input: {video_future_input.shape} (B, F, T_f_clips, H, W)")
        print(f"  video_future_labels: {video_future_labels.shape} (B, F, T_f_clips, H, W)")

        # =================================================================
        # STEP 1: CORRUPTION - Apply to BOTH past and future tokens
        # Paper: "we add noise to the latents by corrupting L with random
        # token replacements at a rate uniformly sampled from U(0, ρ_max)"
        #
        # Algorithm 1 pattern (matches 1xgpt/data.py):
        # - Sample u01 ~ U(0,1) once for entire batch
        # - For each token, sample r ~ U(0,1) and corrupt if r < ρ_max * u01
        # - This gives effective corruption rate ~ U(0, ρ_max)
        # =================================================================
        corrupt_rate = 0.0
        if self.config.max_corrupt_rate > 0:
            # Sample u01 ~ U(0,1) once for entire batch (Algorithm 1 pattern)
            u01 = random.random()
            effective_corrupt_rate = self.config.max_corrupt_rate * u01
            corrupt_rate = effective_corrupt_rate  # For logging/debugging

            print(f"\n[STEP 5] CORRUPTION (rate={corrupt_rate:.3f}):")
            print(f"  Before corruption:")
            print(f"    video_past_input: {video_past_input.shape}")
            print(f"    video_future_input: {video_future_input.shape}")

            # Corrupt PAST tokens (but they remain UNMASKED)
            # For each token: sample r ~ U(0,1), corrupt if r < ρ_max * u01
            r_past = torch.rand(batch_size, T_p_clips, H, W)
            corrupt_mask_past = r_past < effective_corrupt_rate
            print(f"  corrupt_mask_past: {corrupt_mask_past.shape} (B, T_p_clips, H, W)")
            print(f"  corrupt_mask_past.sum(): {corrupt_mask_past.sum().item()} tokens to corrupt")
            
            for f in range(num_factors):
                random_tokens = torch.randint(
                    0, self.config.vocab_size,
                    size=(batch_size, T_p_clips, H, W),
                    dtype=video_past_input.dtype,
                    device=video_past_input.device,
                )
                video_past_input[:, f] = torch.where(
                    corrupt_mask_past, random_tokens, video_past_input[:, f]
                )

            # Corrupt FUTURE tokens (before masking)
            # For each token: sample r ~ U(0,1), corrupt if r < ρ_max * u01
            r_future = torch.rand(batch_size, T_f_clips, H, W)
            corrupt_mask_future = r_future < effective_corrupt_rate
            print(f"  corrupt_mask_future: {corrupt_mask_future.shape} (B, T_f_clips, H, W)")
            print(f"  corrupt_mask_future.sum(): {corrupt_mask_future.sum().item()} tokens to corrupt")
            
            for f in range(num_factors):
                random_tokens = torch.randint(
                    0, self.config.vocab_size,
                    size=(batch_size, T_f_clips, H, W),
                    dtype=video_future_input.dtype,
                    device=video_future_input.device,
                )
                video_future_input[:, f] = torch.where(
                    corrupt_mask_future, random_tokens, video_future_input[:, f]
                )

            print(f"  After corruption:")
            print(f"    video_past_input: {video_past_input.shape}")
            print(f"    video_future_input: {video_future_input.shape}")
        else:
            print(f"\n[STEP 5] CORRUPTION: Skipped (max_corrupt_rate=0)")

        # =================================================================
        # STEP 2: MASKING - Apply ONLY to future tokens with cosine schedule
        # Paper: "apply masking to the future latents L_f using a per-frame
        # thresholding strategy... using a predefined scheduling function"
        #
        # For each frame: sample r ~ U(0,1), compute threshold γ(r) = cos(r*π/2)
        # For each token: sample p ~ U(0,1), mask if p < γ(r)
        # =================================================================

        print(f"\n[STEP 6] MASKING (applied only to future tokens):")
        print(f"  Before masking:")
        print(f"    video_future_input: {video_future_input.shape}")

        # Create mask for future tokens
        # Paper: per-frame threshold, but each sample in batch gets INDEPENDENT r
        mask = torch.zeros(batch_size, T_f_clips, H, W, dtype=torch.bool)
        print(f"  mask (initialized): {mask.shape} (B, T_f_clips, H, W)")

        for b in range(batch_size):
            for t in range(T_f_clips):
                # Sample r ~ U(0,1) for this (sample, frame) pair
                r = random.random()

                # Compute threshold using cosine schedule: γ(r) = cos(r * π/2)
                threshold = cosine_schedule(r)

                # For each token, sample p ~ U(0,1), mask if p < threshold
                token_probs = torch.rand(H, W)
                mask[b, t] = token_probs < threshold

        print(f"  mask (after cosine schedule): {mask.shape}")
        print(f"  mask.sum(): {mask.sum().item()} tokens masked")

        # Ensure at least SOME tokens are masked (avoid division by zero in loss)
        if mask.sum() == 0:
            print(f"  WARNING: No tokens masked! Forcing at least one token per sample...")
            # Force mask at least one random token per sample
            for b in range(batch_size):
                rand_t = random.randint(0, T_f_clips - 1)
                rand_h = random.randint(0, H - 1)
                rand_w = random.randint(0, W - 1)
                mask[b, rand_t, rand_h, rand_w] = True
            print(f"  mask.sum() (after forcing): {mask.sum().item()} tokens masked")

        # Apply mask token to ALL factors at masked positions
        # (mask is shared across factors - same positions masked for all 3 factors)
        for f in range(num_factors):
            video_future_input[:, f][mask] = self.mask_token_id

        print(f"  After applying mask tokens:")
        print(f"    video_future_input: {video_future_input.shape}")
        print(f"    mask: {mask.shape}")

        # Debug output (only if verbose)
        if self.verbose:
            masked_ratio = mask.float().mean().item()
            print(f"[COLLATOR] corrupt_rate={corrupt_rate:.3f}, masked={masked_ratio:.2%}",
                  file=sys.stderr, flush=True)

        result = {
            "video_past": video_past_input,  # (B, F, T_p_clips, H, W) - CORRUPTED only
            "video_future": video_future_input,  # (B, F, T_f_clips, H, W) - CORRUPTED + MASKED
            "video_future_labels": video_future_labels,  # (B, F, T_f_clips, H, W) - ORIGINAL
            "mask": mask,  # (B, T_f_clips, H, W) - True where tokens are masked
            "actions_past": actions_past,  # (B, T_p_frames, action_dim)
            "actions_future": actions_future,  # (B, T_f_frames, action_dim)
        }

        print(f"\n[STEP 7] Final output:")
        print(f"  video_past: {result['video_past'].shape} (B, F, T_p_clips, H, W) - CORRUPTED only")
        print(f"  video_future: {result['video_future'].shape} (B, F, T_f_clips, H, W) - CORRUPTED + MASKED")
        print(f"  video_future_labels: {result['video_future_labels'].shape} (B, F, T_f_clips, H, W) - ORIGINAL")
        print(f"  mask: {result['mask'].shape} (B, T_f_clips, H, W)")
        print(f"  actions_past: {result['actions_past'].shape} (B, T_p_frames, action_dim)")
        print(f"  actions_future: {result['actions_future'].shape} (B, T_f_frames, action_dim)")
        print("="*80 + "\n")

        return result
