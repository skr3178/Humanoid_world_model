"""Δ𝑡PSNR metric for measuring controllability in video generation

This metric measures how much impact latent actions have on video generation
by comparing:
- PSNR between ground-truth and frames generated with ground-truth inferred actions
- PSNR between ground-truth and frames generated with random actions

Δ𝑡PSNR = PSNR(𝑥𝑡, 𝑥̂𝑡) - PSNR(𝑥𝑡, 𝑥̂′𝑡)

where:
- 𝑥𝑡: ground-truth frame at time t
- 𝑥̂𝑡: frame generated from latent actions inferred from ground-truth frames
- 𝑥̂′𝑡: frame generated from randomly sampled latent actions
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List

from .psnr_metric import psnr


class DeltaPSNR:
    """Calculate Δ𝑡PSNR controllability metric"""
    
    def __init__(
        self,
        tokenizer,
        lam,
        dynamics_model,
        device: str = "cuda",
        t: int = 4,
        num_actions: int = 8,
    ):
        """
        Args:
            tokenizer: Trained video tokenizer
            lam: Trained Latent Action Model (LAM)
            dynamics_model: Trained dynamics model
            device: Device to run on
            t: Time step to evaluate (default: 4 as per paper)
            num_actions: Number of discrete actions (default: 8)
        """
        self.tokenizer = tokenizer.to(device).eval()
        self.lam = lam.to(device).eval()
        self.dynamics_model = dynamics_model.to(device).eval()
        self.device = device
        self.t = t
        self.num_actions = num_actions
    
    @torch.no_grad()
    def infer_actions_from_ground_truth(
        self,
        past_frames: torch.Tensor,
        next_frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        Infer latent actions from ground-truth frames using LAM.
        
        Args:
            past_frames: Past frames of shape (B, T, C, H, W)
            next_frame: Next frame of shape (B, C, H, W)
        
        Returns:
            Inferred action indices of shape (B, T, H_patches, W_patches)
        """
        _, actions, _ = self.lam(past_frames, next_frame)
        return actions
    
    @torch.no_grad()
    def generate_frame_with_actions(
        self,
        prompt_frame: torch.Tensor,
        actions: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Generate frames using given actions.
        
        Args:
            prompt_frame: Initial frame of shape (B, C, H, W) or (C, H, W)
            actions: Action indices of shape (B, T, H_patches, W_patches) or (B, T)
            num_frames: Number of frames to generate
        
        Returns:
            Generated video of shape (num_frames, C, H, W) or (B, num_frames, C, H, W)
        """
        # Ensure prompt_frame has batch dimension
        if prompt_frame.dim() == 3:
            prompt_frame = prompt_frame.unsqueeze(0)
        prompt_frame = prompt_frame.to(self.device)
        
        # Normalize to [0, 1] if needed
        if prompt_frame.max() > 1.0:
            prompt_frame = prompt_frame / 255.0
        
        # Tokenize initial frame
        prompt_tokens = self.tokenizer.encode(prompt_frame.unsqueeze(1))  # (B, 1, H_patches, W_patches)
        
        # Handle different action formats
        if actions.dim() == 4:
            # Spatial action map: (B, T, H_patches, W_patches)
            # Convert to temporal actions by taking mode or mean per frame
            B, T, H_p, W_p = actions.shape
            # Flatten spatial dimensions and take mode
            actions_flat = actions.view(B, T, -1)  # (B, T, H_p*W_p)
            # Use most common action per frame
            actions_temporal = torch.mode(actions_flat, dim=-1)[0]  # (B, T)
        elif actions.dim() == 2:
            # Temporal actions: (B, T)
            actions_temporal = actions
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}")
        
        # Generate frames autoregressively
        tokens = [prompt_tokens]
        B = prompt_tokens.shape[0]
        
        for t_idx in range(min(len(actions_temporal[0]), num_frames - 1)):
            # Get action for this timestep
            if actions_temporal.dim() == 2:
                action = actions_temporal[:, t_idx]  # (B,)
            else:
                action = actions_temporal[t_idx]  # (B,)
            
            # Expand action to spatial dimensions if needed
            _, _, H_patches, W_patches = tokens[-1].shape
            if actions.dim() == 4:
                # Use spatial action map
                action_map = actions[:, t_idx]  # (B, H_patches, W_patches)
            else:
                # Expand temporal action to spatial map
                action_map = action.unsqueeze(-1).unsqueeze(-1).expand(
                    B, H_patches, W_patches
                )  # (B, H_patches, W_patches)
            
            # Predict next tokens using dynamics model with MaskGIT
            next_tokens = self._maskgit_sample(
                tokens[-1],
                action_map.unsqueeze(1),  # (B, 1, H_patches, W_patches)
                steps=25,
                temperature=2.0,
            )
            tokens.append(next_tokens)
        
        # Decode tokens to frames
        frames = []
        for token_seq in tokens:
            if isinstance(token_seq, list):
                token_tensor = torch.cat(token_seq, dim=1)
            else:
                token_tensor = token_seq
            
            # Decode
            frame = self.tokenizer.decode(token_tensor)  # (B, T, C, H, W)
            frames.append(frame.squeeze(1))  # (B, C, H, W)
        
        # Stack frames: (num_frames, B, C, H, W) -> (B, num_frames, C, H, W) or (num_frames, C, H, W)
        video = torch.stack(frames, dim=1)  # (B, num_frames, C, H, W)
        if B == 1:
            video = video.squeeze(0)  # (num_frames, C, H, W)
        
        return video
    
    @torch.no_grad()
    def _maskgit_sample(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        steps: int,
        temperature: float,
    ) -> torch.Tensor:
        """Sample next tokens using MaskGIT iterative refinement"""
        B, T, H_patches, W_patches = tokens.shape
        
        # Initialize next frame tokens
        next_tokens = torch.randint(
            0,
            self.tokenizer.quantizer.num_codes,
            (B, 1, H_patches, W_patches),
            device=self.device,
        )
        
        # Concatenate tokens
        all_tokens = torch.cat([tokens, next_tokens], dim=1)  # (B, T+1, H_patches, W_patches)
        
        # Expand actions to match token sequence
        if actions.dim() == 3:
            # (B, 1, H_patches, W_patches) -> (B, T+1, H_patches, W_patches)
            actions_expanded = actions.expand(-1, T + 1, -1, -1)
        else:
            # Convert to temporal actions and expand
            actions_temporal = actions.view(B, -1).mode(dim=-1)[0]  # (B,)
            actions_expanded = actions_temporal.unsqueeze(1).expand(-1, T + 1)
        
        # Apply MaskGIT iterative refinement
        refined_tokens = self.dynamics_model.iterative_refinement(
            all_tokens,
            actions_expanded,
            steps=steps,
            temperature=temperature,
            r=0.5,
        )
        
        # Extract next frame tokens
        next_tokens = refined_tokens[:, -1:]  # (B, 1, H_patches, W_patches)
        return next_tokens
    
    @torch.no_grad()
    def compute(
        self,
        ground_truth_frames: torch.Tensor,
        prompt_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[float, dict]:
        """
        Compute Δ𝑡PSNR metric.
        
        Args:
            ground_truth_frames: Ground-truth video frames of shape (T, C, H, W) or (B, T, C, H, W)
            prompt_frame: Initial prompt frame (if None, uses first frame of ground_truth)
        
        Returns:
            delta_psnr: Δ𝑡PSNR value
            metrics_dict: Dictionary with detailed metrics
        """
        # Handle batch dimension
        if ground_truth_frames.dim() == 4:
            # (T, C, H, W) - single video
            ground_truth_frames = ground_truth_frames.unsqueeze(0)  # (1, T, C, H, W)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        B, T, C, H, W = ground_truth_frames.shape
        
        # Use first frame as prompt if not provided
        if prompt_frame is None:
            prompt_frame = ground_truth_frames[:, 0]  # (B, C, H, W)
        
        # Ensure we have enough frames
        if T < self.t + 1:
            raise ValueError(f"Need at least {self.t + 1} frames, got {T}")
        
        # Get ground-truth frame at time t
        gt_frame_t = ground_truth_frames[:, self.t]  # (B, C, H, W)
        
        # Get past frames (frames 0 to t-1)
        past_frames = ground_truth_frames[:, :self.t]  # (B, t, C, H, W)
        
        # Step 1: Infer actions from ground-truth
        # Use frames 0 to t-1 as past, frame t as next
        next_frame_gt = ground_truth_frames[:, self.t]  # (B, C, H, W)
        inferred_actions = self.infer_actions_from_ground_truth(
            past_frames, next_frame_gt
        )  # (B, t, H_patches, W_patches)
        
        # Step 2: Generate frame with inferred actions
        # Use only actions up to time t-1 to generate frame at t
        actions_for_generation = inferred_actions[:, :self.t]  # (B, t, H_patches, W_patches)
        generated_with_inferred = self.generate_frame_with_actions(
            prompt_frame,
            actions_for_generation,
            num_frames=self.t + 1,
        )  # (B, t+1, C, H, W) or (t+1, C, H, W)
        
        if generated_with_inferred.dim() == 4:
            generated_with_inferred = generated_with_inferred.unsqueeze(0)
        
        x_hat_t = generated_with_inferred[:, self.t]  # (B, C, H, W)
        
        # Step 3: Generate frame with random actions
        # Sample random actions from categorical distribution
        B, t, H_patches, W_patches = inferred_actions.shape
        random_actions = torch.randint(
            0,
            self.num_actions,
            (B, t, H_patches, W_patches),
            device=self.device,
        )
        
        generated_with_random = self.generate_frame_with_actions(
            prompt_frame,
            random_actions,
            num_frames=self.t + 1,
        )  # (B, t+1, C, H, W) or (t+1, C, H, W)
        
        if generated_with_random.dim() == 4:
            generated_with_random = generated_with_random.unsqueeze(0)
        
        x_hat_prime_t = generated_with_random[:, self.t]  # (B, C, H, W)
        
        # Step 4: Calculate PSNR values
        psnr_inferred = psnr(gt_frame_t, x_hat_t, max_val=1.0, reduction='mean')
        psnr_random = psnr(gt_frame_t, x_hat_prime_t, max_val=1.0, reduction='mean')
        
        # Step 5: Calculate Δ𝑡PSNR
        delta_psnr = psnr_inferred - psnr_random
        
        metrics_dict = {
            'delta_psnr': delta_psnr.item(),
            'psnr_inferred': psnr_inferred.item(),
            'psnr_random': psnr_random.item(),
            't': self.t,
        }
        
        return delta_psnr.item(), metrics_dict
