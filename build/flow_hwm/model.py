"""Main Flow-HWM model for continuous latent video generation.

Flow Matching Humanoid World Model that predicts velocity fields for
transforming Gaussian noise into future video latents.

The model:
1. Projects continuous video latents to model dimension
2. Embeds actions using an MLP
3. Embeds timestep using sinusoidal + MLP embeddings
4. Processes through four-stream transformer with AdaLN
5. Projects output back to latent dimension (velocity prediction)

Architecture follows Section 2.3 of the paper with:
- Four streams: v_p (past video), v_f (future video), a_p, a_f (actions)
- Per-stream weights for AdaLN, QKV, and feedforward
- Joint attention integrating all streams
- RoPE: 3D for video, 1D for actions
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange

from .config import FlowHWMConfig
from .time_embedding import TimeEmbedding
from .ada_ln import AdaLN
from .transformer import FlowTransformer

# Import ActionEmbedding from masked_hwm (no modification needed)
import sys
sys.path.insert(0, "/media/skr/storage/robot_world/humanoid_wm/build")
from masked_hwm.action_embedding import ActionEmbedding


class FlowHWM(nn.Module):
    """Flow Matching Humanoid World Model.

    Predicts velocity field u_theta(X_t, P, t) where:
    - X_t is the noisy future video latent at time t
    - P = {v_p, a_p, a_f} is the conditioning context
    - t is the flow timestep in [0, 1]

    The velocity field is used to transform Gaussian noise X_0 into
    target video latent X_1 via ODE integration.

    Args:
        config: FlowHWMConfig with model hyperparameters
    """

    def __init__(self, config: FlowHWMConfig):
        super().__init__()
        self.config = config

        # Video projection: continuous latent -> model dimension with patching
        # Input: (B, latent_dim, T, H, W), Output: (B, d_model, T, H/2, W/2)
        self.video_proj = nn.Conv3d(
            in_channels=config.latent_dim,
            out_channels=config.d_model,
            kernel_size=(
                config.patch_size_temporal,
                config.patch_size_spatial,
                config.patch_size_spatial,
            ),
            stride=(
                config.patch_size_temporal,
                config.patch_size_spatial,
                config.patch_size_spatial,
            ),
            padding=0,
        )

        # Action embedding (reuse from masked_hwm)
        self.action_embedding = ActionEmbedding(
            action_dim=config.action_dim,
            d_model=config.d_model,
        )

        # Time embedding (sinusoidal + MLP)
        self.time_embedding = TimeEmbedding(
            d_model=config.d_model,
            sinusoidal_dim=256,
        )

        # Position embeddings for video (learnable)
        # Per temporal token position, shared across spatial positions
        self.video_pos_embed = nn.Parameter(
            torch.zeros(1, config.total_video_tokens, 1, 1, config.d_model)
        )

        # Position embeddings for actions (aligned to video token count)
        self.action_pos_embed = nn.Parameter(
            torch.zeros(1, config.total_video_tokens, config.d_model)
        )

        # Transformer
        self.transformer = FlowTransformer(config)

        # Output projection: model dim -> latent dim (velocity prediction)
        # Only applied to future video tokens
        self.final_ada_ln = AdaLN(config.d_model)
        self.video_unproj = nn.ConvTranspose3d(
            in_channels=config.d_model,
            out_channels=config.latent_dim,
            kernel_size=(
                config.patch_size_temporal,
                config.patch_size_spatial,
                config.patch_size_spatial,
            ),
            stride=(
                config.patch_size_temporal,
                config.patch_size_spatial,
                config.patch_size_spatial,
            ),
            padding=0,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Initialize position embeddings
        nn.init.normal_(self.video_pos_embed, std=config.init_std)
        nn.init.normal_(self.action_pos_embed, std=config.init_std)

    def _init_weights(self, module):
        """Initialize weights with standard normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

        # Xavier init for final projection (more stable than zero-init)
        if module is self.video_unproj:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _downsample_actions_to_clips(
        self,
        actions: torch.Tensor,
        num_clips: int,
    ) -> torch.Tensor:
        """Downsample frame-level actions to match video token count.

        Uses average pooling to match token rate.

        Args:
            actions: (B, T_frames, d_model)
            num_clips: Target number of video tokens

        Returns:
            Downsampled actions (B, num_clips, d_model)
        """
        B, T_frames, d = actions.shape

        if T_frames <= num_clips:
            # Pad or repeat if we have fewer frames than clips
            if T_frames == num_clips:
                return actions
            # Repeat last frame
            padding = actions[:, -1:].repeat(1, num_clips - T_frames, 1)
            return torch.cat([actions, padding], dim=1)

        # Average pool frames to match clip count
        frames_per_chunk = T_frames // num_clips
        chunks = []
        for i in range(num_clips):
            start = i * frames_per_chunk
            end = (i + 1) * frames_per_chunk if i < num_clips - 1 else T_frames
            chunk = actions[:, start:end].mean(dim=1, keepdim=True)
            chunks.append(chunk)

        return torch.cat(chunks, dim=1)  # (B, num_clips, d_model)

    def forward(
        self,
        v_f_noisy: torch.Tensor,
        v_p: torch.Tensor,
        a_p: torch.Tensor,
        a_f: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            v_f_noisy: Noisy future latent X_t, shape (B, C, T_f, H, W)
            v_p: Past video latent (conditioning), shape (B, C, T_p, H, W)
            a_p: Past actions, shape (B, T_p_frames, action_dim)
            a_f: Future actions, shape (B, T_f_frames, action_dim)
            t: Timesteps in [0, 1], shape (B,)

        Returns:
            velocity: Predicted velocity for v_f, shape (B, C, T_f, H, W)
        """
        B = v_f_noisy.shape[0]
        C = self.config.latent_dim
        T_p = v_p.shape[2]
        T_f = v_f_noisy.shape[2]
        H, W = v_p.shape[3], v_p.shape[4]

        # ===== Project video latents to model dimension =====
        # Input: (B, C, T, H, W) -> Output: (B, d_model, T, H, W)
        v_p_proj = self.video_proj(v_p)
        v_f_proj = self.video_proj(v_f_noisy)

        # Reshape to (B, T, H, W, d_model) for transformer
        v_p_proj = rearrange(v_p_proj, 'b d t h w -> b t h w d')
        v_f_proj = rearrange(v_f_proj, 'b d t h w -> b t h w d')

        # ===== Add position embeddings to video =====
        v_p_proj = v_p_proj + self.video_pos_embed[:, :T_p]
        v_f_proj = v_f_proj + self.video_pos_embed[:, T_p:T_p + T_f]

        # ===== Embed actions =====
        a_p_emb = self.action_embedding(a_p)  # (B, T_p_frames, d_model)
        a_f_emb = self.action_embedding(a_f)  # (B, T_f_frames, d_model)

        # Downsample actions to clip level
        a_p_emb = self._downsample_actions_to_clips(a_p_emb, T_p)
        a_f_emb = self._downsample_actions_to_clips(a_f_emb, T_f)

        # Add position embeddings to actions
        a_p_emb = a_p_emb + self.action_pos_embed[:, :T_p]
        a_f_emb = a_f_emb + self.action_pos_embed[:, T_p:T_p + T_f]

        # ===== Embed timestep =====
        t_emb = self.time_embedding(t)  # (B, d_model)

        # ===== Pass through transformer =====
        v_p_out, v_f_out, a_p_out, a_f_out = self.transformer(
            v_p_proj, v_f_proj, a_p_emb, a_f_emb, t_emb
        )

        # ===== Project future video back to latent dimension =====
        # Apply final AdaLN with time modulation
        v_f_out = self.final_ada_ln(v_f_out, t_emb)

        # Project to velocity: (B, T_f, H, W, d_model) -> (B, d_model, T_f, H, W)
        velocity = rearrange(v_f_out, 'b t h w c -> b c t h w')
        velocity = self.video_unproj(velocity)

        return velocity

    def forward_with_cfg(
        self,
        v_f_noisy: torch.Tensor,
        v_p: torch.Tensor,
        a_p: torch.Tensor,
        a_f: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """Forward pass with classifier-free guidance.

        Computes both conditional and unconditional predictions, then
        combines them using the CFG formula:
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

        Args:
            v_f_noisy: Noisy future latent X_t
            v_p: Past video latent (conditioning)
            a_p: Past actions (conditioning)
            a_f: Future actions
            t: Timesteps
            cfg_scale: Guidance scale (1.0 = no guidance)

        Returns:
            Guided velocity prediction
        """
        # Conditional prediction (full conditioning)
        v_cond = self.forward(v_f_noisy, v_p, a_p, a_f, t)

        if cfg_scale == 1.0:
            return v_cond

        # Unconditional prediction (zero conditioning)
        zeros_v_p = torch.zeros_like(v_p)
        zeros_a_p = torch.zeros_like(a_p)
        v_uncond = self.forward(v_f_noisy, zeros_v_p, zeros_a_p, a_f, t)

        # CFG: v = v_uncond + scale * (v_cond - v_uncond)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.video_pos_embed.numel()
            n_params -= self.action_pos_embed.numel()
        return n_params


def create_flow_hwm(config: Optional[FlowHWMConfig] = None) -> FlowHWM:
    """Create a FlowHWM model with the given configuration.

    Args:
        config: Model configuration. If None, uses default FlowHWMConfig.

    Returns:
        FlowHWM model
    """
    if config is None:
        config = FlowHWMConfig()
    return FlowHWM(config)
