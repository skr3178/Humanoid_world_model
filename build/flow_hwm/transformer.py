"""Four-stream transformer for Flow-HWM with time modulation.

Implements the transformer architecture described in method.md:

Architecture per block:
    Step 1: Modulate with time (AdaLN), QKV projection, concatenate past/future,
            apply RoPE (3D for video, 1D for action)
    Step 2: Joint attention over ALL tokens (video patches + actions) - no parameter sharing
    Step 3: Feedforward with time scaling

Residual connections:
    1. Before step 1 → after step 2 (joint attention + γ₀ rescaling)
    2. After step 2 → after step 3 (feedforward)

Key design: Each stream (v_p, v_f, a_p, a_f) has SEPARATE weights for:
    - Timestep modulation (AdaLN)
    - QKV projection
    - Output projection (per-stream, no sharing in Step 2)
    - Feedforward MLPs

Paper-aligned: Video tokens kept at patch resolution, joint attention over
T*H*W video tokens + T action tokens per stream.
γ₀ rescaling added after attention (paper: "followed by another timestep-dependent rescaling using γ₀").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional

from .config import FlowHWMConfig
from .ada_ln import AdaLN, FeedForwardScale
from .rope_3d import RoPE3D

# Import RoPE1D from masked_hwm (no modification needed)
import sys
sys.path.insert(0, "/media/skr/storage/robot_world/humanoid_wm/build")
from masked_hwm.rope import RoPE1D


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""

    def __init__(
        self,
        d_model: int,
        mlp_hidden: int,
        mlp_drop: float = 0.0,
        mlp_bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_hidden, bias=mlp_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, d_model, bias=mlp_bias)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class FlowTransformerBlock(nn.Module):
    """Four-stream transformer block with time modulation (paper-aligned).

    Per method.md architecture:
    - Step 1: Modulate with time (AdaLN), QKV projection, concat past/future, RoPE
    - Step 2: Joint attention over ALL tokens (video patches + actions)
    - Step 3: Feedforward with time scaling

    Each stream has separate weights for AdaLN, QKV, output projection, and feedforward.
    Joint attention integrates all streams at full spatial resolution.

    Args:
        config: FlowHWMConfig with model hyperparameters
    """

    def __init__(self, config: FlowHWMConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        num_heads = config.num_heads
        head_dim = d_model // num_heads

        # ===== Per-stream AdaLN (Step 1: modulate with time) =====
        self.ada_ln_vp = AdaLN(d_model)
        self.ada_ln_vf = AdaLN(d_model)
        self.ada_ln_ap = AdaLN(d_model)
        self.ada_ln_af = AdaLN(d_model)

        # ===== Per-stream QKV projections =====
        self.qkv_vp = nn.Linear(d_model, d_model * 3, bias=config.qkv_bias)
        self.qkv_vf = nn.Linear(d_model, d_model * 3, bias=config.qkv_bias)
        self.qkv_ap = nn.Linear(d_model, d_model * 3, bias=config.qkv_bias)
        self.qkv_af = nn.Linear(d_model, d_model * 3, bias=config.qkv_bias)

        # ===== RoPE: 3D for video, 1D for actions =====
        self.rope_3d = RoPE3D(
            head_dim=head_dim,
            max_t=config.num_past_clips + config.num_future_clips,
            max_h=config.latent_spatial,
            max_w=config.latent_spatial,
        )
        self.rope_1d = RoPE1D(
            head_dim=head_dim,
            max_seq_len=(config.num_past_clips + config.num_future_clips) * 2,  # Extra margin
        )

        # ===== Per-stream attention output projections (no sharing per method.md) =====
        self.attn_proj_vp = nn.Linear(d_model, d_model, bias=config.proj_bias)
        self.attn_proj_vf = nn.Linear(d_model, d_model, bias=config.proj_bias)
        self.attn_proj_ap = nn.Linear(d_model, d_model, bias=config.proj_bias)
        self.attn_proj_af = nn.Linear(d_model, d_model, bias=config.proj_bias)
        self.attn_drop = nn.Dropout(config.attn_drop)

        # ===== Per-stream γ₀ rescaling after attention (paper requirement) =====
        self.gamma0_vp = FeedForwardScale(d_model)
        self.gamma0_vf = FeedForwardScale(d_model)
        self.gamma0_ap = FeedForwardScale(d_model)
        self.gamma0_af = FeedForwardScale(d_model)

        # ===== Per-stream AdaLN for post-attention (before feedforward) =====
        self.ada_ln2_vp = AdaLN(d_model)
        self.ada_ln2_vf = AdaLN(d_model)
        self.ada_ln2_ap = AdaLN(d_model)
        self.ada_ln2_af = AdaLN(d_model)

        # ===== Per-stream feedforward MLPs =====
        mlp_hidden = config.mlp_hidden
        mlp_drop = config.mlp_drop
        self.mlp_vp = MLP(d_model, mlp_hidden, mlp_drop)
        self.mlp_vf = MLP(d_model, mlp_hidden, mlp_drop)
        self.mlp_ap = MLP(d_model, mlp_hidden, mlp_drop)
        self.mlp_af = MLP(d_model, mlp_hidden, mlp_drop)

        # ===== Per-stream time scaling for feedforward output (γ₁) =====
        self.ff_scale_vp = FeedForwardScale(d_model)
        self.ff_scale_vf = FeedForwardScale(d_model)
        self.ff_scale_ap = FeedForwardScale(d_model)
        self.ff_scale_af = FeedForwardScale(d_model)

        # Scale factor for attention
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        v_p: torch.Tensor,
        v_f: torch.Tensor,
        a_p: torch.Tensor,
        a_f: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through transformer block.

        Args:
            v_p: Past video tokens (B, T_p, H, W, d_model)
            v_f: Future video tokens (B, T_f, H, W, d_model)
            a_p: Past action tokens (B, T_p, d_model) - at clip level
            a_f: Future action tokens (B, T_f, d_model) - at clip level
            t_emb: Time embedding (B, d_model)

        Returns:
            Updated tokens for each stream
        """
        B = v_p.shape[0]
        T_p, T_f = v_p.shape[1], v_f.shape[1]
        H, W = v_p.shape[2], v_p.shape[3]
        T_total = T_p + T_f
        S = H * W  # Spatial tokens per clip

        # ===== Save residuals for first skip connection =====
        v_p_res1, v_f_res1 = v_p, v_f
        a_p_res1, a_f_res1 = a_p, a_f

        # ===== Step 1a: Apply AdaLN (modulate with time) =====
        v_p = self.ada_ln_vp(v_p, t_emb)
        v_f = self.ada_ln_vf(v_f, t_emb)
        a_p = self.ada_ln_ap(a_p, t_emb)
        a_f = self.ada_ln_af(a_f, t_emb)

        # ===== Step 1b: QKV projection (per-stream) =====
        qkv_vp = self.qkv_vp(v_p)
        qkv_vf = self.qkv_vf(v_f)
        qkv_ap = self.qkv_ap(a_p)
        qkv_af = self.qkv_af(a_f)

        # Split into Q, K, V
        q_vp, k_vp, v_vp = qkv_vp.chunk(3, dim=-1)
        q_vf, k_vf, v_vf = qkv_vf.chunk(3, dim=-1)
        q_ap, k_ap, v_ap = qkv_ap.chunk(3, dim=-1)
        q_af, k_af, v_af = qkv_af.chunk(3, dim=-1)

        # ===== Step 1c: Concatenate past and future =====
        q_video = torch.cat([q_vp, q_vf], dim=1)
        k_video = torch.cat([k_vp, k_vf], dim=1)
        v_video = torch.cat([v_vp, v_vf], dim=1)

        q_action = torch.cat([q_ap, q_af], dim=1)
        k_action = torch.cat([k_ap, k_af], dim=1)
        v_action = torch.cat([v_ap, v_af], dim=1)

        # ===== Step 1d: Apply RoPE =====
        q_video = q_video.view(B, T_total, H, W, self.num_heads, self.head_dim)
        k_video = k_video.view(B, T_total, H, W, self.num_heads, self.head_dim)
        v_video = v_video.view(B, T_total, H, W, self.num_heads, self.head_dim)

        q_video = self.rope_3d(q_video)
        k_video = self.rope_3d(k_video)

        q_action = q_action.view(B, T_total, self.num_heads, self.head_dim)
        k_action = k_action.view(B, T_total, self.num_heads, self.head_dim)
        v_action = v_action.view(B, T_total, self.num_heads, self.head_dim)

        q_action = self.rope_1d(q_action)
        k_action = self.rope_1d(k_action)

        # ===== Step 2: Joint attention over ALL tokens =====
        N_video = T_total * S

        q_video_flat = q_video.view(B, N_video, self.num_heads, self.head_dim)
        k_video_flat = k_video.view(B, N_video, self.num_heads, self.head_dim)
        v_video_flat = v_video.view(B, N_video, self.num_heads, self.head_dim)

        q_all = torch.cat([q_video_flat, q_action], dim=1)
        k_all = torch.cat([k_video_flat, k_action], dim=1)
        v_all = torch.cat([v_video_flat, v_action], dim=1)

        q_all = q_all.transpose(1, 2)
        k_all = k_all.transpose(1, 2)
        v_all = v_all.transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            out_all = F.scaled_dot_product_attention(
                q_all, k_all, v_all,
                dropout_p=self.config.attn_drop if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn = (q_all @ k_all.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out_all = attn @ v_all

        out_all = out_all.transpose(1, 2).reshape(B, -1, self.config.d_model)

        out_video = out_all[:, :N_video]
        out_action = out_all[:, N_video:]

        out_video = out_video.view(B, T_total, H, W, self.config.d_model)

        out_vp = out_video[:, :T_p]
        out_vf = out_video[:, T_p:]
        out_ap = out_action[:, :T_p]
        out_af = out_action[:, T_p:]

        # ===== Per-stream output projections =====
        out_vp = self.attn_proj_vp(out_vp)
        out_vf = self.attn_proj_vf(out_vf)
        out_ap = self.attn_proj_ap(out_ap)
        out_af = self.attn_proj_af(out_af)

        # ===== NEW: Per-stream γ₀ rescaling after attention (paper requirement) =====
        out_vp = self.gamma0_vp(out_vp, t_emb)
        out_vf = self.gamma0_vf(out_vf, t_emb)
        out_ap = self.gamma0_ap(out_ap, t_emb)
        out_af = self.gamma0_af(out_af, t_emb)

        # ===== Residual 1: Add pre-step-1 residuals after attention + γ₀ =====
        v_p = v_p_res1 + out_vp
        v_f = v_f_res1 + out_vf
        a_p = a_p_res1 + out_ap
        a_f = a_f_res1 + out_af

        # ===== Save residuals for second skip connection =====
        v_p_res2, v_f_res2 = v_p, v_f
        a_p_res2, a_f_res2 = a_p, a_f

        # ===== Step 3: Feedforward with time scaling =====
        v_p = self.ada_ln2_vp(v_p, t_emb)
        v_f = self.ada_ln2_vf(v_f, t_emb)
        a_p = self.ada_ln2_ap(a_p, t_emb)
        a_f = self.ada_ln2_af(a_f, t_emb)

        v_p = self.mlp_vp(v_p)
        v_f = self.mlp_vf(v_f)
        a_p = self.mlp_ap(a_p)
        a_f = self.mlp_af(a_f)

        # Apply time scaling (γ₁)
        v_p = self.ff_scale_vp(v_p, t_emb)
        v_f = self.ff_scale_vf(v_f, t_emb)
        a_p = self.ff_scale_ap(a_p, t_emb)
        a_f = self.ff_scale_af(a_f, t_emb)

        # ===== Residual 2 =====
        v_p = v_p_res2 + v_p
        v_f = v_f_res2 + v_f
        a_p = a_p_res2 + a_p
        a_f = a_f_res2 + a_f

        return v_p, v_f, a_p, a_f


class FlowTransformerBlockShared(nn.Module):
    """Transformer block with parameter sharing for deeper layers.

    Uses shared weights for QKV and output projections across streams
    to reduce memory when VRAM is tight. Paper shows minimal quality drop.

    Args:
        config: FlowHWMConfig with model hyperparameters
    """

    def __init__(self, config: FlowHWMConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        num_heads = config.num_heads
        head_dim = d_model // num_heads

        # ===== Per-stream AdaLN (still separate for time modulation) =====
        self.ada_ln_vp = AdaLN(d_model)
        self.ada_ln_vf = AdaLN(d_model)
        self.ada_ln_ap = AdaLN(d_model)
        self.ada_ln_af = AdaLN(d_model)

        # ===== SHARED QKV projection (full sharing variant) =====
        self.qkv_shared = nn.Linear(d_model, d_model * 3, bias=config.qkv_bias)

        # ===== RoPE: 3D for video, 1D for actions =====
        self.rope_3d = RoPE3D(
            head_dim=head_dim,
            max_t=config.num_past_clips + config.num_future_clips,
            max_h=config.latent_spatial,
            max_w=config.latent_spatial,
        )
        self.rope_1d = RoPE1D(
            head_dim=head_dim,
            max_seq_len=(config.num_past_clips + config.num_future_clips) * 2,
        )

        # ===== SHARED attention output projection =====
        self.attn_proj_shared = nn.Linear(d_model, d_model, bias=config.proj_bias)
        self.attn_drop = nn.Dropout(config.attn_drop)

        # ===== SHARED γ₀ rescaling after attention (for consistency with sharing) =====
        self.gamma0_shared = FeedForwardScale(d_model)

        # ===== Per-stream AdaLN for post-attention =====
        self.ada_ln2_vp = AdaLN(d_model)
        self.ada_ln2_vf = AdaLN(d_model)
        self.ada_ln2_ap = AdaLN(d_model)
        self.ada_ln2_af = AdaLN(d_model)

        # ===== SHARED feedforward MLP =====
        self.mlp_shared = MLP(d_model, config.mlp_hidden, config.mlp_drop)

        # ===== Per-stream time scaling (γ₁) =====
        self.ff_scale_vp = FeedForwardScale(d_model)
        self.ff_scale_vf = FeedForwardScale(d_model)
        self.ff_scale_ap = FeedForwardScale(d_model)
        self.ff_scale_af = FeedForwardScale(d_model)

        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        v_p: torch.Tensor,
        v_f: torch.Tensor,
        a_p: torch.Tensor,
        a_f: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with shared weights."""
        B = v_p.shape[0]
        T_p, T_f = v_p.shape[1], v_f.shape[1]
        H, W = v_p.shape[2], v_p.shape[3]
        T_total = T_p + T_f
        S = H * W

        # Save residuals
        v_p_res1, v_f_res1 = v_p, v_f
        a_p_res1, a_f_res1 = a_p, a_f

        # Step 1a: AdaLN (per-stream)
        v_p = self.ada_ln_vp(v_p, t_emb)
        v_f = self.ada_ln_vf(v_f, t_emb)
        a_p = self.ada_ln_ap(a_p, t_emb)
        a_f = self.ada_ln_af(a_f, t_emb)

        # Step 1b: Shared QKV projection
        qkv_vp = self.qkv_shared(v_p)
        qkv_vf = self.qkv_shared(v_f)
        qkv_ap = self.qkv_shared(a_p)
        qkv_af = self.qkv_shared(a_f)

        q_vp, k_vp, v_vp = qkv_vp.chunk(3, dim=-1)
        q_vf, k_vf, v_vf = qkv_vf.chunk(3, dim=-1)
        q_ap, k_ap, v_ap = qkv_ap.chunk(3, dim=-1)
        q_af, k_af, v_af = qkv_af.chunk(3, dim=-1)

        # Concatenate past and future
        q_video = torch.cat([q_vp, q_vf], dim=1)
        k_video = torch.cat([k_vp, k_vf], dim=1)
        v_video = torch.cat([v_vp, v_vf], dim=1)
        q_action = torch.cat([q_ap, q_af], dim=1)
        k_action = torch.cat([k_ap, k_af], dim=1)
        v_action = torch.cat([v_ap, v_af], dim=1)

        # Apply RoPE
        q_video = q_video.view(B, T_total, H, W, self.num_heads, self.head_dim)
        k_video = k_video.view(B, T_total, H, W, self.num_heads, self.head_dim)
        v_video = v_video.view(B, T_total, H, W, self.num_heads, self.head_dim)
        q_video = self.rope_3d(q_video)
        k_video = self.rope_3d(k_video)

        q_action = q_action.view(B, T_total, self.num_heads, self.head_dim)
        k_action = k_action.view(B, T_total, self.num_heads, self.head_dim)
        v_action = v_action.view(B, T_total, self.num_heads, self.head_dim)
        q_action = self.rope_1d(q_action)
        k_action = self.rope_1d(k_action)

        # Joint attention
        N_video = T_total * S
        q_video_flat = q_video.view(B, N_video, self.num_heads, self.head_dim)
        k_video_flat = k_video.view(B, N_video, self.num_heads, self.head_dim)
        v_video_flat = v_video.view(B, N_video, self.num_heads, self.head_dim)

        q_all = torch.cat([q_video_flat, q_action], dim=1).transpose(1, 2)
        k_all = torch.cat([k_video_flat, k_action], dim=1).transpose(1, 2)
        v_all = torch.cat([v_video_flat, v_action], dim=1).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            out_all = F.scaled_dot_product_attention(
                q_all, k_all, v_all,
                dropout_p=self.config.attn_drop if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn = (q_all @ k_all.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out_all = attn @ v_all

        out_all = out_all.transpose(1, 2).reshape(B, -1, self.config.d_model)

        # Split and reshape
        out_video = out_all[:, :N_video].view(B, T_total, H, W, self.config.d_model)
        out_action = out_all[:, N_video:]

        out_vp = out_video[:, :T_p]
        out_vf = out_video[:, T_p:]
        out_ap = out_action[:, :T_p]
        out_af = out_action[:, T_p:]

        # Shared output projection
        out_vp = self.attn_proj_shared(out_vp)
        out_vf = self.attn_proj_shared(out_vf)
        out_ap = self.attn_proj_shared(out_ap)
        out_af = self.attn_proj_shared(out_af)

        # ===== NEW: Shared γ₀ rescaling after attention =====
        out_vp = self.gamma0_shared(out_vp, t_emb)
        out_vf = self.gamma0_shared(out_vf, t_emb)
        out_ap = self.gamma0_shared(out_ap, t_emb)
        out_af = self.gamma0_shared(out_af, t_emb)

        # Residual 1
        v_p = v_p_res1 + out_vp
        v_f = v_f_res1 + out_vf
        a_p = a_p_res1 + out_ap
        a_f = a_f_res1 + out_af

        v_p_res2, v_f_res2 = v_p, v_f
        a_p_res2, a_f_res2 = a_p, a_f

        # Step 3: Feedforward
        v_p = self.ada_ln2_vp(v_p, t_emb)
        v_f = self.ada_ln2_vf(v_f, t_emb)
        a_p = self.ada_ln2_ap(a_p, t_emb)
        a_f = self.ada_ln2_af(a_f, t_emb)

        v_p = self.mlp_shared(v_p)
        v_f = self.mlp_shared(v_f)
        a_p = self.mlp_shared(a_p)
        a_f = self.mlp_shared(a_f)

        v_p = self.ff_scale_vp(v_p, t_emb)
        v_f = self.ff_scale_vf(v_f, t_emb)
        a_p = self.ff_scale_ap(a_p, t_emb)
        a_f = self.ff_scale_af(a_f, t_emb)

        v_p = v_p_res2 + v_p
        v_f = v_f_res2 + v_f
        a_p = a_p_res2 + a_p
        a_f = a_f_res2 + a_f

        return v_p, v_f, a_p, a_f


class FlowTransformer(nn.Module):
    """Full Flow-HWM transformer with stacked blocks.

    Supports optional parameter sharing in deeper layers to reduce VRAM.

    Args:
        config: FlowHWMConfig with model hyperparameters
        share_after_layer: Layer index after which to use shared blocks (None = no sharing)
    """

    def __init__(self, config: FlowHWMConfig, share_after_layer: Optional[int] = None):
        super().__init__()
        self.config = config
        self.share_after_layer = share_after_layer

        # Stack of transformer blocks
        blocks = []
        for i in range(config.num_layers):
            if share_after_layer is not None and i >= share_after_layer:
                blocks.append(FlowTransformerBlockShared(config))
            else:
                blocks.append(FlowTransformerBlock(config))
        self.blocks = nn.ModuleList(blocks)

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        v_p: torch.Tensor,
        v_f: torch.Tensor,
        a_p: torch.Tensor,
        a_f: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all transformer blocks.

        Args:
            v_p: Past video tokens (B, T_p, H, W, d_model)
            v_f: Future video tokens (B, T_f, H, W, d_model)
            a_p: Past action tokens (B, T_p, d_model)
            a_f: Future action tokens (B, T_f, d_model)
            t_emb: Time embedding (B, d_model)

        Returns:
            Updated tokens for each stream
        """
        for block in self.blocks:
            v_p, v_f, a_p, a_f = block(v_p, v_f, a_p, a_f, t_emb)

        # Apply final layer norm
        v_p = self.final_norm(v_p)
        v_f = self.final_norm(v_f)
        a_p = self.final_norm(a_p)
        a_f = self.final_norm(a_f)

        return v_p, v_f, a_p, a_f