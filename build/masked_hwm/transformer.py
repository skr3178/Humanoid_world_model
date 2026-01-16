"""Transformer with parameter sharing for Masked-HWM.

CORRECTED IMPLEMENTATION addressing key issues:
1. NO causal masking: Bidirectional attention for MaskGIT-style non-autoregressive generation
2. NO spatial pooling: Each spatial patch treated independently in temporal attention
3. Actions repeated per spatial patch: Enables spatial-specific action conditioning

Implements factorized attention:
- Spatial attention: Applied only to video tokens (per-frame) with 2D RoPE
- Temporal attention: Joint across all 4 streams (v_p, v_f, a_p, a_f) with 1D RoPE
  - Bidirectional (all tokens attend to all tokens)
  - Each spatial patch processed independently

Memory optimizations:
- Gradient checkpointing: Recompute activations during backward pass
- Flash Attention / xformers: Memory-efficient attention implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from typing import Optional, Tuple

from .config import MaskedHWMConfig
from .rope import RoPE1D, RoPE2D

# Try to import memory-efficient attention backends
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    from xformers.ops import memory_efficient_attention
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


class MLP(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(self, d_model: int, mlp_hidden: int, mlp_drop: float = 0.0, mlp_bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_hidden, bias=mlp_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, d_model, bias=mlp_bias)
        self.drop = nn.Dropout(mlp_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class SpatialAttention(nn.Module):
    """Spatial attention for video tokens (applied per frame) with 2D RoPE.

    Attends across spatial positions within each frame.
    Uses 2D Rotary Position Embeddings as described in the paper.

    Memory optimization: Uses Flash Attention or xformers when available.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        spatial_size: int = 16,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        use_rope: bool = True,
        use_flash_attn: bool = True,  # Enable memory-efficient attention
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn and (FLASH_ATTN_AVAILABLE or XFORMERS_AVAILABLE)

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_p = attn_drop

        # 2D RoPE for spatial positions
        if use_rope:
            self.rope_2d = RoPE2D(
                head_dim=self.head_dim,
                max_h=spatial_size,
                max_w=spatial_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention with 2D RoPE.

        Args:
            x: Input tensor (B, T, H, W, d_model)

        Returns:
            Output tensor (B, T, H, W, d_model)
        """
        B, T, H, W, C = x.shape

        # Reshape for per-frame attention: (B*T, H*W, C)
        x_flat = rearrange(x, 'b t h w c -> (b t) (h w) c')

        # Compute QKV
        qkv = self.qkv(x_flat)  # (B*T, H*W, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B*T, H*W, num_heads, head_dim) for flash attn
        # or (B*T, num_heads, H*W, head_dim) for standard attn
        BT = B * T
        seq_len = H * W

        q = q.view(BT, seq_len, self.num_heads, self.head_dim)
        k = k.view(BT, seq_len, self.num_heads, self.head_dim)
        v = v.view(BT, seq_len, self.num_heads, self.head_dim)

        # Apply 2D RoPE to queries and keys
        if self.use_rope:
            # Reshape to (B*T, H, W, num_heads, head_dim) for 2D RoPE
            q_spatial = q.view(BT, H, W, self.num_heads, self.head_dim)
            k_spatial = k.view(BT, H, W, self.num_heads, self.head_dim)

            q_spatial = self.rope_2d(q_spatial)
            k_spatial = self.rope_2d(k_spatial)

            # Reshape back
            q = q_spatial.view(BT, seq_len, self.num_heads, self.head_dim)
            k = k_spatial.view(BT, seq_len, self.num_heads, self.head_dim)

        # Compute attention using memory-efficient implementation if available
        if self.use_flash_attn and FLASH_ATTN_AVAILABLE:
            # Flash Attention expects (B, seq_len, num_heads, head_dim)
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                softmax_scale=self.scale,
            )  # (B*T, H*W, num_heads, head_dim)
            out = out.reshape(BT, seq_len, C)
        elif self.use_flash_attn and XFORMERS_AVAILABLE:
            # xformers expects (B, seq_len, num_heads, head_dim)
            out = memory_efficient_attention(
                q, k, v,
                scale=self.scale,
            )  # (B*T, H*W, num_heads, head_dim)
            out = out.reshape(BT, seq_len, C)
        else:
            # Standard attention fallback
            q = q.transpose(1, 2)  # (B*T, num_heads, H*W, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*T, num_heads, H*W, H*W)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = attn @ v  # (B*T, num_heads, H*W, head_dim)
            out = out.transpose(1, 2).reshape(BT, seq_len, C)  # (B*T, H*W, C)

        # Project output
        out = self.proj(out)

        # Reshape back
        out = rearrange(out, '(b t) (h w) c -> b t h w c', b=B, t=T, h=H, w=W)

        return out


class TemporalAttention(nn.Module):
    """Temporal attention across all streams (joint attention) with 1D RoPE.

    CORRECTED IMPLEMENTATION:
    - NO causal masking (bidirectional non-autoregressive attention for MaskGIT-style generation)
    - NO spatial pooling (each spatial patch treated independently)
    - Actions repeated for each spatial patch to enable spatial-specific conditioning

    Concatenates all streams and applies attention across temporal dimension.
    Uses 1D Rotary Position Embeddings as described in the paper.

    Memory optimization: Uses Flash Attention or xformers when available.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 64,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        use_rope: bool = True,
        use_stream_type_emb: bool = False,
        use_flash_attn: bool = True,  # Enable memory-efficient attention
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.use_stream_type_emb = use_stream_type_emb
        self.use_flash_attn = use_flash_attn and (FLASH_ATTN_AVAILABLE or XFORMERS_AVAILABLE)

        # Single shared QKV projection for all streams
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_p = attn_drop

        # Optional stream-type embeddings to make stream identity explicit
        if use_stream_type_emb:
            self.stream_type_emb = nn.Parameter(torch.zeros(4, d_model))
            nn.init.normal_(self.stream_type_emb, std=0.02)

        # 1D RoPE for temporal positions
        if use_rope:
            self.rope_1d = RoPE1D(
                head_dim=self.head_dim,
                max_seq_len=max_seq_len,
            )

    def forward(
        self,
        v_p: torch.Tensor,
        v_f: torch.Tensor,
        a_p: torch.Tensor,
        a_f: torch.Tensor,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply joint temporal attention across all streams with 1D RoPE.

        Args:
            v_p: Past video tokens (B, T_p, S, d_model) where S = H*W
            v_f: Future video tokens (B, T_f, S, d_model)
            a_p: Past action tokens (B, T_p, d_model)
            a_f: Future action tokens (B, T_f, d_model)
            causal: Whether to apply causal masking (default False for MaskGIT-style)

        Returns:
            Updated tokens for each stream
        """
        B = v_p.shape[0]
        T_p, T_f = v_p.shape[1], v_f.shape[1]
        S = v_p.shape[2]  # Spatial dimension (H*W)

        # Reshape video tokens: (B, T, S, d) -> (B*S, T, d)
        v_p_reshaped = rearrange(v_p, 'b t s d -> (b s) t d')
        v_f_reshaped = rearrange(v_f, 'b t s d -> (b s) t d')

        # Repeat actions for each spatial patch: (B, T, d) -> (B*S, T, d)
        a_p_reshaped = repeat(a_p, 'b t d -> (b s) t d', s=S)
        a_f_reshaped = repeat(a_f, 'b t d -> (b s) t d', s=S)

        # Concatenate all streams: [v_p, v_f, a_p, a_f]
        all_tokens = torch.cat([v_p_reshaped, v_f_reshaped, a_p_reshaped, a_f_reshaped], dim=1)
        BS = B * S
        total_T = all_tokens.shape[1]

        # Optionally add stream-type embeddings
        if self.use_stream_type_emb:
            stream_indices = torch.cat([
                torch.zeros(T_p, dtype=torch.long, device=all_tokens.device),
                torch.ones(T_f, dtype=torch.long, device=all_tokens.device),
                torch.full((T_p,), 2, dtype=torch.long, device=all_tokens.device),
                torch.full((T_f,), 3, dtype=torch.long, device=all_tokens.device),
            ])
            stream_emb = self.stream_type_emb[stream_indices]
            all_tokens = all_tokens + stream_emb.unsqueeze(0)

        # Compute QKV
        qkv = self.qkv(all_tokens)  # (B*S, total_T, 3*d)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape: (B*S, total_T, num_heads, head_dim)
        q = q.view(BS, total_T, self.num_heads, self.head_dim)
        k = k.view(BS, total_T, self.num_heads, self.head_dim)
        v = v.view(BS, total_T, self.num_heads, self.head_dim)

        # Apply 1D RoPE to queries and keys
        if self.use_rope:
            q = self.rope_1d(q)
            k = self.rope_1d(k)

        # Compute attention using memory-efficient implementation if available
        if self.use_flash_attn and FLASH_ATTN_AVAILABLE and not causal:
            # Flash Attention (bidirectional)
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            out = out.reshape(BS, total_T, self.d_model)
        elif self.use_flash_attn and FLASH_ATTN_AVAILABLE and causal:
            # Flash Attention (causal)
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=True,
            )
            out = out.reshape(BS, total_T, self.d_model)
        elif self.use_flash_attn and XFORMERS_AVAILABLE:
            # xformers memory-efficient attention
            out = memory_efficient_attention(
                q, k, v,
                scale=self.scale,
            )
            out = out.reshape(BS, total_T, self.d_model)
        else:
            # Standard attention fallback
            q = q.transpose(1, 2)  # (B*S, num_heads, total_T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * self.scale

            if causal:
                causal_mask = torch.triu(
                    torch.ones(total_T, total_T, device=attn.device, dtype=torch.bool),
                    diagonal=1
                )
                attn = attn.masked_fill(causal_mask, float('-inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = attn @ v
            out = out.transpose(1, 2).reshape(BS, total_T, self.d_model)
        
        # Project output
        out = self.proj(out)
        
        # Split back into streams
        v_p_out = out[:, :T_p]  # (B*S, T_p, d)
        v_f_out = out[:, T_p:T_p + T_f]  # (B*S, T_f, d)
        a_p_out = out[:, T_p + T_f:T_p + T_f + T_p]  # (B*S, T_p, d)
        a_f_out = out[:, T_p + T_f + T_p:]  # (B*S, T_f, d)
        
        # Reshape video back to original shape: (B*S, T, d) -> (B, T, S, d)
        v_p_out = rearrange(v_p_out, '(b s) t d -> b t s d', b=B, s=S)
        v_f_out = rearrange(v_f_out, '(b s) t d -> b t s d', b=B, s=S)
        
        # For actions, take mean across spatial dimension (since we repeated them)
        # This aggregates the spatial-specific conditioning back to a single action representation
        a_p_out = rearrange(a_p_out, '(b s) t d -> b s t d', b=B, s=S).mean(dim=1)  # (B, T_p, d)
        a_f_out = rearrange(a_f_out, '(b s) t d -> b s t d', b=B, s=S).mean(dim=1)  # (B, T_f, d)
        
        return v_p_out, v_f_out, a_p_out, a_f_out


class SharedTransformerBlock(nn.Module):
    """Transformer block with factorized attention and parameter sharing.

    Architecture per block (matches Figure 2 diagram):
    1. Spatial attention (video only) with 2D RoPE
    2. MLP (video only) - intermediate feedforward after spatial attention
    3. Joint temporal attention (all streams) with 1D RoPE
    4. MLP (all streams) - distinct weights per stream

    Parameter sharing:
    - layers < shared_layers_start: No sharing (separate params per stream)
    - layers >= shared_layers_start: Modality sharing for spatial attention
        - Video streams (v_p, v_f) share spatial attention
    - Final MLPs always use distinct weights per stream (as per diagram)

    Memory optimization: Supports Flash Attention / xformers when available.
    """

    def __init__(
        self,
        layer_idx: int,
        config: MaskedHWMConfig,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.use_sharing = layer_idx >= config.shared_layers_start

        # Get flash attention setting from config (default True for auto-detection)
        use_flash_attn = getattr(config, 'use_flash_attn', True)

        # Layer norms
        self.spatial_norm = nn.LayerNorm(config.d_model)
        self.spatial_mlp_norm = nn.LayerNorm(config.d_model)
        self.temporal_norm = nn.LayerNorm(config.d_model)
        self.mlp_norm = nn.LayerNorm(config.d_model)

        # Spatial attention (video only) with 2D RoPE
        if self.use_sharing:
            self.spatial_attn = SpatialAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                spatial_size=config.spatial_size,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                attn_drop=config.attn_drop,
                use_rope=config.use_rope,
                use_flash_attn=use_flash_attn,
            )
        else:
            self.spatial_attn_vp = SpatialAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                spatial_size=config.spatial_size,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                attn_drop=config.attn_drop,
                use_rope=config.use_rope,
                use_flash_attn=use_flash_attn,
            )
            self.spatial_attn_vf = SpatialAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                spatial_size=config.spatial_size,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                attn_drop=config.attn_drop,
                use_rope=config.use_rope,
                use_flash_attn=use_flash_attn,
            )

        # Intermediate MLP for video streams
        if self.use_sharing:
            self.spatial_mlp = MLP(
                d_model=config.d_model,
                mlp_hidden=config.mlp_hidden,
                mlp_drop=config.mlp_drop,
            )
        else:
            self.spatial_mlp_vp = MLP(
                d_model=config.d_model,
                mlp_hidden=config.mlp_hidden,
                mlp_drop=config.mlp_drop,
            )
            self.spatial_mlp_vf = MLP(
                d_model=config.d_model,
                mlp_hidden=config.mlp_hidden,
                mlp_drop=config.mlp_drop,
            )

        # Temporal attention (joint across all streams) with 1D RoPE
        max_temporal_len = 2 * (config.num_past_frames + config.num_future_frames)
        self.temporal_attn = TemporalAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            max_seq_len=max_temporal_len,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            attn_drop=config.attn_drop,
            use_rope=config.use_rope,
            use_stream_type_emb=config.use_stream_type_emb,
            use_flash_attn=use_flash_attn,
        )

        # Final MLP - distinct weights per stream (as per diagram caption:
        # "Each stream uses distinct MLP weights in the feedforward stage")
        self.mlp_vp = MLP(
            d_model=config.d_model,
            mlp_hidden=config.mlp_hidden,
            mlp_drop=config.mlp_drop,
        )
        self.mlp_vf = MLP(
            d_model=config.d_model,
            mlp_hidden=config.mlp_hidden,
            mlp_drop=config.mlp_drop,
        )
        self.mlp_ap = MLP(
            d_model=config.d_model,
            mlp_hidden=config.mlp_hidden,
            mlp_drop=config.mlp_drop,
        )
        self.mlp_af = MLP(
            d_model=config.d_model,
            mlp_hidden=config.mlp_hidden,
            mlp_drop=config.mlp_drop,
        )
    
    def forward(
        self,
        v_p: torch.Tensor,
        v_f: torch.Tensor,
        a_p: torch.Tensor,
        a_f: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through transformer block.

        Args:
            v_p: Past video tokens (B, T_p, H, W, d_model)
            v_f: Future video tokens (B, T_f, H, W, d_model)
            a_p: Past action tokens (B, T_p, d_model)
            a_f: Future action tokens (B, T_f, d_model)

        Returns:
            Updated tokens for each stream
        """
        B = v_p.shape[0]
        T_p, T_f = v_p.shape[1], v_f.shape[1]
        H, W = v_p.shape[2], v_p.shape[3]

        # 1. Spatial attention (video only) with 2D RoPE
        if self.use_sharing:
            v_p = v_p + self.spatial_attn(self.spatial_norm(v_p))
            v_f = v_f + self.spatial_attn(self.spatial_norm(v_f))
        else:
            v_p = v_p + self.spatial_attn_vp(self.spatial_norm(v_p))
            v_f = v_f + self.spatial_attn_vf(self.spatial_norm(v_f))

        # 2. Intermediate MLP (video only) - between spatial and temporal attention
        if self.use_sharing:
            v_p = v_p + self.spatial_mlp(self.spatial_mlp_norm(v_p))
            v_f = v_f + self.spatial_mlp(self.spatial_mlp_norm(v_f))
        else:
            v_p = v_p + self.spatial_mlp_vp(self.spatial_mlp_norm(v_p))
            v_f = v_f + self.spatial_mlp_vf(self.spatial_mlp_norm(v_f))

        # 3. Joint temporal attention with 1D RoPE
        # Flatten spatial dimensions for video: (B, T, H, W, d) -> (B, T, H*W, d)
        v_p_flat = rearrange(v_p, 'b t h w d -> b t (h w) d')
        v_f_flat = rearrange(v_f, 'b t h w d -> b t (h w) d')

        # Apply temporal attention
        v_p_norm = self.temporal_norm(v_p_flat)
        v_f_norm = self.temporal_norm(v_f_flat)
        a_p_norm = self.temporal_norm(a_p)
        a_f_norm = self.temporal_norm(a_f)

        v_p_temp, v_f_temp, a_p_temp, a_f_temp = self.temporal_attn(
            v_p_norm, v_f_norm, a_p_norm, a_f_norm, causal=False  # CORRECTED: Bidirectional attention for MaskGIT-style
        )

        # Add residuals
        v_p_flat = v_p_flat + v_p_temp
        v_f_flat = v_f_flat + v_f_temp
        a_p = a_p + a_p_temp
        a_f = a_f + a_f_temp

        # Reshape video back to spatial
        v_p = rearrange(v_p_flat, 'b t (h w) d -> b t h w d', h=H, w=W)
        v_f = rearrange(v_f_flat, 'b t (h w) d -> b t h w d', h=H, w=W)

        # 4. Final MLP - distinct weights per stream
        v_p = v_p + self.mlp_vp(self.mlp_norm(v_p))
        v_f = v_f + self.mlp_vf(self.mlp_norm(v_f))
        a_p = a_p + self.mlp_ap(self.mlp_norm(a_p))
        a_f = a_f + self.mlp_af(self.mlp_norm(a_f))

        return v_p, v_f, a_p, a_f


class SharedTransformer(nn.Module):
    """Transformer with parameter sharing for 4-stream inputs.

    Implements the Masked-HWM architecture with:
    - Factorized attention: Spatial (video only) + Temporal (joint)
    - Parameter sharing: First 4 layers unshared, remaining layers shared

    Memory optimization:
    - Gradient checkpointing: Recomputes activations during backward pass
      to reduce memory usage by ~50% at the cost of ~20% extra compute.
    """

    def __init__(self, config: MaskedHWMConfig):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)

        self.layers = nn.ModuleList([
            SharedTransformerBlock(i, config)
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.use_gradient_checkpointing = False

    def _checkpoint_forward(self, layer, v_p, v_f, a_p, a_f):
        """Wrapper for checkpointing that handles multiple outputs."""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        return checkpoint(
            create_custom_forward(layer),
            v_p, v_f, a_p, a_f,
            use_reentrant=False,  # Recommended for PyTorch 2.0+
        )

    def forward(
        self,
        v_p: torch.Tensor,
        v_f: torch.Tensor,
        a_p: torch.Tensor,
        a_f: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through transformer.

        Args:
            v_p: Past video tokens (B, T_p, H, W, d_model)
            v_f: Future video tokens (B, T_f, H, W, d_model)
            a_p: Past action tokens (B, T_p, d_model)
            a_f: Future action tokens (B, T_f, d_model)

        Returns:
            Updated tokens for each stream
        """
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                v_p, v_f, a_p, a_f = self._checkpoint_forward(layer, v_p, v_f, a_p, a_f)
            else:
                v_p, v_f, a_p, a_f = layer(v_p, v_f, a_p, a_f)

        # Final layer norm
        v_p = self.final_norm(v_p)
        v_f = self.final_norm(v_f)
        a_p = self.final_norm(a_p)
        a_f = self.final_norm(a_f)

        return v_p, v_f, a_p, a_f


# Utility function to print memory optimization status
def print_memory_optimization_status():
    """Print the status of available memory optimizations."""
    print("=" * 50)
    print("Memory Optimization Status:")
    print("=" * 50)
    print(f"  Flash Attention:  {'Available' if FLASH_ATTN_AVAILABLE else 'Not installed (pip install flash-attn)'}")
    print(f"  xformers:         {'Available' if XFORMERS_AVAILABLE else 'Not installed (pip install xformers)'}")
    print(f"  Gradient Ckpt:    Always available (PyTorch built-in)")
    print("=" * 50)
