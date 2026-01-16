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
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional, Tuple

from .config import MaskedHWMConfig
from .rope import RoPE1D, RoPE2D


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
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
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
        
        # Reshape for multi-head attention: (B*T, num_heads, H*W, head_dim)
        q = q.view(-1, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply 2D RoPE to queries and keys
        if self.use_rope:
            # Reshape to (B*T, H, W, num_heads, head_dim) for 2D RoPE
            q_spatial = q.transpose(1, 2).view(-1, H, W, self.num_heads, self.head_dim)
            k_spatial = k.transpose(1, 2).view(-1, H, W, self.num_heads, self.head_dim)
            
            q_spatial = self.rope_2d(q_spatial)
            k_spatial = self.rope_2d(k_spatial)
            
            # Reshape back to (B*T, num_heads, H*W, head_dim)
            q = q_spatial.view(-1, H * W, self.num_heads, self.head_dim).transpose(1, 2)
            k = k_spatial.view(-1, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*T, num_heads, H*W, H*W)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = attn @ v  # (B*T, num_heads, H*W, head_dim)
        out = out.transpose(1, 2).reshape(-1, H * W, C)  # (B*T, H*W, C)
        
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
    
    Design note: Uses a single shared QKV projection for all streams. Stream identity
    is implicit via position in the concatenated sequence [v_p, v_f, a_p, a_f].
    Optionally, stream-type embeddings can be added to make stream identity explicit.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.use_stream_type_emb = use_stream_type_emb
        
        # Single shared QKV projection for all streams
        # Alternative: separate QKV per stream (not implemented here)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Optional stream-type embeddings to make stream identity explicit
        # 4 stream types: v_p (past video), v_f (future video), a_p (past action), a_f (future action)
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
        causal: bool = False,  # CORRECTED: Default to False for bidirectional attention
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply joint temporal attention across all streams with 1D RoPE.
        
        CORRECTED: No causal masking (bidirectional), no spatial pooling.
        
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
        
        # CORRECTED: NO spatial pooling - treat each spatial patch independently
        # Reshape video tokens: (B, T, S, d) -> (B*S, T, d)
        # This allows each spatial patch to attend independently
        v_p_reshaped = rearrange(v_p, 'b t s d -> (b s) t d')  # (B*S, T_p, d)
        v_f_reshaped = rearrange(v_f, 'b t s d -> (b s) t d')  # (B*S, T_f, d)
        
        # CORRECTED: Repeat actions for each spatial patch
        # This enables spatial-specific action conditioning
        # Use einops repeat for cleaner code: (B, T, d) -> (B*S, T, d)
        a_p_reshaped = repeat(a_p, 'b t d -> (b s) t d', s=S)  # (B*S, T_p, d)
        a_f_reshaped = repeat(a_f, 'b t d -> (b s) t d', s=S)  # (B*S, T_f, d)
        
        # Concatenate all streams: [v_p, v_f, a_p, a_f]
        # Total sequence length: T_p + T_f + T_p + T_f = 2*(T_p + T_f)
        # Stream identity is implicit via position in this concatenation
        all_tokens = torch.cat([v_p_reshaped, v_f_reshaped, a_p_reshaped, a_f_reshaped], dim=1)  # (B*S, total_T, d)
        total_T = all_tokens.shape[1]
        
        # Optionally add stream-type embeddings to make stream identity explicit
        if self.use_stream_type_emb:
            # Create stream type indices: [0,0,...,0, 1,1,...,1, 2,2,...,2, 3,3,...,3]
            # for [v_p, v_f, a_p, a_f] respectively
            stream_indices = torch.cat([
                torch.zeros(T_p, dtype=torch.long, device=all_tokens.device),  # v_p
                torch.ones(T_f, dtype=torch.long, device=all_tokens.device),  # v_f
                torch.full((T_p,), 2, dtype=torch.long, device=all_tokens.device),  # a_p
                torch.full((T_f,), 3, dtype=torch.long, device=all_tokens.device),  # a_f
            ])  # (total_T,)
            stream_emb = self.stream_type_emb[stream_indices]  # (total_T, d_model)
            all_tokens = all_tokens + stream_emb.unsqueeze(0)  # (B*S, total_T, d_model)
        
        # Compute QKV
        qkv = self.qkv(all_tokens)  # (B*S, total_T, 3*d)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention: (B*S, num_heads, total_T, head_dim)
        q = q.view(-1, total_T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, total_T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, total_T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply 1D RoPE to queries and keys
        if self.use_rope:
            # Transpose for RoPE: (B*S, total_T, num_heads, head_dim)
            q_temp = q.transpose(1, 2)
            k_temp = k.transpose(1, 2)
            
            q_temp = self.rope_1d(q_temp)
            k_temp = self.rope_1d(k_temp)
            
            # Transpose back: (B*S, num_heads, total_T, head_dim)
            q = q_temp.transpose(1, 2)
            k = k_temp.transpose(1, 2)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*S, num_heads, total_T, total_T)
        
        # CORRECTED: NO causal masking by default (bidirectional for MaskGIT-style)
        # Only apply if explicitly requested (for compatibility)
        if causal:
            # Apply causal mask
            causal_mask = torch.triu(torch.ones(total_T, total_T, device=attn.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = attn @ v  # (B*S, num_heads, total_T, head_dim)
        out = out.transpose(1, 2).reshape(-1, total_T, self.d_model)  # (B*S, total_T, d)
        
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

        # Layer norms
        self.spatial_norm = nn.LayerNorm(config.d_model)
        self.spatial_mlp_norm = nn.LayerNorm(config.d_model)  # For intermediate MLP after spatial attn
        self.temporal_norm = nn.LayerNorm(config.d_model)
        self.mlp_norm = nn.LayerNorm(config.d_model)

        # Spatial attention (video only) with 2D RoPE
        if self.use_sharing:
            # Shared spatial attention for both video streams
            self.spatial_attn = SpatialAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                spatial_size=config.spatial_size,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                attn_drop=config.attn_drop,
                use_rope=config.use_rope,
            )
        else:
            # Separate spatial attention for each video stream
            self.spatial_attn_vp = SpatialAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                spatial_size=config.spatial_size,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                attn_drop=config.attn_drop,
                use_rope=config.use_rope,
            )
            self.spatial_attn_vf = SpatialAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                spatial_size=config.spatial_size,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                attn_drop=config.attn_drop,
                use_rope=config.use_rope,
            )

        # Intermediate MLP for video streams (after spatial attention, before temporal)
        # This matches the diagram which shows MLP between spatial and temporal for video
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
        # max_seq_len = 2 * (T_p + T_f) for concatenated streams
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
    """
    
    def __init__(self, config: MaskedHWMConfig):
        super().__init__()
        self.config = config
        
        self.layers = nn.ModuleList([
            SharedTransformerBlock(i, config)
            for i in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)
    
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
            v_p, v_f, a_p, a_f = layer(v_p, v_f, a_p, a_f)
        
        # Final layer norm
        v_p = self.final_norm(v_p)
        v_f = self.final_norm(v_f)
        a_p = self.final_norm(a_p)
        a_f = self.final_norm(a_f)
        
        return v_p, v_f, a_p, a_f
