"""Attention module with RoPE support for Masked-HWM."""

import torch
import torch.nn as nn
from typing import Optional

try:
    from xformers.ops import memory_efficient_attention, LowerTriangularMask
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

from .rope import RoPE1D, RoPE2D


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        use_rope_1d: bool = False,
        use_rope_2d: bool = False,
        max_seq_len: int = 2048,
        max_spatial: int = 256,
    ):
        """Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            proj_bias: Whether to use bias in output projection
            attn_drop: Attention dropout rate
            use_rope_1d: Whether to use 1D RoPE (for temporal)
            use_rope_2d: Whether to use 2D RoPE (for spatial)
            max_seq_len: Maximum sequence length for 1D RoPE
            max_spatial: Maximum spatial dimension for 2D RoPE
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # RoPE
        self.use_rope_1d = use_rope_1d
        self.use_rope_2d = use_rope_2d
        if use_rope_1d:
            self.rope_1d = RoPE1D(self.head_dim, max_seq_len)
        if use_rope_2d:
            self.rope_2d = RoPE2D(self.head_dim, max_spatial, max_spatial)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        apply_spatial_rope: bool = False,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, seq_len, d_model) or (B, H, W, d_model) for spatial
            mask: Optional attention mask
            causal: Whether to apply causal masking
            apply_spatial_rope: Whether to apply 2D RoPE (for spatial attention)
            
        Returns:
            Output tensor with same shape as input
        """
        B = x.shape[0]
        
        if apply_spatial_rope and self.use_rope_2d:
            # Spatial attention: (B, H, W, d_model)
            H, W = x.shape[1], x.shape[2]
            x_flat = x.reshape(B, H * W, self.d_model)
        else:
            # Temporal attention: (B, seq_len, d_model)
            seq_len = x.shape[1]
            x_flat = x
        
        # Compute QKV
        qkv = self.qkv(x_flat)  # (B, seq_len, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, seq_len, d_model)
        
        # Reshape for multi-head attention
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if apply_spatial_rope and self.use_rope_2d:
            # Reshape for 2D RoPE: (B, num_heads, H, W, head_dim)
            q = q.view(B, self.num_heads, H, W, self.head_dim)
            k = k.view(B, self.num_heads, H, W, self.head_dim)
            q = self.rope_2d(q)
            k = self.rope_2d(k)
            # Reshape back: (B, num_heads, H*W, head_dim)
            q = q.view(B, self.num_heads, H * W, self.head_dim)
            k = k.view(B, self.num_heads, H * W, self.head_dim)
        elif self.use_rope_1d:
            # Apply 1D RoPE: (B, num_heads, seq_len, head_dim)
            q = self.rope_1d(q)
            k = self.rope_1d(k)
        
        # Compute attention
        if XFORMERS_AVAILABLE and not apply_spatial_rope:
            # Use xformers for efficiency (only for temporal attention)
            attn_bias = LowerTriangularMask() if causal else None
            out = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale)
            out = out.reshape(B, -1, self.d_model)
        else:
            # Standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, seq_len, seq_len)
            
            if causal:
                # Apply causal mask
                seq_len = attn.shape[-1]
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attn.device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(causal_mask, float('-inf'))
            
            if mask is not None:
                attn = attn.masked_fill(~mask, float('-inf'))
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            out = attn @ v  # (B, num_heads, seq_len, head_dim)
            out = out.transpose(1, 2).reshape(B, -1, self.d_model)  # (B, seq_len, d_model)
        
        # Output projection
        out = self.proj(out)
        
        if apply_spatial_rope and self.use_rope_2d:
            # Reshape back to spatial: (B, H, W, d_model)
            out = out.view(B, H, W, self.d_model)
        
        return out
