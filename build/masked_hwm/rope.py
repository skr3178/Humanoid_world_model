"""Rotary Position Embeddings (RoPE) for Masked-HWM."""

import torch
import torch.nn as nn
from typing import Optional


def apply_rope_1d(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply 1D RoPE to input tensor.
    
    Args:
        x: Input tensor (..., seq_len, head_dim)
        freqs: Frequencies (seq_len, head_dim // 2)
        
    Returns:
        Rotated tensor with same shape as x
    """
    # Split into real and imaginary parts
    head_dim = x.shape[-1]
    x_real = x[..., : head_dim // 2]
    x_imag = x[..., head_dim // 2 :]
    
    # Apply rotation
    cos_freqs = freqs.cos()
    sin_freqs = freqs.sin()
    
    x_rotated_real = x_real * cos_freqs - x_imag * sin_freqs
    x_rotated_imag = x_real * sin_freqs + x_imag * cos_freqs
    
    # Concatenate back
    x_rotated = torch.cat([x_rotated_real, x_rotated_imag], dim=-1)
    
    return x_rotated


def apply_rope_2d(x: torch.Tensor, freqs_h: torch.Tensor, freqs_w: torch.Tensor) -> torch.Tensor:
    """Apply 2D RoPE to input tensor.
    
    Args:
        x: Input tensor (..., H, W, head_dim)
        freqs_h: Frequencies for height (H, head_dim // 4)
        freqs_w: Frequencies for width (W, head_dim // 4)
        
    Returns:
        Rotated tensor with same shape as x
    """
    head_dim = x.shape[-1]
    
    # Split into 4 parts for 2D rotation
    dim_per_part = head_dim // 4
    x_parts = [
        x[..., :dim_per_part],
        x[..., dim_per_part:2*dim_per_part],
        x[..., 2*dim_per_part:3*dim_per_part],
        x[..., 3*dim_per_part:]
    ]
    
    # Apply 2D rotation
    cos_h = freqs_h.cos()
    sin_h = freqs_h.sin()
    cos_w = freqs_w.cos()
    sin_w = freqs_w.sin()
    
    # Rotate height dimension
    x_h_real = x_parts[0] * cos_h - x_parts[1] * sin_h
    x_h_imag = x_parts[0] * sin_h + x_parts[1] * cos_h
    
    # Rotate width dimension
    x_w_real = x_parts[2] * cos_w - x_parts[3] * sin_w
    x_w_imag = x_parts[2] * sin_w + x_parts[3] * cos_w
    
    # Concatenate back
    x_rotated = torch.cat([x_h_real, x_h_imag, x_w_real, x_w_imag], dim=-1)
    
    return x_rotated


class RoPE1D(nn.Module):
    """1D Rotary Position Embeddings for temporal dimension."""
    
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """Initialize 1D RoPE.
        
        Args:
            head_dim: Dimension per attention head
            max_seq_len: Maximum sequence length
            base: Base frequency for RoPE
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim // 2, dtype=torch.float32) / (head_dim // 2)))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply 1D RoPE.
        
        Args:
            x: Input tensor (B, seq_len, num_heads, head_dim) or (B, seq_len, head_dim)
            positions: Optional position indices (seq_len,)
            
        Returns:
            Rotated tensor with same shape as x
        """
        if x.dim() == 4:
            # (B, seq_len, num_heads, head_dim)
            B, seq_len, num_heads, head_dim = x.shape
            x_reshaped = x.reshape(B * seq_len, num_heads, head_dim)
        else:
            # (B, seq_len, head_dim)
            B, seq_len, head_dim = x.shape
            num_heads = 1
            x_reshaped = x.reshape(B * seq_len, head_dim)
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        # Compute frequencies
        freqs = torch.outer(positions, self.inv_freq.to(x.device))  # (seq_len, head_dim // 2)
        
        # Apply rotation to each position
        x_list = []
        for i in range(seq_len):
            x_i = x_reshaped[i::seq_len]  # Get all batch items at position i
            freq_i = freqs[i]  # (head_dim // 2)
            
            # Split and rotate
            head_dim = x_i.shape[-1]
            x_real = x_i[..., :head_dim // 2]
            x_imag = x_i[..., head_dim // 2:]
            
            cos_freq = freq_i.cos()
            sin_freq = freq_i.sin()
            
            x_rotated_real = x_real * cos_freq - x_imag * sin_freq
            x_rotated_imag = x_real * sin_freq + x_imag * cos_freq
            
            x_rotated = torch.cat([x_rotated_real, x_rotated_imag], dim=-1)
            x_list.append(x_rotated)
        
        # Reshape back
        x_rotated = torch.stack(x_list, dim=1)  # (B, seq_len, num_heads, head_dim) or (B, seq_len, head_dim)
        
        if x.dim() == 3:
            x_rotated = x_rotated.squeeze(2)
        
        return x_rotated


class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings for spatial dimensions."""
    
    def __init__(self, head_dim: int, max_h: int = 256, max_w: int = 256, base: float = 10000.0):
        """Initialize 2D RoPE.
        
        Args:
            head_dim: Dimension per attention head
            max_h: Maximum height
            max_w: Maximum width
            base: Base frequency for RoPE
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_h = max_h
        self.max_w = max_w
        self.base = base
        
        # Precompute frequencies for height and width
        dim_per_axis = head_dim // 4
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_per_axis, dtype=torch.float32) / dim_per_axis))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(
        self,
        x: torch.Tensor,
        positions_h: Optional[torch.Tensor] = None,
        positions_w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply 2D RoPE.
        
        Args:
            x: Input tensor (B, H, W, num_heads, head_dim)
            positions_h: Optional height position indices (H,)
            positions_w: Optional width position indices (W,)
            
        Returns:
            Rotated tensor with same shape as x
        """
        H, W = x.shape[1], x.shape[2]
        
        if positions_h is None:
            positions_h = torch.arange(H, device=x.device, dtype=torch.float32)
        if positions_w is None:
            positions_w = torch.arange(W, device=x.device, dtype=torch.float32)
        
        # Compute frequencies
        freqs_h = torch.outer(positions_h, self.inv_freq)  # (H, head_dim // 4)
        freqs_w = torch.outer(positions_w, self.inv_freq)  # (W, head_dim // 4)
        
        # Expand to match x shape
        freqs_h = freqs_h.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, H, 1, 1, head_dim // 4)
        freqs_w = freqs_w.unsqueeze(0).unsqueeze(1).unsqueeze(3)  # (1, 1, W, 1, head_dim // 4)
        
        # Apply rotation
        return apply_rope_2d(x, freqs_h, freqs_w)
