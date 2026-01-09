"""Rotary Position Embeddings (RoPE) for Masked-HWM.

Optimized implementation using complex number multiplication.
Inspired by RoPE-ND: https://github.com/lucidrains/rotary-embedding-torch
"""

import torch
import torch.nn as nn
from typing import Optional


class RoPE1D(nn.Module):
    """1D Rotary Position Embeddings for temporal dimension.

    Uses complex number multiplication for efficient vectorized rotation.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """Initialize 1D RoPE.

        Args:
            head_dim: Dimension per attention head (must be even)
            max_seq_len: Maximum sequence length to precompute
            base: Base frequency for RoPE
        """
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation angles
        # theta_k = 1 / (base^(2k/d)) for k in [0, d/2)
        k = torch.arange(head_dim // 2, dtype=torch.float32)
        theta = 1.0 / (base ** (k / (head_dim // 2)))

        # Precompute rotations for all positions up to max_seq_len
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, theta)  # (max_seq_len, head_dim // 2)

        # Convert to complex rotations: e^(i * angle) = cos(angle) + i * sin(angle)
        rotations = torch.polar(torch.ones_like(angles), angles)  # (max_seq_len, head_dim // 2)

        self.register_buffer("rotations", rotations)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply 1D RoPE.

        Args:
            x: Input tensor (B, seq_len, num_heads, head_dim)
            positions: Optional custom position indices (seq_len,). If None, uses 0, 1, 2, ...

        Returns:
            Rotated tensor with same shape as x
        """
        seq_len = x.shape[1]

        # Get rotations for the sequence
        if positions is None:
            # Use precomputed rotations
            assert seq_len <= self.max_seq_len, f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"
            rot = self.rotations[:seq_len]  # (seq_len, head_dim // 2)
        else:
            # Compute rotations for custom positions
            k = torch.arange(self.head_dim // 2, device=x.device, dtype=torch.float32)
            theta = 1.0 / (self.base ** (k / (self.head_dim // 2)))
            angles = torch.outer(positions.float(), theta)
            rot = torch.polar(torch.ones_like(angles), angles)

        # Reshape rotations for broadcasting: (1, seq_len, 1, head_dim // 2)
        rot = rot.unsqueeze(0).unsqueeze(2)

        # Convert x to complex: view pairs of adjacent dims as complex numbers
        # x shape: (B, seq_len, num_heads, head_dim)
        # Reshape to (B, seq_len, num_heads, head_dim // 2, 2) then view as complex
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # Apply rotation via complex multiplication
        x_rotated = x_complex * rot

        # Convert back to real: (B, seq_len, num_heads, head_dim // 2) complex
        # -> (B, seq_len, num_heads, head_dim // 2, 2) real -> (B, seq_len, num_heads, head_dim)
        return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings for spatial dimensions.

    Uses complex number multiplication for efficient vectorized rotation.
    Splits head_dim into two halves: one for height, one for width.
    """

    def __init__(self, head_dim: int, max_h: int = 256, max_w: int = 256, base: float = 10000.0):
        """Initialize 2D RoPE.

        Args:
            head_dim: Dimension per attention head (must be divisible by 4)
            max_h: Maximum height to precompute
            max_w: Maximum width to precompute
            base: Base frequency for RoPE
        """
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"

        self.head_dim = head_dim
        self.max_h = max_h
        self.max_w = max_w
        self.base = base

        # Each spatial dimension gets head_dim // 4 complex pairs (head_dim // 2 real dims)
        dim_per_axis = head_dim // 4

        # Precompute rotation angles for each axis
        k = torch.arange(dim_per_axis, dtype=torch.float32)
        theta = 1.0 / (base ** (k / dim_per_axis))

        # Precompute rotations for height positions
        positions_h = torch.arange(max_h, dtype=torch.float32)
        angles_h = torch.outer(positions_h, theta)  # (max_h, dim_per_axis)
        rotations_h = torch.polar(torch.ones_like(angles_h), angles_h)

        # Precompute rotations for width positions
        positions_w = torch.arange(max_w, dtype=torch.float32)
        angles_w = torch.outer(positions_w, theta)  # (max_w, dim_per_axis)
        rotations_w = torch.polar(torch.ones_like(angles_w), angles_w)

        self.register_buffer("rotations_h", rotations_h)
        self.register_buffer("rotations_w", rotations_w)

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
        dim_per_axis = self.head_dim // 4

        # Get rotations for height
        if positions_h is None:
            assert H <= self.max_h, f"H {H} > max_h {self.max_h}"
            rot_h = self.rotations_h[:H]  # (H, dim_per_axis)
        else:
            k = torch.arange(dim_per_axis, device=x.device, dtype=torch.float32)
            theta = 1.0 / (self.base ** (k / dim_per_axis))
            angles_h = torch.outer(positions_h.float(), theta)
            rot_h = torch.polar(torch.ones_like(angles_h), angles_h)

        # Get rotations for width
        if positions_w is None:
            assert W <= self.max_w, f"W {W} > max_w {self.max_w}"
            rot_w = self.rotations_w[:W]  # (W, dim_per_axis)
        else:
            k = torch.arange(dim_per_axis, device=x.device, dtype=torch.float32)
            theta = 1.0 / (self.base ** (k / dim_per_axis))
            angles_w = torch.outer(positions_w.float(), theta)
            rot_w = torch.polar(torch.ones_like(angles_w), angles_w)

        # Reshape rotations for broadcasting
        # rot_h: (H, dim_per_axis) -> (1, H, 1, 1, dim_per_axis)
        # rot_w: (W, dim_per_axis) -> (1, 1, W, 1, dim_per_axis)
        rot_h = rot_h.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        rot_w = rot_w.unsqueeze(0).unsqueeze(1).unsqueeze(3)

        # Split x into height and width parts
        # x shape: (B, H, W, num_heads, head_dim)
        # Split last dim: first half for height rotation, second half for width rotation
        x_h = x[..., :self.head_dim // 2]  # (B, H, W, num_heads, head_dim // 2)
        x_w = x[..., self.head_dim // 2:]  # (B, H, W, num_heads, head_dim // 2)

        # Convert to complex: reshape to (..., dim_per_axis, 2) then view as complex
        x_h_complex = torch.view_as_complex(x_h.float().reshape(*x_h.shape[:-1], -1, 2))
        x_w_complex = torch.view_as_complex(x_w.float().reshape(*x_w.shape[:-1], -1, 2))

        # Apply rotations
        x_h_rotated = x_h_complex * rot_h
        x_w_rotated = x_w_complex * rot_w

        # Convert back to real and concatenate
        x_h_real = torch.view_as_real(x_h_rotated).flatten(-2)
        x_w_real = torch.view_as_real(x_w_rotated).flatten(-2)

        return torch.cat([x_h_real, x_w_real], dim=-1).type_as(x)


class RoPENd(nn.Module):
    """N-dimensional Rotary Position Embeddings (general purpose).

    Can handle arbitrary spatial dimensions. For 1D use shape=(seq_len, head_dim),
    for 2D use shape=(H, W, head_dim), etc.
    """

    def __init__(self, shape: tuple, base: float = 10000.0):
        """Initialize N-dimensional RoPE.

        Args:
            shape: Tuple of (dim1, dim2, ..., dimN, head_dim) specifying max sizes
            base: Base frequency for RoPE
        """
        super().__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % (2 * len(channel_dims)) == 0, (
            f"head_dim ({feature_dim}) must be divisible by 2 * num_dims ({2 * len(channel_dims)})"
        )

        # Compute rotation frequencies
        theta = 1.0 / (base ** (torch.arange(k_max, dtype=torch.float32) / k_max))

        # Create meshgrid of positions for all dimensions
        grids = torch.meshgrid(
            [torch.arange(d, dtype=torch.float32) for d in channel_dims],
            indexing='ij'
        )

        # Stack angles: multiply each position grid by theta
        angles = torch.cat([g.unsqueeze(-1) * theta for g in grids], dim=-1)

        # Convert to complex rotations
        rotations = torch.polar(torch.ones_like(angles), angles)

        self.register_buffer("rotations", rotations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply N-dimensional RoPE.

        Args:
            x: Input tensor with shape matching initialized dimensions

        Returns:
            Rotated tensor with same shape as x
        """
        # Convert to complex, rotate, convert back
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * self.rotations
        return torch.view_as_real(x_rotated).flatten(-2).type_as(x)
