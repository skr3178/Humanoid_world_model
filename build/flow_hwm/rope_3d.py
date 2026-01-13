"""3D Rotary Position Embeddings for Flow-HWM video tokens.

Per method.md: "RoPE is applied by modality: 3D for video tokens, 1D for action tokens"

3D RoPE applies rotary position embeddings across three dimensions:
- Temporal (T): Frame/clip position in time
- Height (H): Vertical spatial position
- Width (W): Horizontal spatial position

The head dimension is split into 3 parts, with each part receiving rotations
from one axis. This allows the model to encode 3D positional information
in a way that preserves relative position relationships.

Uses complex number multiplication for efficient vectorized rotation,
following the pattern from the existing RoPE1D/RoPE2D implementations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RoPE3D(nn.Module):
    """3D Rotary Position Embeddings for video tokens (T, H, W).

    Splits head_dim into 3 equal parts, each receiving rotations from
    one spatial/temporal axis.

    Uses complex number multiplication for efficient vectorized rotation:
        rot(x, theta) = x * e^(i*theta) = x * (cos(theta) + i*sin(theta))

    Args:
        head_dim: Dimension per attention head (must be divisible by 6,
                  since we need 3 axes × 2 for complex pairs)
        max_t: Maximum temporal dimension
        max_h: Maximum height dimension
        max_w: Maximum width dimension
        base: Base frequency for RoPE

    Input:
        x: Tensor of shape (B, T, H, W, num_heads, head_dim)

    Output:
        Rotated tensor with same shape as x
    """

    def __init__(
        self,
        head_dim: int,
        max_t: int = 16,
        max_h: int = 32,
        max_w: int = 32,
        base: float = 10000.0,
    ):
        super().__init__()

        # head_dim must be divisible by 2 (for complex pairs)
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

        self.head_dim = head_dim
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.base = base

        # Split head_dim across 3 axes (each must be even for complex pairs)
        # Allocate dims evenly, with any remainder going to temporal axis
        base_dim = (head_dim // 6) * 2  # Even allocation per axis
        remainder = head_dim - 3 * base_dim

        # Distribute: temporal gets remainder, height and width get base
        self.dim_t = base_dim + remainder
        self.dim_h = base_dim
        self.dim_w = base_dim

        # If base_dim is 0 (very small head_dim), give all to temporal
        if base_dim == 0:
            self.dim_t = head_dim
            self.dim_h = 0
            self.dim_w = 0

        # Complex pairs per axis
        self.pairs_t = self.dim_t // 2
        self.pairs_h = self.dim_h // 2
        self.pairs_w = self.dim_w // 2

        # Precompute rotation angles for temporal axis
        if self.pairs_t > 0:
            k_t = torch.arange(self.pairs_t, dtype=torch.float32)
            theta_t = 1.0 / (base ** (k_t / max(self.pairs_t, 1)))
            positions_t = torch.arange(max_t, dtype=torch.float32)
            angles_t = torch.outer(positions_t, theta_t)
            rotations_t = torch.polar(torch.ones_like(angles_t), angles_t)
            self.register_buffer("rotations_t", rotations_t)
        else:
            self.register_buffer("rotations_t", torch.zeros(max_t, 0))

        # Precompute rotation angles for height axis
        if self.pairs_h > 0:
            k_h = torch.arange(self.pairs_h, dtype=torch.float32)
            theta_h = 1.0 / (base ** (k_h / max(self.pairs_h, 1)))
            positions_h = torch.arange(max_h, dtype=torch.float32)
            angles_h = torch.outer(positions_h, theta_h)
            rotations_h = torch.polar(torch.ones_like(angles_h), angles_h)
            self.register_buffer("rotations_h", rotations_h)
        else:
            self.register_buffer("rotations_h", torch.zeros(max_h, 0))

        # Precompute rotation angles for width axis
        if self.pairs_w > 0:
            k_w = torch.arange(self.pairs_w, dtype=torch.float32)
            theta_w = 1.0 / (base ** (k_w / max(self.pairs_w, 1)))
            positions_w = torch.arange(max_w, dtype=torch.float32)
            angles_w = torch.outer(positions_w, theta_w)
            rotations_w = torch.polar(torch.ones_like(angles_w), angles_w)
            self.register_buffer("rotations_w", rotations_w)
        else:
            self.register_buffer("rotations_w", torch.zeros(max_w, 0))

    def forward(
        self,
        x: torch.Tensor,
        positions_t: Optional[torch.Tensor] = None,
        positions_h: Optional[torch.Tensor] = None,
        positions_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply 3D RoPE.

        Args:
            x: Input tensor (B, T, H, W, num_heads, head_dim)
            positions_t: Optional temporal position indices (T,)
            positions_h: Optional height position indices (H,)
            positions_w: Optional width position indices (W,)

        Returns:
            Rotated tensor with same shape as x
        """
        T, H, W = x.shape[1], x.shape[2], x.shape[3]

        # Split x into parts for each axis based on allocated dims
        parts = []
        offset = 0

        # Temporal part
        if self.dim_t > 0:
            x_t = x[..., offset:offset + self.dim_t]
            rot_t = self._get_rotations(positions_t, T, self.rotations_t, self.max_t, self.pairs_t)
            rot_t = rot_t.reshape(1, T, 1, 1, 1, -1)
            x_t_rotated = self._apply_rotation(x_t, rot_t)
            parts.append(x_t_rotated)
            offset += self.dim_t

        # Height part
        if self.dim_h > 0:
            x_h = x[..., offset:offset + self.dim_h]
            rot_h = self._get_rotations(positions_h, H, self.rotations_h, self.max_h, self.pairs_h)
            rot_h = rot_h.reshape(1, 1, H, 1, 1, -1)
            x_h_rotated = self._apply_rotation(x_h, rot_h)
            parts.append(x_h_rotated)
            offset += self.dim_h

        # Width part
        if self.dim_w > 0:
            x_w = x[..., offset:offset + self.dim_w]
            rot_w = self._get_rotations(positions_w, W, self.rotations_w, self.max_w, self.pairs_w)
            rot_w = rot_w.reshape(1, 1, 1, W, 1, -1)
            x_w_rotated = self._apply_rotation(x_w, rot_w)
            parts.append(x_w_rotated)

        # Concatenate all parts
        return torch.cat(parts, dim=-1)

    def _get_rotations(
        self,
        positions: Optional[torch.Tensor],
        size: int,
        precomputed: torch.Tensor,
        max_size: int,
        num_pairs: int,
    ) -> torch.Tensor:
        """Get rotations for a given axis."""
        if num_pairs == 0:
            return precomputed[:size]

        if positions is None:
            assert size <= max_size, f"size {size} > max_size {max_size}"
            return precomputed[:size]
        else:
            # Compute rotations for custom positions
            k = torch.arange(num_pairs, device=positions.device, dtype=torch.float32)
            theta = 1.0 / (self.base ** (k / max(num_pairs, 1)))
            angles = torch.outer(positions.float(), theta)
            return torch.polar(torch.ones_like(angles), angles)

    def _apply_rotation(
        self,
        x: torch.Tensor,
        rot: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotation to a tensor."""
        # Convert x to complex: reshape to (..., dim_per_axis, 2) then view as complex
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # Apply rotation via complex multiplication
        x_rotated = x_complex * rot

        # Convert back to real
        return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class RoPE3DVideo(nn.Module):
    """3D RoPE specifically for video tokens in Flow-HWM.

    Wrapper around RoPE3D that handles the common case of video tokens
    with shape (B, T, H*W, num_heads, head_dim) by reshaping to 3D.

    This is useful when spatial dimensions are flattened for attention
    but we still want to apply 3D positional encoding.

    Args:
        head_dim: Dimension per attention head
        max_t: Maximum temporal dimension (number of clips)
        spatial_size: Height/width of spatial grid (assumes square)
        base: Base frequency for RoPE
    """

    def __init__(
        self,
        head_dim: int,
        max_t: int = 16,
        spatial_size: int = 32,
        base: float = 10000.0,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.rope_3d = RoPE3D(
            head_dim=head_dim,
            max_t=max_t,
            max_h=spatial_size,
            max_w=spatial_size,
            base=base,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D RoPE to flattened video tokens.

        Args:
            x: Input tensor (B, T, S, num_heads, head_dim) where S = H*W

        Returns:
            Rotated tensor with same shape
        """
        B, T, S, num_heads, head_dim = x.shape
        H = W = self.spatial_size
        assert S == H * W, f"S={S} != H*W={H*W}"

        # Reshape to (B, T, H, W, num_heads, head_dim)
        x = x.view(B, T, H, W, num_heads, head_dim)

        # Apply 3D RoPE
        x = self.rope_3d(x)

        # Reshape back to (B, T, S, num_heads, head_dim)
        return x.view(B, T, S, num_heads, head_dim)


def apply_rope_3d_to_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_3d: RoPE3D,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D RoPE to queries and keys.

    Convenience function for attention computation.

    Args:
        q: Query tensor (B, T, H, W, num_heads, head_dim)
        k: Key tensor (B, T, H, W, num_heads, head_dim)
        rope_3d: RoPE3D module

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    return rope_3d(q), rope_3d(k)
