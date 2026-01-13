"""Adaptive Layer Normalization (AdaLN) for Flow-HWM.

Implements time-modulated layer normalization following DiT (Peebles & Xie, 2023)
and DDPM practices. AdaLN enables the transformer to adapt its behavior based
on the current timestep in the flow matching process.

The key idea is to generate scale and shift parameters from the time embedding,
then apply them after layer normalization:
    output = (1 + scale) * LayerNorm(x) + shift

This allows the model to:
1. Adjust feature magnitudes based on timestep (early vs late in flow)
2. Learn timestep-dependent biases
3. Dynamically modulate information flow through the network
"""

import torch
import torch.nn as nn
from torch import Tensor


class AdaLN(nn.Module):
    """Adaptive Layer Normalization with time modulation.

    Applies layer normalization followed by learned scale and shift
    parameters derived from the time embedding.

    Formula:
        output = (1 + scale(t_emb)) * LayerNorm(x) + shift(t_emb)

    The "+1" ensures that at initialization (when scale ≈ 0), the output
    is approximately equal to LayerNorm(x), providing stable training dynamics.

    Args:
        d_model: Feature dimension
        eps: Epsilon for layer normalization numerical stability

    Input:
        x: Input tensor of shape (..., d_model)
        t_emb: Time embedding of shape (B, d_model)

    Output:
        Modulated tensor of shape (..., d_model)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model

        # Layer normalization without learnable parameters
        # (scale/shift come from time embedding instead)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)

        # Project time embedding to scale and shift parameters
        # Output: 2 * d_model (scale and shift)
        self.modulation = nn.Linear(d_model, 2 * d_model)

        # Initialize to zero so initial output ≈ LayerNorm(x)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor of shape (..., d_model)
            t_emb: Time embedding of shape (B, d_model)

        Returns:
            Modulated tensor of shape (..., d_model)
        """
        # Normalize input
        x_norm = self.norm(x)

        # Get scale and shift from time embedding
        # modulation output: (B, 2 * d_model)
        modulation = self.modulation(t_emb)
        scale, shift = modulation.chunk(2, dim=-1)  # Each: (B, d_model)

        # Reshape for broadcasting
        # x_norm may be (B, T, H, W, d_model) or (B, T, d_model)
        # scale/shift are (B, d_model), need to broadcast
        if x_norm.dim() == 5:  # (B, T, H, W, d_model) - video
            scale = scale[:, None, None, None, :]
            shift = shift[:, None, None, None, :]
        elif x_norm.dim() == 3:  # (B, T, d_model) - actions
            scale = scale[:, None, :]
            shift = shift[:, None, :]
        # For (B, d_model), no reshaping needed

        # Apply modulation: (1 + scale) * norm(x) + shift
        return (1 + scale) * x_norm + shift


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization for residual paths.

    Similar to AdaLN but outputs an additional gate parameter that starts at
    zero, making the residual contribution initially zero. This is useful for
    the output of attention/MLP blocks to ensure stable initialization.

    Formula:
        output = gate(t_emb) * ((1 + scale(t_emb)) * LayerNorm(x) + shift(t_emb))

    Args:
        d_model: Feature dimension
        eps: Epsilon for layer normalization

    Input:
        x: Input tensor of shape (..., d_model)
        t_emb: Time embedding of shape (B, d_model)

    Output:
        Modulated tensor of shape (..., d_model)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)

        # Output: scale, shift, gate (3 * d_model)
        self.modulation = nn.Linear(d_model, 3 * d_model)

        # Initialize to zero
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """Apply adaptive layer normalization with gating.

        Args:
            x: Input tensor of shape (..., d_model)
            t_emb: Time embedding of shape (B, d_model)

        Returns:
            Modulated tensor of shape (..., d_model)
        """
        x_norm = self.norm(x)

        # Get scale, shift, and gate from time embedding
        modulation = self.modulation(t_emb)
        scale, shift, gate = modulation.chunk(3, dim=-1)

        # Reshape for broadcasting
        if x_norm.dim() == 5:  # (B, T, H, W, d_model)
            scale = scale[:, None, None, None, :]
            shift = shift[:, None, None, None, :]
            gate = gate[:, None, None, None, :]
        elif x_norm.dim() == 3:  # (B, T, d_model)
            scale = scale[:, None, :]
            shift = shift[:, None, :]
            gate = gate[:, None, :]

        # Apply gated modulation
        return gate * ((1 + scale) * x_norm + shift)


class FeedForwardScale(nn.Module):
    """Time-dependent scaling for feedforward output.

    Per method.md: "scale with timestep" after feedforward layers.
    This module generates a per-channel scale factor from the time embedding.

    Formula:
        output = (1 + scale(t_emb)) * x

    Args:
        d_model: Feature dimension

    Input:
        x: Input tensor of shape (..., d_model)
        t_emb: Time embedding of shape (B, d_model)

    Output:
        Scaled tensor of shape (..., d_model)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.scale_proj = nn.Linear(d_model, d_model)

        # Initialize to zero for identity at start
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """Apply time-dependent scaling.

        Args:
            x: Input tensor of shape (..., d_model)
            t_emb: Time embedding of shape (B, d_model)

        Returns:
            Scaled tensor of shape (..., d_model)
        """
        scale = self.scale_proj(t_emb)

        # Reshape for broadcasting
        if x.dim() == 5:  # (B, T, H, W, d_model)
            scale = scale[:, None, None, None, :]
        elif x.dim() == 3:  # (B, T, d_model)
            scale = scale[:, None, :]

        return (1 + scale) * x
