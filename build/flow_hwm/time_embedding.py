"""Time embedding modules for Flow-HWM.

Implements sinusoidal timestep embeddings following DDPM (Ho et al., 2020),
with an MLP projection to the model dimension.

The time embedding provides the model with information about the current
timestep t in the flow matching process, enabling it to predict the
appropriate velocity field at each point along the flow.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings (DDPM style).

    Creates fixed sinusoidal embeddings for continuous timesteps in [0, 1].
    Similar to positional encodings in transformers but applied to scalar time.

    The embedding uses log-spaced frequencies to capture both fine and coarse
    temporal information across the flow matching trajectory.

    Args:
        dim: Embedding dimension (will be the output dimension)
        max_period: Maximum period for the lowest frequency component

    Input:
        t: Tensor of shape (B,) with values in [0, 1]

    Output:
        embeddings: Tensor of shape (B, dim)
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """Generate sinusoidal embeddings for timesteps.

        Args:
            t: Timesteps tensor of shape (B,) with values in [0, 1]

        Returns:
            Embeddings tensor of shape (B, dim)
        """
        # Ensure t is the right shape
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2

        # Compute frequencies: log-spaced from 1 to max_period
        # freq_i = max_period^(-2i/dim) for i in [0, half_dim)
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / half_dim
        )

        # Scale timesteps by frequencies: (B, 1) * (half_dim,) -> (B, half_dim)
        args = t[:, None].float() * freqs[None, :]

        # Concatenate sin and cos embeddings
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return embedding


class TimeEmbedding(nn.Module):
    """Full time embedding module: sinusoidal + MLP projection.

    Combines sinusoidal embeddings with an MLP to project to the model
    dimension. The MLP allows the model to learn task-specific time
    representations beyond the fixed sinusoidal basis.

    Architecture:
        sinusoidal(t) -> Linear -> SiLU -> Linear -> output

    Args:
        d_model: Output embedding dimension
        sinusoidal_dim: Dimension of sinusoidal embeddings (default: 256)
        max_period: Maximum period for sinusoidal embeddings

    Input:
        t: Tensor of shape (B,) with values in [0, 1]

    Output:
        embeddings: Tensor of shape (B, d_model)
    """

    def __init__(
        self,
        d_model: int,
        sinusoidal_dim: int = 256,
        max_period: int = 10000,
    ):
        super().__init__()
        self.d_model = d_model
        self.sinusoidal_dim = sinusoidal_dim

        # Sinusoidal embedding layer
        self.sinusoidal = SinusoidalTimestepEmbedding(
            dim=sinusoidal_dim,
            max_period=max_period,
        )

        # MLP projection: sinusoidal_dim -> d_model
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Generate time embeddings.

        Args:
            t: Timesteps tensor of shape (B,) with values in [0, 1]

        Returns:
            Time embeddings of shape (B, d_model)
        """
        # Sinusoidal embedding
        t_emb = self.sinusoidal(t)

        # MLP projection (match module dtype for mixed precision)
        t_emb = t_emb.to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_emb)

        return t_emb


class TimestepEmbedSequential(nn.Sequential):
    """Sequential module that passes time embeddings to child modules.

    Useful for building blocks that need time conditioning throughout.
    Modules that accept time embeddings should have a `forward` signature
    that takes (x, t_emb) as arguments.
    """

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        for module in self:
            if hasattr(module, "forward") and "t_emb" in module.forward.__code__.co_varnames:
                x = module(x, t_emb)
            else:
                x = module(x)
        return x
