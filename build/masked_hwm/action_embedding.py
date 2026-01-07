"""Action embedding module for Masked-HWM."""

import torch
import torch.nn as nn


class ActionEmbedding(nn.Module):
    """Embed action vectors (R^25) to token dimension.
    
    v2.0 dataset: 25 dimensions matching paper's R^25 specification:
    - Indices 0-20: Joint positions (21 dims)
    - Index 21: Left hand closure (1 dim)
    - Index 22: Right hand closure (1 dim)
    - Index 23: Linear Velocity (1 dim)
    - Index 24: Angular Velocity (1 dim)
    
    Architecture: Linear(25, d_model) -> GELU -> Linear(d_model, d_model)
    """
    
    def __init__(self, action_dim: int = 25, d_model: int = 512):
        """Initialize action embedding.
        
        Args:
            action_dim: Dimension of action vectors (default: 25 for v2.0)
            d_model: Output token dimension (default: 512)
        """
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model
        
        self.embedding = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed action vectors to tokens.
        
        Args:
            actions: Action vectors (B, T, action_dim) or (B, action_dim)
            
        Returns:
            embedded: Embedded actions (B, T, d_model) or (B, d_model)
        """
        return self.embedding(actions)
