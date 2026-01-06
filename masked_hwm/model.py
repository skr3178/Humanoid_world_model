"""Main Masked-HWM model for v2.0 dataset."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple

from .config import MaskedHWMConfig
from .action_embedding import ActionEmbedding
from .transformer import SharedTransformer


class FactorizedEmbedding(nn.Module):
    """Factorized embedding for v2.0 tokenized video.
    
    v2.0 uses 3 factorized tokens per spatial position.
    Each factor has vocab_size=512, total vocab = 512^3.
    
    We embed each factor separately and sum the embeddings.
    """
    
    def __init__(
        self,
        num_factored_vocabs: int,
        vocab_size: int,
        d_model: int,
    ):
        """Initialize factorized embedding.
        
        Args:
            num_factored_vocabs: Number of factored tokens per position (3 for v2.0)
            vocab_size: Vocabulary size per factor (512 for v2.0)
            d_model: Model dimension
        """
        super().__init__()
        self.num_factored_vocabs = num_factored_vocabs
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Separate embedding for each factor
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, d_model)  # +1 for mask token
            for _ in range(num_factored_vocabs)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed factorized tokens.
        
        Args:
            x: Factorized tokens (B, num_factors, ...) or (..., num_factors)
               where num_factors is at position 1 or -1
               
        Returns:
            Summed embeddings (..., d_model)
        """
        # Assume x shape is (B, num_factors, T, H, W) for video
        # or (B, T, num_factors, H, W) depending on format
        
        # Sum embeddings from each factor
        embeds = []
        for i, embed in enumerate(self.embeddings):
            # Select factor i: x[:, i] gives (B, T, H, W)
            factor_tokens = x[:, i]  # (B, T, H, W)
            factor_embed = embed(factor_tokens)  # (B, T, H, W, d_model)
            embeds.append(factor_embed)
        
        # Sum all factor embeddings
        return sum(embeds)


class MaskedHWM(nn.Module):
    """Masked Humanoid World Model with shared parameters (v2.0 format).
    
    Handles v2.0 dataset with:
    - Factorized token embeddings (3 factors × vocab_size=512)
    - Clip-based video (17 frames per clip, temporally compressed)
    - 25-dimensional action space (paper's R^25)
    """
    
    def __init__(self, config: MaskedHWMConfig):
        """Initialize Masked-HWM model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Factorized token embeddings for video tokens
        self.video_token_embed = FactorizedEmbedding(
            num_factored_vocabs=config.num_factored_vocabs,
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )
        
        # Action embedding
        self.action_embedding = ActionEmbedding(
            action_dim=config.action_dim,
            d_model=config.d_model
        )
        
        # Position embeddings
        # For video: spatial + temporal positions (working at clip level)
        # Note: num_past_frames and num_future_frames are in frames, 
        # but video is at clip level. We use frame-level for actions.
        num_past_clips = max(1, config.num_past_frames // config.frames_per_clip + 1)
        num_future_clips = max(1, config.num_future_frames // config.frames_per_clip + 1)
        total_clips = num_past_clips + num_future_clips
        
        self.video_pos_embed = nn.Parameter(
            torch.zeros(1, total_clips, config.spatial_size, config.spatial_size, config.d_model)
        )
        # Action position embeddings at clip level (to match temporal attention)
        self.action_pos_embed = nn.Parameter(
            torch.zeros(1, total_clips, config.d_model)
        )
        
        self.num_past_clips = num_past_clips
        self.num_future_clips = num_future_clips
        
        # Transformer
        self.transformer = SharedTransformer(config)
        
        # Output projection - separate head for each factor
        self.output_projs = nn.ModuleList([
            nn.Linear(config.d_model, config.vocab_size)
            for _ in range(config.num_factored_vocabs)
        ])
        # Mark output projections for Xavier initialization (per paper)
        for proj in self.output_projs:
            proj._is_output_proj = True
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Initialize position embeddings
        nn.init.normal_(self.video_pos_embed, std=config.init_std)
        nn.init.normal_(self.action_pos_embed, std=config.init_std)
    
    def _init_weights(self, module):
        """Initialize weights.
        
        Per paper: Standard normal (µ=0, σ=0.02) for all weights, except:
        - Xavier initialization for mask token embeddings
        - Xavier initialization for output projections
        """
        if isinstance(module, nn.Linear):
            # Check if this is an output projection
            if hasattr(module, '_is_output_proj') and module._is_output_proj:
                # Xavier initialization for output projections (per paper)
                nn.init.xavier_uniform_(module.weight)
            else:
                # Standard normal initialization for all other linear layers
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize all embeddings with normal, then override mask token with Xavier
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            # Mask token is at index vocab_size (last index: vocab_size + 1 - 1)
            # Check if this embedding has a mask token (vocab_size + 1 entries)
            if module.weight.shape[0] == self.config.vocab_size + 1:
                # Xavier initialization for mask token row (per paper)
                nn.init.xavier_uniform_(module.weight[-1:])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _downsample_actions_to_clips(
        self, 
        actions: torch.Tensor, 
        num_clips: int
    ) -> torch.Tensor:
        """Downsample frame-level actions to clip-level.
        
        Uses average pooling to match clip rate.
        
        Args:
            actions: (B, T_frames, d_model)
            num_clips: Target number of clips
            
        Returns:
            Downsampled actions (B, num_clips, d_model)
        """
        B, T_frames, d = actions.shape
        
        if T_frames <= num_clips:
            # Pad or repeat if we have fewer frames than clips
            if T_frames == num_clips:
                return actions
            # Repeat last frame
            padding = actions[:, -1:].repeat(1, num_clips - T_frames, 1)
            return torch.cat([actions, padding], dim=1)
        
        # Average pool frames to match clip count
        # Simple approach: chunk and average
        frames_per_chunk = T_frames // num_clips
        chunks = []
        for i in range(num_clips):
            start = i * frames_per_chunk
            end = (i + 1) * frames_per_chunk if i < num_clips - 1 else T_frames
            chunk = actions[:, start:end].mean(dim=1, keepdim=True)
            chunks.append(chunk)
        
        return torch.cat(chunks, dim=1)  # (B, num_clips, d_model)
    
    def forward(
        self,
        v_p_tokens: torch.Tensor,  # (B, num_factors, T_p_clips, H, W)
        v_f_tokens: torch.Tensor,  # (B, num_factors, T_f_clips, H, W)
        a_p: torch.Tensor,  # (B, T_p_frames, action_dim)
        a_f: torch.Tensor,  # (B, T_f_frames, action_dim)
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            v_p_tokens: Past video tokens (B, num_factors, T_p_clips, H, W)
            v_f_tokens: Future video tokens (B, num_factors, T_f_clips, H, W)
            a_p: Past actions (B, T_p_frames, action_dim)
            a_f: Future actions (B, T_f_frames, action_dim)
            
        Returns:
            logits: List of prediction logits, one per factor
                    Each: (B, T_f_clips, H, W, vocab_size)
        """
        B = v_p_tokens.shape[0]
        T_p_clips = v_p_tokens.shape[2]
        T_f_clips = v_f_tokens.shape[2]
        H, W = self.config.spatial_size, self.config.spatial_size
        
        # Embed video tokens using factorized embedding
        # Input: (B, num_factors, T, H, W)
        v_p_emb = self.video_token_embed(v_p_tokens)  # (B, T_p, H, W, d_model)
        v_f_emb = self.video_token_embed(v_f_tokens)  # (B, T_f, H, W, d_model)
        
        # Add position embeddings
        v_p_emb = v_p_emb + self.video_pos_embed[:, :T_p_clips]
        v_f_emb = v_f_emb + self.video_pos_embed[:, T_p_clips:T_p_clips + T_f_clips]
        
        # Embed actions at frame level first
        a_p_emb = self.action_embedding(a_p)  # (B, T_p_frames, d_model)
        a_f_emb = self.action_embedding(a_f)  # (B, T_f_frames, d_model)
        
        # Downsample actions to match clip level (required for temporal attention)
        a_p_emb = self._downsample_actions_to_clips(a_p_emb, T_p_clips)  # (B, T_p_clips, d_model)
        a_f_emb = self._downsample_actions_to_clips(a_f_emb, T_f_clips)  # (B, T_f_clips, d_model)
        
        # Add position embeddings (now at clip level)
        a_p_emb = a_p_emb + self.action_pos_embed[:, :T_p_clips]
        a_f_emb = a_f_emb + self.action_pos_embed[:, T_p_clips:T_p_clips + T_f_clips]
        
        # Pass through transformer (expects 5D video: B, T, H, W, d_model)
        v_p_out, v_f_out, a_p_out, a_f_out = self.transformer(
            v_p_emb, v_f_emb, a_p_emb, a_f_emb
        )
        
        # Project to vocabulary for each factor
        logits = [proj(v_f_out) for proj in self.output_projs]
        # Stack: (B, T_f, H, W, vocab_size) × num_factors -> (num_factors, B, T_f, H, W, vocab_size)
        logits = torch.stack(logits, dim=0)
        
        return logits
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked cross-entropy loss for factorized tokens.
        
        Args:
            logits: Prediction logits (num_factors, B, T_f, H, W, vocab_size)
            targets: Target tokens (B, num_factors, T_f, H, W)
            mask: Mask indicating which tokens to predict (B, T_f, H, W)
            
        Returns:
            loss: Scalar loss value (sum of losses for all factors)
        """
        num_factors = logits.shape[0]
        total_loss = 0.0
        
        for i in range(num_factors):
            factor_logits = logits[i]  # (B, T_f, H, W, vocab_size)
            factor_targets = targets[:, i]  # (B, T_f, H, W)
            
            # Flatten spatial and temporal dimensions
            logits_flat = rearrange(factor_logits, 'b t h w v -> (b t h w) v')
            targets_flat = rearrange(factor_targets, 'b t h w -> (b t h w)')
            mask_flat = rearrange(mask, 'b t h w -> (b t h w)')
            
            # Compute cross-entropy
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            
            # Apply mask: only compute loss on masked positions
            loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            total_loss = total_loss + loss
        
        return total_loss
