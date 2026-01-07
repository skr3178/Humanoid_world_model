"""Wrapper for Cosmos DV 8×8×8 tokenizer."""

import os
import torch
import numpy as np
from typing import Tuple, Optional


class CosmosTokenizerWrapper:
    """Wrapper for Cosmos Discrete Video (DV) tokenizer.
    
    Handles encoding/decoding of videos using Cosmos DV 8×8×8 tokenizer.
    Input: RGB video (B, 3, T, 256, 256) in range [-1, 1]
    Output: Discrete tokens (B, T, 32, 32) with values in [0, vocab_size-1]
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        dtype: str = "bfloat16"
    ):
        """Initialize Cosmos tokenizer.
        
        Args:
            checkpoint_dir: Directory containing encoder.jit and decoder.jit
            device: Device to run on ('cuda' or 'cpu')
            dtype: Data type ('bfloat16' or 'float32')
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        
        # Lazy loading - will load when needed
        self._encoder = None
        self._decoder = None
        
    def _load_encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                from cosmos_tokenizer.video_lib import CausalVideoTokenizer
            except ImportError:
                raise ImportError(
                    "cosmos_tokenizer not found. Please activate the 'cosmos-tokenizer' conda environment."
                )
            
            encoder_path = os.path.join(self.checkpoint_dir, "encoder.jit")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
            
            self._encoder = CausalVideoTokenizer(
                checkpoint_enc=encoder_path,
                device=self.device,
                dtype="bfloat16" if self.dtype == torch.bfloat16 else "float32"
            )
        return self._encoder
    
    def _load_decoder(self):
        """Lazy load decoder."""
        if self._decoder is None:
            try:
                from cosmos_tokenizer.video_lib import CausalVideoTokenizer
            except ImportError:
                raise ImportError(
                    "cosmos_tokenizer not found. Please activate the 'cosmos-tokenizer' conda environment."
                )
            
            decoder_path = os.path.join(self.checkpoint_dir, "decoder.jit")
            if not os.path.exists(decoder_path):
                raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_path}")
            
            self._decoder = CausalVideoTokenizer(
                checkpoint_dec=decoder_path,
                device=self.device,
                dtype="bfloat16" if self.dtype == torch.bfloat16 else "float32"
            )
        return self._decoder
    
    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to discrete tokens.
        
        Args:
            video: Input video tensor (B, 3, T, H, W) in range [-1, 1]
            
        Returns:
            tokens: Discrete token indices (B, T, H//8, W//8)
                   For 256×256 input, output is (B, T, 32, 32)
        """
        if video.dim() != 5:
            raise ValueError(f"Expected 5D tensor (B, C, T, H, W), got {video.dim()}D")
        
        # Ensure video is on correct device and dtype
        video = video.to(self.device).to(self.dtype)
        
        encoder = self._load_encoder()
        
        # Cosmos encode returns (indices, codes) for discrete tokenizer
        # indices: (B, T, H//8, W//8) - these are the discrete tokens we need
        output = encoder.encode(video)
        
        if isinstance(output, tuple):
            # For discrete tokenizer, first element is indices
            indices = output[0]  # (B, T, H//8, W//8)
        else:
            # If single tensor, assume it's the indices
            indices = output
        
        return indices.long()  # Ensure integer type
    
    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens to video.
        
        IMPORTANT: For factorized tokens (v2.0 dataset format), pass [B, 3, H, W] directly.
        The Cosmos decoder accepts the 3-factor format natively - do NOT combine factors.
        
        Args:
            tokens: Token indices
              - For single tokens: (B, T, H, W) - standard format
              - For factorized tokens: (B, 3, H, W) - v2.0 dataset format (all 3 factors)
              For 32×32 tokens, output is (B, 3, T, 256, 256) or (B, 3, 17, 256, 256) for factorized
            
        Returns:
            video: Reconstructed video (B, 3, T, H*8, W*8) in range [-1, 1]
                  For factorized input [B, 3, H, W], output is [B, 3, 17, 256, 256] (17 frames per clip)
        """
        # Handle both formats: [B, T, H, W] (single tokens) or [B, 3, H, W] (factorized)
        if tokens.dim() == 4:
            if tokens.shape[1] == 3:
                # Factorized format: [B, 3, H, W] - pass directly to decoder
                # This is the correct format for v2.0 dataset
                tokens = tokens.to(self.device).long()
                decoder = self._load_decoder()
                video = decoder.decode(tokens)
                return video  # (B, 3, 17, 256, 256) - 17 frames per clip
            else:
                # Standard format: [B, T, H, W]
                tokens = tokens.to(self.device).long()
                decoder = self._load_decoder()
                video = decoder.decode(tokens)
                return video  # (B, 3, T, H*8, W*8)
        else:
            raise ValueError(f"Expected 4D tensor (B, T, H, W) or (B, 3, H, W), got {tokens.dim()}D")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size (codebook size)."""
        # Cosmos DV uses FSQ with ~64K codebook
        return 65536
