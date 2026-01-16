"""Configuration optimized for RTX 3060 (12GB VRAM).

Matches working config_minimal settings with memory optimizations:
- 3 layers (same as minimal)
- 96 d_model (same as minimal - key for embedding memory)
- 4 heads (24 dim per head)
- 384 MLP hidden
- batch_size=2 with grad_accum=8 (effective batch=16)

Memory optimizations enabled:
- Mixed precision (bf16): Halves memory for activations and gradients
- F.scaled_dot_product_attention: 60-80% less attention memory (FlashAttention)
- Gradient checkpointing: 50-70% less activation memory (trades compute)

With these optimizations, can potentially scale up to larger models.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MaskedHWMRTX3060Config:
    """Configuration optimized for RTX 3060 (12GB VRAM).

    Matches config_minimal (proven to work on 12GB):
    - 3 layers, 96 dim, 4 heads
    - batch_size=2 with grad_accum=8 (effective batch=16)
    - Mixed precision (bf16)
    - Sequence: 2 past + 1 future clip (per paper specification)
    """

    # Model architecture - matches working config_minimal
    num_layers: int = 3  # Same as minimal
    num_heads: int = 4  # (96/4 = 24 dim per head)
    d_model: int = 96  # Same as minimal - key for embedding memory
    mlp_hidden: int = 384  # 96 * 4 = 384
    mlp_ratio: float = 4.0

    # Sequence lengths - Per paper: "2 past latents + 1 future latent"
    num_past_clips: int = 2  # 2 past latents (paper specification)
    num_future_clips: int = 1  # 1 future latent (paper specification)
    num_past_frames: int = 34  # 2 clips * 17 frames/clip
    num_future_frames: int = 17  # 1 clip * 17 frames/clip

    # Action configuration (unchanged - matches paper's R^25)
    action_dim: int = 25

    # Tokenizer configuration (unchanged - critical for decoding)
    vocab_size: int = 65536  # per-factor vocabulary size (2^16) - MUST match Cosmos
    num_factored_vocabs: int = 3  # number of factored tokens per position
    spatial_size: int = 32  # 32x32 tokens per frame
    temporal_compression: int = 8  # Cosmos DV temporal compression
    frames_per_clip: int = 17  # frames per temporally-compressed clip

    # Parameter sharing (first 1 layer unshared for 3-layer model)
    shared_layers_start: int = 1  # First 1 layer unshared, remaining 2 shared

    # Training configuration - matches working config_minimal
    learning_rate: float = 5e-4  # Higher LR for small model (same as minimal)
    warmup_steps: int = 50  # Short warmup (same as minimal)
    max_steps: int = 100000  # Train longer for better convergence
    batch_size: int = 2  # Same as minimal (proven to work)
    gradient_accumulation_steps: int = 8  # Effective batch size = 2 * 8 = 16

    # Masking configuration (per paper)
    max_corrupt_rate: float = 0.2  # rho_max = 0.2 (per paper)
    mask_token_id: int = 65536  # vocab_size (used as mask token for each factor)

    # Attention configuration
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    mlp_drop: float = 0.0
    use_rope: bool = True
    use_stream_type_emb: bool = False

    # Initialization
    init_std: float = 0.02

    # Memory optimization - bf16 + gradient checkpointing + SDPA
    use_gradient_checkpointing: bool = True  # Saves ~50-70% activation memory
    mixed_precision: Optional[str] = "bf16"  # Half precision for memory savings

    # Paths (v2.0 dataset)
    tokenizer_checkpoint_dir: str = "/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer"
    train_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
    val_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
    test_data_dir: Optional[str] = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/test_v2.0"

    # Other
    seed: int = 42
    save_steps: int = 1000  # Save checkpoints regularly
    eval_steps: int = 1000  # Evaluate regularly
    logging_steps: int = 100  # Log frequently
