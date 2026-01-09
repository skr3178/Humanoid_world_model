"""Ultra-minimal configuration for very limited GPU memory (<8GB).

Extremely reduced model size for testing on small GPUs:
- 3 layers (ultra-minimal depth)
- 96 dim (very small embedding dimension)
- 4 heads (24 dim per head)
- 384 MLP hidden
- batch_size=2 with grad_accum=8 (effective batch=16)
- Sequence: 2 past + 1 future clip (per paper: 34 + 17 frames)
- Mixed precision (bf16) enabled

Memory optimizations:
- Minimal transformer architecture (3 layers, 96 dim)
- bf16 mixed precision halves activation memory
- Small batch with high accumulation
- Paper-specified sequence length maintained

Maintains critical settings:
- vocab_size: 65536 (required for Cosmos tokenizer)
- action_dim: 25 (matches paper's R^25)
- All tokenizer settings unchanged
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MaskedHWMMinimalConfig:
    """Ultra-minimal configuration for very limited GPU memory (<8GB).

    Extremely reduced model for testing on small GPUs:
    - 3 layers, 96 dim, 4 heads (ultra-minimal transformer)
    - batch_size=2 with grad_accum=8 (effective batch=16)
    - Sequence: 2 past + 1 future clip (per paper specification)
    - Memory optimizations enabled (bf16)

    Use this config when GPU memory is extremely limited (<8GB available).
    """

    # Ultra-minimal model architecture for <8GB VRAM
    num_layers: int = 3  # Ultra-minimal depth
    num_heads: int = 4  # 96/4 = 24 dim per head
    d_model: int = 96  # Very small embedding dimension
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
    spatial_size: int = 32  # 32×32 tokens per frame
    temporal_compression: int = 8  # Cosmos DV temporal compression
    frames_per_clip: int = 17  # frames per temporally-compressed clip

    # Parameter sharing (first 1 layer unshared for 3-layer model)
    shared_layers_start: int = 1  # First 1 layer unshared, remaining 2 shared

    # Training configuration optimized for <8GB GPU
    # Small models need HIGHER learning rates to learn faster
    learning_rate: float = 5e-4  # Higher LR for small model (5x baseline)
    warmup_steps: int = 50  # Short warmup to start learning quickly
    max_steps: int = 60000  # Same as full config
    batch_size: int = 2  # Very small batch to fit in memory
    gradient_accumulation_steps: int = 8  # Effective batch size = 2 * 8 = 16

    # Masking configuration (per paper)
    max_corrupt_rate: float = 0.2  # ρ_max = 0.2 (per paper - helps model learn)
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

    # Memory optimization - bf16 enabled for <8GB GPU
    # Note: Gradient checkpointing requires model-level implementation (future work)
    use_gradient_checkpointing: bool = False  # Not yet implemented at model level
    mixed_precision: Optional[str] = "bf16"  # Half precision for memory savings (critical)

    # Paths (v2.0 dataset)
    tokenizer_checkpoint_dir: str = "/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer"
    train_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
    val_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
    test_data_dir: Optional[str] = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/test_v2.0"

    # Other
    seed: int = 42
    save_steps: int = 500  # Save more frequently for quick training
    eval_steps: int = 500  # Evaluate more frequently
    logging_steps: int = 50  # Log more frequently
