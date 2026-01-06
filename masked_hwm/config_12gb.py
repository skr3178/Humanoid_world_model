"""Reduced configuration for 12GB GPU memory constraint.

Reduces model size while maintaining architecture:
- Fewer layers: 12 instead of 24
- Smaller dimensions: 256 instead of 512
- Fewer heads: 4 instead of 8
- Smaller MLP: 1024 instead of 2048

Maintains critical settings:
- vocab_size: 65536 (required for Cosmos tokenizer)
- action_dim: 25 (matches paper's R^25)
- All other tokenizer settings unchanged
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MaskedHWM12GBConfig:
    """Reduced configuration for 12GB GPU (approximately 50% of full model size).
    
    Reduces model parameters while maintaining:
    - Same vocabulary size (65536) for correct decoding
    - Same action space (25 dims)
    - Same architecture principles
    """
    
    # Reduced model architecture (50% of full model)
    num_layers: int = 12  # Reduced from 24
    num_heads: int = 4  # Reduced from 8
    d_model: int = 256  # Reduced from 512
    mlp_hidden: int = 1024  # Reduced from 2048
    mlp_ratio: float = 4.0
    
    # Sequence lengths (unchanged)
    num_past_frames: int = 9
    num_future_frames: int = 8
    
    # Action configuration (unchanged - matches paper's R^25)
    action_dim: int = 25
    
    # Tokenizer configuration (unchanged - critical for decoding)
    vocab_size: int = 65536  # per-factor vocabulary size (2^16) - MUST match Cosmos
    num_factored_vocabs: int = 3  # number of factored tokens per position
    spatial_size: int = 32  # 32×32 tokens per frame
    temporal_compression: int = 8  # Cosmos DV temporal compression
    frames_per_clip: int = 17  # frames per temporally-compressed clip
    
    # Parameter sharing (unchanged)
    shared_layers_start: int = 4  # First 4 layers unshared, remaining shared
    
    # Training configuration (per paper)
    learning_rate: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 60000
    batch_size: int = 16  # Per paper: single GPU with batch size 16
    gradient_accumulation_steps: int = 1
    
    # Masking configuration (unchanged)
    max_corrupt_rate: float = 0.5
    mask_token_id: int = 65536  # vocab_size (used as mask token for each factor)
    
    # Attention configuration (unchanged)
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    mlp_drop: float = 0.0
    use_rope: bool = True
    use_stream_type_emb: bool = False
    
    # Initialization (unchanged)
    init_std: float = 0.02
    
    # Paths (v2.0 dataset)
    tokenizer_checkpoint_dir: str = "/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer"
    train_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
    val_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
    
    # Other
    seed: int = 42
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
