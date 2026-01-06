"""Configuration for Masked-HWM model (v2.0 dataset)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MaskedHWMConfig:
    """Configuration for Masked Humanoid World Model.
    
    Updated for v2.0 dataset with:
    - 25-dimensional action space (matches paper's R^25)
    - 32×32 spatial tokens (Cosmos DV 8×8×8 tokenizer)
    """
    
    # Model architecture
    num_layers: int = 24
    num_heads: int = 8
    d_model: int = 512
    mlp_hidden: int = 2048
    mlp_ratio: float = 4.0
    
    # Sequence lengths
    num_past_frames: int = 9
    num_future_frames: int = 8
    
    # Action configuration (v2.0 dataset: 25 dimensions)
    # Matches paper's R^25 specification:
    # - Indices 0-20: Joint positions (21 dims)
    #   HIP (yaw, roll, pitch), KNEE (pitch), ANKLE (roll, pitch)
    #   LEFT: SHOULDER (pitch, roll, yaw), ELBOW (pitch, yaw), WRIST (pitch, roll)
    #   RIGHT: SHOULDER (pitch, roll, yaw), ELBOW (pitch, yaw), WRIST (pitch, roll)
    #   NECK (pitch)
    # - Index 21: Left hand closure (1 dim)
    # - Index 22: Right hand closure (1 dim)
    # - Index 23: Linear Velocity (1 dim)
    # - Index 24: Angular Velocity (1 dim)
    action_dim: int = 25
    
    # Tokenizer configuration (v2.0 uses Cosmos DV 8×8×8)
    # Cosmos tokenizer: 256×256 → 32×32 spatial tokens (8× compression)
    # Factorized vocabulary: 3 tokens per position, each with vocab ~65536
    # We embed each factor separately
    vocab_size: int = 65536  # per-factor vocabulary size (2^16)
    num_factored_vocabs: int = 3  # number of factored tokens per position
    spatial_size: int = 32  # 32×32 tokens per frame
    temporal_compression: int = 8  # Cosmos DV temporal compression
    frames_per_clip: int = 17  # frames per temporally-compressed clip
    
    # Parameter sharing
    shared_layers_start: int = 4  # First 4 layers unshared, remaining shared
    
    # Training configuration (per paper)
    learning_rate: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 60000
    batch_size: int = 16  # Per paper: single NVIDIA A6000 with batch size 16
    gradient_accumulation_steps: int = 1
    
    # Masking configuration
    max_corrupt_rate: float = 0.5
    mask_token_id: int = 65536  # vocab_size (used as mask token for each factor)
    
    # Attention configuration
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    mlp_drop: float = 0.0
    use_rope: bool = True
    use_stream_type_emb: bool = False  # Add explicit stream-type embeddings to temporal attention
    
    # Initialization
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
