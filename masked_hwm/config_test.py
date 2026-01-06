"""Reduced configuration for quick testing (~5 minutes)."""

from dataclasses import dataclass


@dataclass
class MaskedHWMTestConfig:
    """Reduced configuration for quick testing.
    
    Smaller model for fast iteration:
    - Fewer layers (4 instead of 24)
    - Smaller dimensions (128 instead of 512)
    - Reduced vocab for memory (8192 instead of 262144)
    - Fewer training steps (100 instead of 60k)
    
    v2.0 dataset format:
    - 25-dimensional action space (matches paper's R^25)
    - 32×32 spatial tokens (Cosmos DV 8×8×8 tokenizer)
    """
    
    # Reduced model architecture
    num_layers: int = 4
    num_heads: int = 4
    d_model: int = 128
    mlp_hidden: int = 512
    mlp_ratio: float = 4.0
    
    # Sequence lengths
    num_past_frames: int = 9
    num_future_frames: int = 8
    
    # Action configuration (v2.0: 25 dimensions matching paper's R^25)
    action_dim: int = 25
    
    # Tokenizer configuration - v2.0 format
    # IMPORTANT: Must match full model vocab_size for correct decoding
    vocab_size: int = 65536  # per-factor vocabulary size (2^16) - matches full model
    num_factored_vocabs: int = 3  # number of factored tokens per position
    spatial_size: int = 32  # 32×32 tokens per frame
    temporal_compression: int = 8  # Cosmos DV temporal compression
    frames_per_clip: int = 17  # frames per temporally-compressed clip
    
    # Parameter sharing
    shared_layers_start: int = 2
    
    # Reduced training configuration
    learning_rate: float = 1e-4
    warmup_steps: int = 10
    max_steps: int = 100
    batch_size: int = 8
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
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 10
