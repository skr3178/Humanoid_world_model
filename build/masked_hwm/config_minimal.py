"""Minimal configuration for testing model architecture on small GPUs.

This config is designed to:
- Fit in 4-6GB GPU memory
- Run quickly for testing (100 steps)
- Verify architecture works correctly
- Show loss decreasing

NOTE: Uses reduced sequence length (1 past + 1 future clip) for testing.
For production, use config_12gb.py with full sequence length.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MaskedHWMMinimalConfig:
    """Minimal configuration for testing on small GPUs.
    
    Reduced to fit in 4-6GB VRAM:
    - 4 layers, 128 dim, 4 heads
    - 1 past clip + 1 future clip (reduced from 2+1)
    - batch_size=1 with grad_accum=16 (effective batch=16)
    - vocab_size=65536 (must match tokenizer, cannot reduce)
    """
    
    # Minimal model architecture
    num_layers: int = 4
    num_heads: int = 4
    d_model: int = 128
    mlp_hidden: int = 512
    mlp_ratio: float = 4.0
    
    # Reduced sequence length for testing (1 past + 1 future clip)
    num_past_clips: int = 1  # Reduced from 2 for testing
    num_future_clips: int = 1  # Same as paper
    # Keep frame counts for action processing
    num_past_frames: int = 17  # 1 clip * 17 frames/clip
    num_future_frames: int = 17  # 1 clip * 17 frames/clip
    
    # Action configuration (unchanged - matches paper's R^25)
    action_dim: int = 25
    
    # Tokenizer configuration (unchanged - critical for decoding)
    vocab_size: int = 65536  # per-factor vocabulary size (2^16) - MUST match Cosmos
    num_factored_vocabs: int = 3  # number of factored tokens per position
    spatial_size: int = 32  # 32×32 tokens per frame
    temporal_compression: int = 8  # Cosmos DV temporal compression
    frames_per_clip: int = 17  # frames per temporally-compressed clip
    
    # Parameter sharing
    shared_layers_start: int = 2  # First 2 layers unshared, remaining shared
    
    # Minimal training configuration
    learning_rate: float = 1e-4
    warmup_steps: int = 10
    max_steps: int = 100  # Quick test
    batch_size: int = 1  # Minimal batch size
    gradient_accumulation_steps: int = 16  # Effective batch size = 16
    
    # Masking configuration
    max_corrupt_rate: float = 0.2  # Per paper
    mask_token_id: int = 65536  # vocab_size
    
    # Attention configuration
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    mlp_drop: float = 0.0
    use_rope: bool = True
    use_stream_type_emb: bool = False
    
    # Initialization
    init_std: float = 0.02
    
    # Paths (v2.0 dataset)
    tokenizer_checkpoint_dir: str = "/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer"
    train_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
    val_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
    test_data_dir: Optional[str] = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/test_v2.0"
    
    # Other
    seed: int = 42
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 10
