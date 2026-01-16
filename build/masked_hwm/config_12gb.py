"""Configuration for medium-to-small model size (~11GB GPU).

Reduced model size optimized for ~11GB GPUs:
- Layers: 6 (25% of full 24-layer model)
- Dimensions: 192 (37.5% of full 512-dim model)
- Heads: 6 (32 dim per head)
- MLP: 768 (scaled with d_model, mlp_ratio=4.0)
- Batch size: 1 with grad_accum=16 (effective batch=16)

This configuration provides a good balance between:
- Model capacity (enough to learn complex patterns)
- Memory efficiency (fits in ~11GB VRAM)
- Training speed (maintains effective batch size of 16)

Maintains critical settings:
- vocab_size: 65536 (required for Cosmos tokenizer)
- action_dim: 25 (matches paper's R^25)
- All other tokenizer settings unchanged
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MaskedHWM12GBConfig:
    """Medium-to-small configuration for ~11GB GPUs.
    
    Reduced model size that fits in ~11GB VRAM while maintaining:
    - Same vocabulary size (65536) for correct decoding
    - Same action space (25 dims)
    - Same architecture principles
    - Effective batch size of 16 (via batch_size=1, grad_accum=16)
    
    Model architecture: 6 layers, 192 dim, 6 heads
    Reduced to fit in ~11GB VRAM with batch_size=1 (backward pass needs gradient memory).
    """
    
    # Medium model architecture (reduced for ~11GB GPU)
    num_layers: int = 6  # Further reduced for memory efficiency
    num_heads: int = 6  # Reduced from 8 (192/6 = 32 dim per head)
    d_model: int = 192  # Reduced from 256 for memory efficiency
    mlp_hidden: int = 768  # Scaled with d_model, mlp_ratio=4.0
    mlp_ratio: float = 4.0
    
    # Sequence lengths - Per paper: "2 fully unmasked past latents and 1 partially masked future latent"
    # Each latent = 1 clip (17 frames after Cosmos 8x temporal compression)
    num_past_clips: int = 2  # 2 past latents (paper specification)
    num_future_clips: int = 1  # 1 future latent (paper specification)
    # Keep frame counts for action processing (17 frames per clip)
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
    
    # Parameter sharing (adjusted for reduced model)
    shared_layers_start: int = 2  # First 2 layers unshared, remaining 4 shared (6 layers total)
    
    # Training configuration (optimized for ~11GB GPU)
    learning_rate: float = 1e-4  # Slightly higher for smaller model
    warmup_steps: int = 100  # Standard warmup
    max_steps: int = 60000
    batch_size: int = 1  # Minimal batch size for ~11GB GPU (backward pass needs gradient memory)
    gradient_accumulation_steps: int = 16  # Effective batch size = 1 * 16 = 16
    
    # Masking configuration (per paper: max_corrupt_rate=0.2)
    max_corrupt_rate: float = 0.2  # Reduced from 0.5 to match paper
    mask_token_id: int = 65536  # vocab_size (valid: embeddings have vocab_size+1 entries)
    
    # Attention configuration (unchanged)
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    mlp_drop: float = 0.0
    use_rope: bool = True
    use_stream_type_emb: bool = False
    
    # Initialization (unchanged)
    init_std: float = 0.02
    
    # Memory optimization - bf16 enabled for efficiency
    use_gradient_checkpointing: bool = False  # Not yet implemented at model level
    mixed_precision: Optional[str] = "bf16"  # Half precision for memory savings
    
    # Paths (v2.0 dataset)
    tokenizer_checkpoint_dir: str = "/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer"
    train_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
    val_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
    test_data_dir: Optional[str] = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/test_v2.0"
    
    # Other
    seed: int = 42
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
