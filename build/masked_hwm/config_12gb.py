"""Configuration for 12GB GPU memory constraint.

Reduced model size to fit with batch_size=4:
- Layers: 6 (25% of full 24-layer model)
- Dimensions: 192 (37.5% of full 512-dim model)
- Heads: 4 (192/4 = 48 dim per head)
- MLP: 768 (scaled with d_model, mlp_ratio=4.0)
- Batch size: 4 with grad_accum=4 (effective batch=16)

Memory bottleneck analysis:
- Main issue: Activation memory from attention matrices (32×32)² = 1024×1024 per frame
- Model parameters: Embeddings ~100MB (still manageable)
- Dataset: NOT the issue (batched loading)

Maintains critical settings:
- vocab_size: 65536 (required for Cosmos tokenizer)
- action_dim: 25 (matches paper's R^25)
- All other tokenizer settings unchanged
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MaskedHWM12GBConfig:
    """Configuration for 12GB GPU (reduced to fit with batch_size=4).
    
    Reduced model size to fit in 12GB VRAM while maintaining:
    - Same vocabulary size (65536) for correct decoding
    - Same action space (25 dims)
    - Same architecture principles
    - Effective batch size of 16 (via batch_size=4, grad_accum=4)
    
    Memory bottleneck: Attention matrices (32×32)² = 1024×1024 per frame per layer
    Reduced to 6 layers and 192 dim to fit with batch_size=4.
    """
    
    # Reduced model architecture to fit 12GB VRAM with batch_size=4
    num_layers: int = 6  # Reduced from 10 (25% of full 24-layer model)
    num_heads: int = 4  # Reduced from 8 (192/4 = 48 dim per head)
    d_model: int = 192  # Reduced from 256 (37.5% of full 512-dim model)
    mlp_hidden: int = 768  # Reduced from 1024 (scaled with d_model, mlp_ratio=4.0)
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
    shared_layers_start: int = 2  # First 2 layers unshared, remaining shared (6 layers total)
    
    # Training configuration (adjusted for 12GB GPU)
    # Increased LR to compensate for loss averaging (divided by 3 factors)
    # Original 3e-5 * 3 = 9e-5, using 1e-4 for round number
    learning_rate: float = 1e-4  # Compensates for loss being averaged across 3 factors
    warmup_steps: int = 500  # Increased from 100 for more stable training
    max_steps: int = 60000
    batch_size: int = 4  # Increased from 1 for better gradient estimates
    gradient_accumulation_steps: int = 4  # Effective batch size = 4 * 4 = 16
    
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
