"""Configuration for Masked-HWM model (v2.0 dataset).

Cosmos Tokenization Details:
- Tokenizer: Cosmos DV 8×8×8 (Discrete Video)
- Quantization: Finite Scalar Quantization (FSQ)
- FSQ Levels: [8, 8, 8, 5, 5, 5]
- Actual Codebook Size: 8 × 8 × 8 × 5 × 5 × 5 = 64,000 tokens per factor
- Model Vocab Size: 65,536 (2^16) per factor (implementation convenience)
- Factorized: 3 separate FSQ quantizations per spatial position
- Spatial Compression: 256×256 → 32×32 (8× compression)
- Temporal Compression: 8× (17 frames per clip)
"""

from dataclasses import dataclass
from typing import Optional

# Cosmos FSQ Quantization Constants
COSMOS_FSQ_LEVELS = [8, 8, 8, 5, 5, 5]  # FSQ quantization levels
COSMOS_FSQ_CODEBOOK_SIZE = 8 * 8 * 8 * 5 * 5 * 5  # 64,000 actual tokens
COSMOS_VOCAB_SIZE = 65536  # Model vocab size (2^16, power-of-2 for convenience)
COSMOS_NUM_FACTORS = 3  # Number of factorized tokens per spatial position


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
    # 
    # Cosmos FSQ Quantization Details:
    # - Uses Finite Scalar Quantization (FSQ) with levels [8, 8, 8, 5, 5, 5]
    # - Codebook size: 8 × 8 × 8 × 5 × 5 × 5 = 64,000 tokens
    # - Model uses vocab_size=65536 (2^16) for implementation convenience:
    #   * Power-of-2 boundary (efficient for hardware/software)
    #   * Close approximation to 64K (only 1,536 more tokens)
    #   * Provides headroom for potential expansion
    #   * Fits in uint16 range (0-65535) for efficient storage
    #
    # Factorized representation:
    # - 3 separate FSQ quantizations per spatial position (factorized encoder)
    # - Each factor has its own ~64K codebook, stored as [num_clips, 3, 32, 32]
    # - Total effective vocabulary: ~64K^3 ≈ 281 trillion combinations
    # - We embed each factor separately and sum the embeddings
    vocab_size: int = 65536  # per-factor vocabulary size (2^16, matches Cosmos FSQ) - see COSMOS_VOCAB_SIZE constant
    num_factored_vocabs: int = 3  # number of factored tokens per position (Cosmos factorized encoder) - see COSMOS_NUM_FACTORS constant
    spatial_size: int = 32  # 32×32 tokens per frame (256×256 → 32×32 spatial compression)
    temporal_compression: int = 8  # Cosmos DV temporal compression factor
    frames_per_clip: int = 17  # frames per temporally-compressed clip (Cosmos DV 8×8×8)
    
    # Parameter sharing
    shared_layers_start: int = 4  # First 4 layers unshared, remaining shared
    
    # Training configuration (per paper)
    learning_rate: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 60000
    batch_size: int = 16  # Per paper: single NVIDIA A6000 with batch size 16
    gradient_accumulation_steps: int = 1
    
    # Masking configuration
    # Per paper: "inject Copilot-4D style noise with a uniform corruption rate
    # sampled from U(0, ρ_max), where ρ_max = 0.2"
    # This applies random token replacements to both past and future latents
    # before masking (which is only applied to future latents)
    max_corrupt_rate: float = 0.2  # ρ_max = 0.2 (per paper)
    mask_token_id: int = 65536  # vocab_size (used as mask token for each factor) - matches COSMOS_VOCAB_SIZE
    
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
    test_data_dir: Optional[str] = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/test_v2.0"
    
    # Memory optimization
    use_gradient_checkpointing: bool = False  # Recompute activations during backward (~50% memory reduction)
    use_flash_attn: bool = True  # Auto-detect and use Flash Attention or xformers if available
    mixed_precision: Optional[str] = None  # None, 'fp16', or 'bf16' (handled by Accelerator in train.py)

    # Other
    seed: int = 42
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100


@dataclass
class MaskedHWMRTX4090Config(MaskedHWMConfig):
    """Configuration optimized for RTX 4090 (24GB VRAM).

    Maintains the full model architecture from the paper (24 layers, 512 dim)
    while optimizing memory usage to fit in 24GB VRAM:
    - Reduced batch size from 16 to 8
    - Gradient accumulation of 2 steps (effective batch size = 16)
    - Gradient checkpointing enabled
    - Mixed precision training (bf16)
    - Flash Attention / xformers enabled

    This configuration preserves the paper's effective batch size and model
    quality while fitting within GPU memory constraints.

    Memory savings breakdown:
    - Gradient checkpointing: ~50% activation memory reduction
    - Mixed precision (bf16): ~40-50% memory reduction
    - Flash Attention: ~60-70% attention memory reduction
    - Combined: Can reduce peak memory by 60-80%
    """

    # Memory-optimized training (maintains effective batch size = 16)
    batch_size: int = 8  # Reduced from 16 to fit in 24GB
    gradient_accumulation_steps: int = 2  # Effective batch size = 8 * 2 = 16

    # Memory optimization techniques
    use_gradient_checkpointing: bool = True
    use_flash_attn: bool = True  # Auto-detect Flash Attention / xformers
    mixed_precision: str = "bf16"  # Use bf16 for RTX 4090

    # Model architecture (UNCHANGED - maintains paper quality)
    # num_layers: int = 24
    # num_heads: int = 8
    # d_model: int = 512
    # mlp_hidden: int = 2048
