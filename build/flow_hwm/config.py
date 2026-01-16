"""Configuration for Flow Matching Humanoid World Model.

Flow-HWM operates in continuous latent space (pre-quantization) rather than
discrete tokens like Masked-HWM. It uses flow matching to learn a velocity
field that transforms Gaussian noise into target video latents.

Key differences from Masked-HWM:
- Input: Continuous latents (B, latent_dim, T, H, W) vs discrete tokens
- Loss: MSE on velocity field vs cross-entropy on tokens
- Conditioning: Time embedding (sinusoidal + AdaLN) vs mask tokens
- Inference: Euler ODE solver vs iterative MaskGIT unmasking
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FlowHWMConfig:
    """Configuration for Flow Matching Humanoid World Model.

    Attributes:
        Model Architecture:
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            d_model: Model dimension (hidden size)
            mlp_hidden: MLP hidden dimension
            mlp_ratio: MLP expansion ratio (alternative to mlp_hidden)

        Latent Dimensions (Cosmos continuous latents):
            latent_dim: Number of latent channels from Cosmos encoder
            latent_spatial: Spatial size of latent (16x16)
            temporal_tokens_per_clip: Temporal tokens per clip (CV 8x => 3 for 17 frames)
            patch_size_spatial: Spatial patch size (p_lw)
            patch_size_temporal: Temporal patch size (p_t)
            num_past_clips: Number of past video clips for context
            num_future_clips: Number of future clips to predict

        Action Configuration:
            action_dim: Action space dimension (25 for humanoid robot)
            frames_per_clip: Frames per temporally-compressed clip (17)

        Flow Matching Parameters:
            sigma_min: Small constant for path construction (prevents collapse)
            cfg_drop_prob: Probability of dropping conditioning (classifier-free guidance)
            cfg_scale: Guidance scale at inference time
            num_euler_steps: Number of Euler ODE integration steps

        Training Configuration:
            learning_rate: Initial learning rate
            warmup_steps: Linear warmup steps
            max_steps: Maximum training steps
            batch_size: Per-device batch size
            gradient_accumulation_steps: Steps to accumulate gradients

        Memory Optimization:
            use_gradient_checkpointing: Enable gradient checkpointing
            mixed_precision: Mixed precision mode ('bf16', 'fp16', or None)

        Paths:
            tokenizer_checkpoint_dir: Path to Cosmos tokenizer (DV decoder)
            cv_tokenizer_checkpoint_dir: Path to Cosmos CV tokenizer (encoder)
            train_data_dir: Training data directory
            val_data_dir: Validation data directory
            test_data_dir: Test data directory
    """

    # Model architecture
    num_layers: int = 24
    num_heads: int = 8
    d_model: int = 512
    mlp_hidden: int = 2048
    mlp_ratio: float = 4.0

    # Latent dimensions (Cosmos continuous latents)
    # Cosmos CV 8×16×16: 256×256 → 16×16 spatial, 8× temporal compression
    latent_dim: int = 16  # Cosmos encoder output channels
    latent_spatial: int = 16  # 16×16 spatial tokens per frame
    temporal_tokens_per_clip: int = 3  # 17 frames -> 3 temporal tokens (ceil(17/8))
    patch_size_spatial: int = 2  # p_lw
    patch_size_temporal: int = 1  # p_t
    num_past_clips: int = 2  # Past context clips
    num_future_clips: int = 1  # Future clips to predict

    # Action configuration (same as Masked-HWM)
    # 25-dimensional action space (paper's R^25):
    # - Indices 0-20: Joint positions (21 dims)
    # - Index 21: Left hand closure
    # - Index 22: Right hand closure
    # - Index 23: Linear Velocity
    # - Index 24: Angular Velocity
    action_dim: int = 25
    frames_per_clip: int = 17  # Dataset clips aligned to 17-frame DV segments

    # Flow matching parameters
    sigma_min: float = 0.0  # Must be 0 for Cosmos tokenizer (decoder is extremely sensitive to token values)
    noise_std: float = 0.5  # Noise std to match data distribution (latents have std ~0.48)
    cfg_drop_prob: float = 0.1  # Classifier-free guidance dropout probability
    cfg_scale: float = 3.0  # Guidance scale at inference (1.0 = no guidance)
    num_euler_steps: int = 50  # ODE integration steps for inference

    # Training configuration
    learning_rate: float = 3e-5
    warmup_steps: int = 0
    max_steps: int = 60000
    batch_size: int = 4  # Per-device batch size for 24GB GPU
    gradient_accumulation_steps: int = 4  # Effective batch size = 16

    # Memory optimization for 24GB GPU
    use_gradient_checkpointing: bool = True
    mixed_precision: Optional[str] = "bf16"  # None, 'fp16', or 'bf16'

    # Attention configuration
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    mlp_drop: float = 0.0

    # Initialization
    init_std: float = 0.02

    # Dataset/tokenizer behavior
    use_cv_tokenizer: bool = True

    # Paths (v2.0 dataset)
    # DV checkpoint path (decoder used to reconstruct RGB before CV encoding)
    tokenizer_checkpoint_dir: str = "/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer"
    # CV checkpoint path (continuous encoder for CV 8×16×16)
    cv_tokenizer_checkpoint_dir: str = "/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer/Continuous_video"
    train_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0"
    val_data_dir: str = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0"
    test_data_dir: Optional[str] = "/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/test_v2.0"

    # Other
    seed: int = 42
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.num_heads

    @property
    def total_clips(self) -> int:
        """Total number of clips (past + future)."""
        return self.num_past_clips + self.num_future_clips

    @property
    def total_video_tokens(self) -> int:
        """Total video tokens across all clips."""
        return self.total_clips * self.temporal_tokens_per_clip

    @property
    def past_video_tokens(self) -> int:
        """Video tokens for past clips."""
        return self.num_past_clips * self.temporal_tokens_per_clip

    @property
    def future_video_tokens(self) -> int:
        """Video tokens for future clips."""
        return self.num_future_clips * self.temporal_tokens_per_clip

    @property
    def token_spatial(self) -> int:
        """Spatial tokens per frame after patching."""
        return self.latent_spatial // self.patch_size_spatial

    @property
    def past_frames(self) -> int:
        """Total past frames (for action sequences)."""
        return self.num_past_clips * self.frames_per_clip

    @property
    def future_frames(self) -> int:
        """Total future frames (for action sequences)."""
        return self.num_future_clips * self.frames_per_clip


@dataclass
class FlowHWMConfigMedium(FlowHWMConfig):
    """Medium configuration for training with limited GPU memory.

    Balanced model size between full and small:
    - 12 layers instead of 24 (half of full)
    - 384 dim instead of 512 (3/4 of full)
    - 6 heads instead of 8
    - 1536 mlp instead of 2048 (3/4 of full)
    """

    num_layers: int = 12
    num_heads: int = 6
    d_model: int = 384
    mlp_hidden: int = 1536

    # Slightly reduced batch size to help with memory
    batch_size: int = 3
    gradient_accumulation_steps: int = 4  # Effective batch size = 12


@dataclass
class FlowHWMConfigSmall(FlowHWMConfig):
    """Smaller configuration for testing and debugging.

    Reduced model size for quick iteration:
    - 8 layers instead of 24
    - 256 dim instead of 512
    - 4 heads instead of 8
    """

    num_layers: int = 8
    num_heads: int = 4
    d_model: int = 256
    mlp_hidden: int = 1024

    # Can use larger batch size with smaller model
    batch_size: int = 8
    gradient_accumulation_steps: int = 2


@dataclass
class FlowHWMConfigTest(FlowHWMConfig):
    """Minimal configuration for unit tests.

    Very small model for fast tests:
    - 2 layers
    - 64 dim
    - 2 heads
    """

    num_layers: int = 2
    num_heads: int = 2
    d_model: int = 64
    mlp_hidden: int = 256

    # Minimal data
    num_past_clips: int = 1
    num_future_clips: int = 1

    # Fast training
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_steps: int = 100

    # Disable checkpointing for small model
    use_gradient_checkpointing: bool = False
