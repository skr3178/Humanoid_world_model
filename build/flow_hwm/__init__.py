"""Flow Matching Humanoid World Model (Flow-HWM).

A continuous flow matching approach for video generation, operating in
continuous latent space rather than discrete tokens.

Architecture:
- Four-stream transformer: v_p (past video), v_f (future video),
  a_p (past actions), a_f (future actions)
- Time conditioning via sinusoidal embeddings + AdaLN
- RoPE: 3D for video tokens, 1D for action tokens
- Euler ODE solver for inference with classifier-free guidance

Usage:
    from flow_hwm import FlowHWMConfig, FlowHWM, create_flow_hwm

    config = FlowHWMConfig()
    model = create_flow_hwm(config)

    # Training
    from flow_hwm.flow_matching import construct_flow_path, compute_target_velocity

    # Inference
    from flow_hwm.inference import generate_video_latents
"""

from .config import FlowHWMConfig, FlowHWMConfigSmall, FlowHWMConfigTest
from .model import FlowHWM, create_flow_hwm
from .time_embedding import TimeEmbedding, SinusoidalTimestepEmbedding
from .ada_ln import AdaLN, AdaLNZero, FeedForwardScale
from .rope_3d import RoPE3D, RoPE3DVideo
from .transformer import FlowTransformerBlock, FlowTransformer
from .flow_matching import (
    construct_flow_path,
    compute_target_velocity,
    flow_matching_loss,
    sample_timesteps,
    sample_noise,
    euler_step,
    euler_integrate,
    FlowMatchingTrainer,
)

__all__ = [
    # Config
    "FlowHWMConfig",
    "FlowHWMConfigSmall",
    "FlowHWMConfigTest",
    # Model
    "FlowHWM",
    "create_flow_hwm",
    # Components
    "TimeEmbedding",
    "SinusoidalTimestepEmbedding",
    "AdaLN",
    "AdaLNZero",
    "FeedForwardScale",
    "RoPE3D",
    "RoPE3DVideo",
    "FlowTransformerBlock",
    "FlowTransformer",
    # Flow matching utilities
    "construct_flow_path",
    "compute_target_velocity",
    "flow_matching_loss",
    "sample_timesteps",
    "sample_noise",
    "euler_step",
    "euler_integrate",
    "FlowMatchingTrainer",
]
