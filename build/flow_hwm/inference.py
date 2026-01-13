"""Inference script for Flow-HWM.

Generates future video latents using Euler ODE integration with
classifier-free guidance (CFG).

The generation process:
1. Start from Gaussian noise X_0 ~ N(0, I)
2. Integrate the learned velocity field from t=0 to t=1
3. X_{t+dt} = X_t + dt * velocity(X_t, t)
4. Final X_1 is the generated video latent

Classifier-free guidance enhances generation quality:
    v = v_uncond + cfg_scale * (v_cond - v_uncond)

Usage:
    python inference.py --checkpoint /path/to/checkpoint.pt --output_dir ./outputs

    # With different CFG scale
    python inference.py --checkpoint /path/to/checkpoint.pt --cfg_scale 2.0

    # More integration steps for better quality
    python inference.py --checkpoint /path/to/checkpoint.pt --num_steps 100
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_hwm.config import FlowHWMConfig, FlowHWMConfigSmall, FlowHWMConfigTest
from flow_hwm.model import FlowHWM, create_flow_hwm
from flow_hwm.flow_matching import sample_noise, euler_step


@torch.no_grad()
def generate_video_latents(
    model: FlowHWM,
    v_p: torch.Tensor,
    a_p: torch.Tensor,
    a_f: torch.Tensor,
    num_steps: int = 50,
    cfg_scale: float = 1.5,
    verbose: bool = False,
) -> torch.Tensor:
    """Generate future video latents using Euler ODE integration.

    Integrates the learned velocity field from t=0 (noise) to t=1 (data):
        dX_t/dt = u_theta(X_t, P, t)

    Uses classifier-free guidance for improved generation quality.

    Args:
        model: Trained FlowHWM model
        v_p: Past video latent for conditioning, shape (B, C, T_p, H, W)
        a_p: Past actions, shape (B, T_p_frames, action_dim)
        a_f: Future actions, shape (B, T_f_frames, action_dim)
        num_steps: Number of Euler integration steps
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
        verbose: Print progress

    Returns:
        Generated video latent X_1, shape (B, C, T_f, H, W)
    """
    model.eval()
    device = v_p.device
    B = v_p.shape[0]
    config = model.config

    # Determine output shape from config
    C = config.latent_dim
    T_f = config.num_future_clips
    H = W = config.latent_spatial

    # Start from Gaussian noise X_0
    x = sample_noise((B, C, T_f, H, W), device)

    # Time step size
    dt = 1.0 / num_steps

    # Euler integration from t=0 to t=1
    for step in range(num_steps):
        t = torch.full((B,), step / num_steps, device=device)

        if cfg_scale != 1.0:
            # Classifier-free guidance: compute both conditional and unconditional
            v_cond = model(x, v_p, a_p, a_f, t)

            # Unconditional: zero out conditioning
            zeros_v_p = torch.zeros_like(v_p)
            zeros_a_p = torch.zeros_like(a_p)
            v_uncond = model(x, zeros_v_p, zeros_a_p, a_f, t)

            # CFG formula: v = v_uncond + scale * (v_cond - v_uncond)
            velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            # No guidance, just use conditional prediction
            velocity = model(x, v_p, a_p, a_f, t)

        # Euler step: X_{t+dt} = X_t + dt * velocity
        x = euler_step(x, velocity, dt)

        if verbose and (step + 1) % 10 == 0:
            print(f"  Integration step {step + 1}/{num_steps}")

    return x


@torch.no_grad()
def generate_multiple_futures(
    model: FlowHWM,
    v_p: torch.Tensor,
    a_p: torch.Tensor,
    a_f: torch.Tensor,
    num_samples: int = 4,
    num_steps: int = 50,
    cfg_scale: float = 1.5,
) -> torch.Tensor:
    """Generate multiple possible futures for the same context.

    Since flow matching is stochastic (starts from different noise samples),
    we can generate diverse predictions by running inference multiple times.

    Args:
        model: Trained FlowHWM model
        v_p: Past video latent (B, C, T_p, H, W) - will be repeated
        a_p: Past actions (B, T_p_frames, action_dim)
        a_f: Future actions (B, T_f_frames, action_dim)
        num_samples: Number of diverse samples to generate
        num_steps: Euler integration steps
        cfg_scale: CFG scale

    Returns:
        Multiple generated latents, shape (num_samples * B, C, T_f, H, W)
    """
    all_samples = []

    for i in range(num_samples):
        sample = generate_video_latents(
            model, v_p, a_p, a_f,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
        )
        all_samples.append(sample)

    return torch.cat(all_samples, dim=0)


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[FlowHWMConfig] = None,
    device: str = "cuda",
) -> FlowHWM:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration (if None, inferred from checkpoint)
        device: Device to load model on

    Returns:
        Loaded FlowHWM model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to get config from checkpoint
    if config is None:
        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            print("Warning: No config in checkpoint, using default FlowHWMConfig")
            config = FlowHWMConfig()

    # Create model
    model = create_flow_hwm(config)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def quantize_latents_to_tokens(
    latents: torch.Tensor,
    num_levels: int = 65536,
) -> torch.Tensor:
    """Quantize continuous latents to discrete tokens.

    This is needed if you want to decode the generated latents using
    the Cosmos tokenizer, which expects discrete indices.

    Args:
        latents: Continuous latents (B, C, T, H, W)
        num_levels: Number of quantization levels

    Returns:
        Discrete token indices (B, C, T, H, W) as int64
    """
    # Normalize to [0, 1]
    latents_min = latents.min()
    latents_max = latents.max()
    latents_norm = (latents - latents_min) / (latents_max - latents_min + 1e-8)

    # Quantize to discrete levels
    tokens = (latents_norm * (num_levels - 1)).round().long()
    tokens = tokens.clamp(0, num_levels - 1)

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Generate videos with Flow-HWM")

    # Model and checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--use_small_config",
        action="store_true",
        help="Use small model config",
    )
    parser.add_argument(
        "--use_test_config",
        action="store_true",
        help="Use test model config",
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0",
        help="Data directory for conditioning context",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Sample index from dataset",
    )

    # Generation parameters
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of Euler integration steps",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_latents",
        help="Output directory for generated latents",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    # Select config
    if args.use_test_config:
        config = FlowHWMConfigTest()
    elif args.use_small_config:
        config = FlowHWMConfigSmall()
    else:
        config = FlowHWMConfig()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, config, args.device)
    print(f"Model loaded: {model.get_num_params():,} parameters")

    # Load conditioning data
    print(f"Loading conditioning data from {args.data_dir}...")
    from flow_hwm.dataset_latent import FlowHWMDataset

    dataset = FlowHWMDataset(
        data_dir=args.data_dir,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        latent_dim=config.latent_dim,
    )

    # Get sample
    sample = dataset[args.sample_idx]
    v_p = sample["latent_past"].unsqueeze(0).to(args.device)  # (1, C, T_p, H, W)
    a_p = sample["actions_past"].unsqueeze(0).to(args.device)  # (1, T_p_frames, action_dim)
    a_f = sample["actions_future"].unsqueeze(0).to(args.device)  # (1, T_f_frames, action_dim)

    print(f"Conditioning shapes:")
    print(f"  v_p: {v_p.shape}")
    print(f"  a_p: {a_p.shape}")
    print(f"  a_f: {a_f.shape}")

    # Generate
    print(f"\nGenerating {args.num_samples} sample(s) with {args.num_steps} steps, CFG={args.cfg_scale}...")

    if args.num_samples == 1:
        generated = generate_video_latents(
            model, v_p, a_p, a_f,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            verbose=True,
        )
    else:
        generated = generate_multiple_futures(
            model, v_p, a_p, a_f,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        )

    print(f"Generated latents shape: {generated.shape}")

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save generated latents
    output_path = output_dir / f"generated_sample_{args.sample_idx}.pt"
    torch.save({
        "generated_latents": generated.cpu(),
        "conditioning": {
            "v_p": v_p.cpu(),
            "a_p": a_p.cpu(),
            "a_f": a_f.cpu(),
        },
        "ground_truth": sample["latent_future"],
        "config": {
            "num_steps": args.num_steps,
            "cfg_scale": args.cfg_scale,
        },
    }, output_path)

    print(f"Saved generated latents to {output_path}")

    # Also save quantized tokens (for potential decoding)
    tokens = quantize_latents_to_tokens(generated)
    tokens_path = output_dir / f"tokens_sample_{args.sample_idx}.pt"
    torch.save(tokens.cpu(), tokens_path)
    print(f"Saved quantized tokens to {tokens_path}")


if __name__ == "__main__":
    main()
