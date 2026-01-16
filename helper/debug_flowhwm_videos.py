#!/usr/bin/env python3
"""Debug script for FlowHWM video generation.

Two verification passes to diagnose reconstruction quality issues:

1. CHECKPOINT DECODE PASS:
   - Decode ground-truth tokens directly (bypasses model entirely)
   - Run model inference and decode predicted latents
   - Side-by-side comparison videos

2. NOISE SCHEDULE PASS:
   - Visualize X_t along the flow path at multiple timesteps
   - Shows progression from noise (t=0) to target (t=1)
   - Helps verify the latent-to-token conversion is correct

Usage:
    python helper/debug_flowhwm_videos.py --num_samples 2 --num_noise_steps 6

    # With custom checkpoint
    python helper/debug_flowhwm_videos.py --checkpoint /path/to/model.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add build directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "build"))

from flow_hwm.config import FlowHWMConfig, FlowHWMConfigMedium
from flow_hwm.dataset_latent import FlowHWMDataset
from flow_hwm.flow_matching import construct_flow_path, sample_noise
from flow_hwm.inference import generate_video_latents, load_model_from_checkpoint
from data.dataset import HumanoidWorldModelDataset

try:
    import imageio
except ImportError:
    print("ERROR: imageio not available. Install with: pip install imageio imageio-ffmpeg")
    sys.exit(1)


def latents_to_tokens(latents: torch.Tensor) -> torch.Tensor:
    """Convert continuous latents back to discrete tokens for Cosmos decoder.

    The FlowHWMDataset creates latents by:
        latents = tokens / 65535.0 * 2 - 1  (normalized to [-1, 1])

    We reverse this:
        tokens = (latents + 1) / 2 * 65535

    Args:
        latents: Continuous latents (B, C, T, H, W) where C >= 3
                 Values expected in [-1, 1] range

    Returns:
        tokens: Discrete tokens (B, 3, T, H, W) as int64 in [0, 65535]
    """
    B, C, T, H, W = latents.shape

    # Take first 3 channels (or average groups if C > 3)
    if C == 3:
        latents_3ch = latents
    elif C > 3:
        # For latent_dim=16, the dataset repeats each factor ~5 times
        # Take channels 0, 5, 10 (or similar spacing) to get the 3 factors
        # Or simply take first 3 channels as they correspond to factor 0
        # Actually the dataset interleaves: [f0, f0, f0, f0, f0, f1, f1, f1, f1, f1, f2, f2, f2, f2, f2, f0]
        # For latent_dim=16 with 3 factors: 16//3 = 5 repeats, remainder 1
        repeats = C // 3
        # Average each group of repeats back to single factor
        latents_3ch = torch.stack([
            latents[:, i*repeats:(i+1)*repeats].mean(dim=1) for i in range(3)
        ], dim=1)
    else:
        # C < 3, repeat to get 3 channels
        latents_3ch = latents.repeat(1, 3, 1, 1, 1)[:, :3]

    # Clamp to expected range
    latents_3ch = torch.clamp(latents_3ch, -1.0, 1.0)

    # Reverse normalization: [-1, 1] -> [0, 65535]
    tokens = (latents_3ch + 1.0) / 2.0 * 65535.0
    tokens = tokens.round().long()
    tokens = torch.clamp(tokens, 0, 65535)

    return tokens


def decode_tokens_to_video(
    tokens: torch.Tensor,
    decoder,
    device: str,
) -> np.ndarray:
    """Decode factored tokens to RGB video frames.

    Args:
        tokens: (B, 3, T, H, W) int64 tokens where T is number of clips
        decoder: Cosmos CausalVideoTokenizer decoder
        device: Device string

    Returns:
        video: (B, total_frames, H_out, W_out, 3) uint8 RGB frames
    """
    B, num_factors, T, H, W = tokens.shape

    all_videos = []
    for b in range(B):
        clip_frames = []
        for t in range(T):
            # Get single clip tokens: (3, H, W) -> (1, 3, H, W)
            clip_tokens = tokens[b:b+1, :, t, :, :].to(device)

            with torch.no_grad():
                # Decoder expects (B, 3, H, W) and returns (B, 3, T_out, H_out, W_out)
                decoded = decoder.decode(clip_tokens)
                # Convert to numpy: (T_out, H_out, W_out, 3)
                decoded = decoded[0].permute(1, 2, 3, 0)  # (T, H, W, C)
                decoded = ((decoded + 1) / 2 * 255).clamp(0, 255).byte()
                decoded = decoded.cpu().numpy()
                clip_frames.append(decoded)

        # Concatenate all clips: (total_frames, H, W, 3)
        video = np.concatenate(clip_frames, axis=0)
        all_videos.append(video)

    return np.stack(all_videos, axis=0)


def decode_latents_to_video(
    latents: torch.Tensor,
    decoder,
    device: str,
) -> np.ndarray:
    """Decode continuous latents to RGB video.

    Args:
        latents: (B, C, T, H, W) continuous latents in [-1, 1]
        decoder: Cosmos decoder
        device: Device string

    Returns:
        video: (B, total_frames, H_out, W_out, 3) uint8 RGB
    """
    tokens = latents_to_tokens(latents)
    return decode_tokens_to_video(tokens, decoder, device)


def add_text_overlay(frame: np.ndarray, text: str, position: str = "top") -> np.ndarray:
    """Add text overlay to frame using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Position
    x = (frame.shape[1] - text_w) // 2
    y = 10 if position == "top" else frame.shape[0] - text_h - 10

    # Draw background rectangle
    draw.rectangle([x-5, y-5, x+text_w+5, y+text_h+5], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return np.array(img)


def save_video(frames: np.ndarray, output_path: Path, fps: int = 30) -> None:
    """Save numpy array as MP4 video with high quality encoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use imageio-ffmpeg with high quality settings
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",  # Required for compatibility
        output_params=[
            "-crf", "18",  # Lower = higher quality (0-51, 18 is visually lossless)
            "-preset", "slow",  # Better compression
        ],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved: {output_path}")


def create_side_by_side(
    left_video: np.ndarray,
    right_video: np.ndarray,
    left_label: str = "Ground Truth",
    right_label: str = "Prediction",
) -> np.ndarray:
    """Create side-by-side comparison video."""
    def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        if (frame.shape[1], frame.shape[0]) == size:
            return frame
        try:
            from PIL import Image
        except ImportError:
            return frame
        return np.array(Image.fromarray(frame).resize(size, resample=Image.BICUBIC))

    T = min(len(left_video), len(right_video))
    frames = []
    target_w = max(left_video[0].shape[1], right_video[0].shape[1])
    target_h = max(left_video[0].shape[0], right_video[0].shape[0])
    target_size = (target_w, target_h)

    for t in range(T):
        left_frame = resize_frame(left_video[t], target_size)
        right_frame = resize_frame(right_video[t], target_size)
        left_frame = add_text_overlay(left_frame.copy(), left_label)
        right_frame = add_text_overlay(right_frame.copy(), right_label)
        combined = np.concatenate([left_frame, right_frame], axis=1)
        frames.append(combined)

    return np.stack(frames, axis=0)


def create_noise_schedule_grid(
    target_latents: torch.Tensor,
    decoder,
    device: str,
    sigma_min: float,
    noise_std: float,
    num_timesteps: int,
    seed: int,
) -> np.ndarray:
    """Create video showing X_t at multiple timesteps.

    Shows the flow path from noise (t=0) to target (t=1).

    Args:
        target_latents: (1, C, T, H, W) target X_1
        decoder: Cosmos decoder
        device: Device string
        sigma_min: Flow matching sigma_min
        noise_std: Std of noise distribution (should match data std)
        num_timesteps: Number of timesteps to visualize
        seed: Random seed for noise

    Returns:
        grid_video: (num_frames, H, W*num_timesteps, 3) video grid
    """
    torch.manual_seed(seed)
    x0 = sample_noise(target_latents.shape, device, std=noise_std)

    t_values = torch.linspace(0.0, 1.0, num_timesteps, device=device)

    all_videos = []
    labels = []

    for t_val in t_values:
        # Construct X_t along flow path
        x_t = construct_flow_path(x0, target_latents, t_val, sigma_min)

        # Decode to video
        video = decode_latents_to_video(x_t, decoder, device)[0]  # (T, H, W, 3)
        all_videos.append(video)
        labels.append(f"t={float(t_val):.2f}")

    # Find minimum frame count
    min_frames = min(v.shape[0] for v in all_videos)

    # Create grid: stack horizontally for each frame
    grid_frames = []
    for frame_idx in range(min_frames):
        row = []
        for video, label in zip(all_videos, labels):
            frame = add_text_overlay(video[frame_idx].copy(), label)
            row.append(frame)
        grid_frames.append(np.concatenate(row, axis=1))

    return np.stack(grid_frames, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Debug FlowHWM video generation")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/checkpoints_flow_hwm_medium/model-60000.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/videos/debug_flowhwm",
        help="Output directory",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer",
        help="Path to Cosmos tokenizer",
    )
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to process")
    parser.add_argument("--num_steps", type=int, default=50, help="Euler integration steps")
    parser.add_argument("--cfg_scale", type=float, default=1.5, help="CFG scale")
    parser.add_argument("--num_noise_steps", type=int, default=6, help="Timesteps for noise schedule")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--max_shards", type=int, default=3, help="Max data shards to load")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_model", action="store_true", help="Skip model inference (GT only)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Output directory: {output_dir}")

    # Determine config from checkpoint path
    use_medium = "medium" in args.checkpoint.lower()
    config = FlowHWMConfigMedium() if use_medium else FlowHWMConfig()
    print(f"Config: {'FlowHWMConfigMedium' if use_medium else 'FlowHWMConfig'}")
    print(f"  latent_dim={config.latent_dim}, num_past_clips={config.num_past_clips}, num_future_clips={config.num_future_clips}")

    # Load Cosmos decoder
    print("\nLoading Cosmos decoder...")
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    decoder = CausalVideoTokenizer(
        checkpoint_dec=f"{args.tokenizer_dir}/decoder.jit",
        device=args.device,
        dtype="bfloat16" if args.device == "cuda" else "float32",
    )
    print("Decoder loaded successfully")

    # Load datasets
    print(f"\nLoading datasets from {args.data_dir}...")

    # Base dataset for original tokens
    base_dataset = HumanoidWorldModelDataset(
        data_dir=args.data_dir,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        use_factored_tokens=True,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=args.max_shards,
    )

    # Flow dataset for continuous latents
    flow_dataset = FlowHWMDataset(
        data_dir=args.data_dir,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        latent_dim=config.latent_dim,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=args.max_shards,
    )

    print(f"Base dataset: {len(base_dataset)} samples")
    print(f"Flow dataset: {len(flow_dataset)} samples")

    if len(base_dataset) == 0:
        print("ERROR: No samples in dataset!")
        return

    # Load model if not skipping
    model = None
    model_dtype = torch.float32
    if not args.skip_model:
        print(f"\nLoading model from {args.checkpoint}...")
        if Path(args.checkpoint).exists():
            model = load_model_from_checkpoint(args.checkpoint, config, args.device)
            if args.device.startswith("cuda"):
                model = model.to(dtype=torch.bfloat16)
                model_dtype = torch.bfloat16
            else:
                model_dtype = next(model.parameters()).dtype
            print(f"Model loaded: {model.get_num_params():,} parameters")
        else:
            print(f"WARNING: Checkpoint not found: {args.checkpoint}")
            print("Skipping model inference")

    # Process samples
    for sample_idx in range(min(args.num_samples, len(base_dataset))):
        print(f"\n{'='*70}")
        print(f"SAMPLE {sample_idx}")
        print(f"{'='*70}")

        sample_dir = output_dir / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # PASS 1: Ground Truth Decoding (from original tokens)
        # =====================================================================
        print("\n[PASS 1] Decoding ground truth from original tokens...")

        base_sample = base_dataset[sample_idx]
        gt_tokens_future = base_sample["video_future"]  # (T_clips, 3, H, W)
        gt_tokens_past = base_sample["video_past"]  # (T_clips, 3, H, W)

        # Reshape for decoder: (1, 3, T_clips, H, W)
        gt_tokens_future = gt_tokens_future.unsqueeze(0).permute(0, 2, 1, 3, 4)
        gt_tokens_past = gt_tokens_past.unsqueeze(0).permute(0, 2, 1, 3, 4)

        print(f"  GT future tokens shape: {gt_tokens_future.shape}")
        print(f"  GT future tokens range: [{gt_tokens_future.min()}, {gt_tokens_future.max()}]")

        # Decode ground truth
        gt_video_future = decode_tokens_to_video(gt_tokens_future, decoder, args.device)[0]
        gt_video_past = decode_tokens_to_video(gt_tokens_past, decoder, args.device)[0]

        print(f"  Decoded GT future video: {gt_video_future.shape}")
        print(f"  Decoded GT past video: {gt_video_past.shape}")

        # Concatenate past + future for full context video
        gt_video_full = np.concatenate([gt_video_past, gt_video_future], axis=0)

        save_video(gt_video_future, sample_dir / "gt_future_from_tokens.mp4", fps=args.fps)
        save_video(gt_video_past, sample_dir / "gt_past_from_tokens.mp4", fps=args.fps)
        save_video(gt_video_full, sample_dir / "gt_full_sequence.mp4", fps=args.fps)

        # =====================================================================
        # PASS 1b: Ground Truth via Latent Round-trip
        # =====================================================================
        print("\n[PASS 1b] Testing latent round-trip (tokens -> latents -> tokens)...")

        flow_sample = flow_dataset[sample_idx]
        latent_future = flow_sample["latent_future"].unsqueeze(0).to(args.device)  # (1, C, T, H, W)
        latent_past = flow_sample["latent_past"].unsqueeze(0).to(args.device)

        print(f"  Latent future shape: {latent_future.shape}")
        print(f"  Latent future range: [{latent_future.min():.3f}, {latent_future.max():.3f}]")

        # Decode latents (tests the latent-to-token conversion)
        roundtrip_video = decode_latents_to_video(latent_future, decoder, args.device)[0]

        print(f"  Round-trip video shape: {roundtrip_video.shape}")
        save_video(roundtrip_video, sample_dir / "gt_future_via_latent_roundtrip.mp4", fps=args.fps)

        # Compare: side-by-side of direct tokens vs latent round-trip
        if len(gt_video_future) == len(roundtrip_video):
            comparison = create_side_by_side(
                gt_video_future, roundtrip_video,
                "Direct Tokens", "Latent Round-trip"
            )
            save_video(comparison, sample_dir / "comparison_tokens_vs_roundtrip.mp4", fps=args.fps)

        # =====================================================================
        # PASS 2: Noise Schedule Visualization
        # =====================================================================
        print(f"\n[PASS 2] Creating noise schedule visualization ({args.num_noise_steps} timesteps)...")

        # Get noise_std from config (default 0.5 for backwards compatibility)
        noise_std = getattr(config, 'noise_std', 0.5)
        print(f"  Using noise_std={noise_std}")

        noise_grid = create_noise_schedule_grid(
            target_latents=latent_future,
            decoder=decoder,
            device=args.device,
            sigma_min=config.sigma_min,
            noise_std=noise_std,
            num_timesteps=args.num_noise_steps,
            seed=args.seed + sample_idx,
        )

        print(f"  Noise schedule grid shape: {noise_grid.shape}")
        save_video(noise_grid, sample_dir / "noise_schedule.mp4", fps=args.fps)

        # =====================================================================
        # PASS 3: Model Prediction (if model loaded)
        # =====================================================================
        if model is not None:
            print(f"\n[PASS 3] Running model inference ({args.num_steps} Euler steps, CFG={args.cfg_scale})...")

            latent_past = latent_past.to(dtype=model_dtype)
            a_p = flow_sample["actions_past"].unsqueeze(0).to(args.device, dtype=model_dtype)
            a_f = flow_sample["actions_future"].unsqueeze(0).to(args.device, dtype=model_dtype)

            print(f"  Conditioning: v_p={latent_past.shape}, a_p={a_p.shape}, a_f={a_f.shape}")

            with torch.no_grad():
                torch.manual_seed(args.seed + sample_idx)
                pred_latents = generate_video_latents(
                    model,
                    latent_past,
                    a_p,
                    a_f,
                    num_steps=args.num_steps,
                    cfg_scale=args.cfg_scale,
                    verbose=True,
                )

            print(f"  Predicted latents shape: {pred_latents.shape}")
            print(f"  Predicted latents range: [{pred_latents.min():.3f}, {pred_latents.max():.3f}]")

            # Decode prediction
            pred_video = decode_latents_to_video(pred_latents, decoder, args.device)[0]

            print(f"  Predicted video shape: {pred_video.shape}")
            save_video(pred_video, sample_dir / "prediction.mp4", fps=args.fps)

            # Side-by-side: GT vs Prediction
            min_len = min(len(gt_video_future), len(pred_video))
            comparison = create_side_by_side(
                gt_video_future[:min_len],
                pred_video[:min_len],
                "Ground Truth", "Prediction"
            )
            save_video(comparison, sample_dir / "comparison_gt_vs_pred.mp4", fps=args.fps)

            # Also compare latent statistics
            print(f"\n  Latent Statistics:")
            print(f"    GT latent   - mean: {latent_future.mean():.4f}, std: {latent_future.std():.4f}")
            print(f"    Pred latent - mean: {pred_latents.mean():.4f}, std: {pred_latents.std():.4f}")
            print(f"    MSE between GT and Pred latents: {((latent_future - pred_latents)**2).mean():.4f}")

    print(f"\n{'='*70}")
    print(f"Done! All outputs saved to: {output_dir}")
    print(f"{'='*70}")

    # Summary of what to check
    print("\nDEBUGGING CHECKLIST:")
    print("1. gt_future_from_tokens.mp4 - Should be CLEAN (verifies decoder works)")
    print("2. gt_future_via_latent_roundtrip.mp4 - Should match #1 (verifies latent conversion)")
    print("3. comparison_tokens_vs_roundtrip.mp4 - Should look identical (if not, latent conversion broken)")
    print("4. noise_schedule.mp4 - Should show smooth transition from noise to clean")
    print("5. prediction.mp4 - Model output (compare to ground truth)")
    print("6. comparison_gt_vs_pred.mp4 - Side-by-side for easy comparison")


if __name__ == "__main__":
    main()
