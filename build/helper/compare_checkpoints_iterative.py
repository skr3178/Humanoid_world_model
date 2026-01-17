#!/usr/bin/env python3
"""Compare multiple checkpoints using iterative decoding (MaskGIT style).

This script generates comparison videos showing how model quality improves
across training checkpoints, using the proper iterative decoding strategy
with K iterations.

Similar to step 6 in debug_masked_hwm_videos.py, but runs across multiple
checkpoints to visualize training progression.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import HumanoidWorldModelDataset
from masked_hwm.config import MaskedHWMConfig
from masked_hwm.config_test import MaskedHWMTestConfig
from masked_hwm.model import MaskedHWM

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("WARNING: imageio not available. Install with: pip install imageio imageio-ffmpeg")


# =============================================================================
# HELPER FUNCTIONS (from debug_masked_hwm_videos.py)
# =============================================================================

def load_cosmos_decoder(tokenizer_dir: str, device: str = "cuda"):
    """Load Cosmos DV decoder for discrete tokens."""
    try:
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    except ImportError:
        raise ImportError(
            "cosmos_tokenizer not found. Please activate the 'cosmos-tokenizer' conda environment."
        )

    decoder_path = f"{tokenizer_dir}/decoder.jit"
    if not Path(decoder_path).exists():
        raise FileNotFoundError(f"Decoder not found: {decoder_path}")

    decoder = CausalVideoTokenizer(
        checkpoint_dec=decoder_path,
        device=device,
        dtype="bfloat16" if device == "cuda" else "float32"
    )
    return decoder


def decode_video_from_tokens(tokens: torch.Tensor, decoder, device: str) -> torch.Tensor:
    """Decode factorized tokens to video frames.

    Args:
        tokens: (T_clips, 3, H, W) - factorized discrete tokens per clip
        decoder: Cosmos DV decoder

    Returns:
        video: (3, T_total, H_img, W_img) - decoded video tensor
    """
    T_clips = tokens.shape[0]
    decoded_clips = []

    for t in range(T_clips):
        clip_tokens = tokens[t:t+1]  # (1, 3, H, W)
        with torch.no_grad():
            decoded = decoder.decode(clip_tokens.to(device))  # (1, 3, 17, H_img, W_img)
            decoded_clips.append(decoded)

    video = torch.cat(decoded_clips, dim=2)  # (1, 3, T_total, H_img, W_img)
    return video[0]  # (3, T_total, H_img, W_img)


def tensor_to_numpy_video(video: torch.Tensor) -> np.ndarray:
    """Convert video tensor to numpy array for saving.

    Args:
        video: (3, T, H, W) float tensor in [-1, 1]

    Returns:
        video_np: (T, H, W, 3) uint8 numpy array
    """
    video = video.float().cpu()
    video = (video + 1) / 2  # [-1, 1] -> [0, 1]
    video = video.clamp(0, 1)
    video = (video * 255).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).numpy()  # (T, H, W, 3)
    return video


def save_video(video: np.ndarray, output_path: Path, fps: int = 30) -> None:
    """Save numpy video to file."""
    if not HAS_IMAGEIO:
        print(f"  ERROR: Cannot save video - imageio not available")
        return

    imageio.mimsave(
        str(output_path),
        video,
        fps=fps,
        codec='libx264',
        quality=8
    )
    print(f"  Saved: {output_path}")


def add_label_bar(frame: np.ndarray, label: str, bar_height: int = 40) -> np.ndarray:
    """Add a label bar to the top of a frame."""
    from PIL import Image, ImageDraw, ImageFont

    H, W, C = frame.shape

    # Create label bar
    bar_img = Image.new('RGB', (W, bar_height), color='black')
    draw = ImageDraw.Draw(bar_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (W - text_width) // 2
    text_y = (bar_height - (bbox[3] - bbox[1])) // 2
    draw.text((text_x, text_y), label, fill='white', font=font)

    return np.concatenate([np.array(bar_img), frame], axis=0)


def create_side_by_side_video(
    videos: List[np.ndarray],
    labels: List[str],
    output_path: Path,
    fps: int = 30,
    add_labels: bool = True
) -> None:
    """Create horizontally concatenated comparison video."""
    T = min(v.shape[0] for v in videos)

    combined_frames = []
    for t in range(T):
        row = []
        for video, label in zip(videos, labels):
            frame = video[t]
            if add_labels:
                frame = add_label_bar(frame, label)
            row.append(frame)
        combined_frames.append(np.concatenate(row, axis=1))

    save_video(np.stack(combined_frames, axis=0), output_path, fps=fps)


# =============================================================================
# ITERATIVE DECODING (from debug_masked_hwm_videos.py)
# =============================================================================

def iterative_decode(
    model: MaskedHWM,
    video_past: torch.Tensor,
    video_future_shape: tuple,
    actions_past: torch.Tensor,
    actions_future: torch.Tensor,
    config,
    device: str,
    K: int = 16,
    verbose: bool = False
) -> torch.Tensor:
    """Perform MaskGIT-style iterative decoding.

    Args:
        model: MaskedHWM model
        video_past: (B, num_factors, T_clips, H, W) past video tokens
        video_future_shape: Shape for future tokens
        actions_past: (B, T, action_dim) past actions
        actions_future: (B, T, action_dim) future actions
        config: Model config
        device: Device
        K: Number of decoding iterations
        verbose: Print progress

    Returns:
        decoded_tokens: (B, num_factors, T_clips, H, W)
    """
    B, num_factors, T_clips, H, W = video_future_shape
    num_tokens = T_clips * H * W

    # Start with all tokens masked
    current_tokens = torch.full(video_future_shape, config.mask_token_id, device=device)
    mask = torch.ones(B, num_factors, T_clips, H, W, dtype=torch.bool, device=device)

    def cosine_schedule(ratio):
        return np.cos(ratio * np.pi / 2)

    for iteration in range(K):
        ratio = (iteration + 1) / K
        num_to_unmask = int(num_tokens * (1 - cosine_schedule(ratio)))

        if verbose:
            print(f"  Iteration {iteration+1}/{K}: unmasking {num_to_unmask}/{num_tokens} tokens")

        # Forward pass
        with torch.no_grad():
            logits = model(
                v_p_tokens=video_past,
                v_f_tokens=current_tokens,
                a_p=actions_past,
                a_f=actions_future
            )  # (num_factors, B, T_clips, H, W, vocab_size)

        # Temperature annealing
        temperature = 1.2 * (1.0 - ratio) + 0.8 * ratio
        logits_scaled = logits / max(temperature, 0.1)
        probs = torch.softmax(logits_scaled, dim=-1)

        # Categorical sampling
        original_shape = logits_scaled.shape[:-1]
        vocab_size = logits_scaled.shape[-1]
        flat_logits = logits_scaled.reshape(-1, vocab_size)
        flat_probs = torch.softmax(flat_logits, dim=-1)
        predicted_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        predicted = predicted_flat.reshape(original_shape)

        # Get confidence
        confidence = torch.gather(probs, dim=-1, index=predicted.unsqueeze(-1)).squeeze(-1)

        # Gumbel noise for stochastic masking
        choice_temperature = 4.5 * (1.0 - ratio)
        if choice_temperature > 0:
            gumbel_noise = torch.distributions.gumbel.Gumbel(0, 1).sample(confidence.shape).to(device)
            confidence = torch.log(confidence + 1e-8) + choice_temperature * gumbel_noise

        # Unmask most confident predictions
        for f in range(num_factors):
            factor_mask = mask[0, f].flatten().clone()
            factor_conf = confidence[f, 0].flatten()
            factor_pred = predicted[f, 0].flatten()

            masked_indices = factor_mask.nonzero(as_tuple=True)[0]
            if len(masked_indices) == 0:
                continue

            masked_conf = factor_conf[masked_indices]
            num_unmask_this = min(num_to_unmask, len(masked_indices))

            if num_unmask_this > 0:
                _, topk_local = masked_conf.topk(num_unmask_this)
                topk_indices = masked_indices[topk_local]

                current_flat = current_tokens[0, f].flatten().clone()
                current_flat[topk_indices] = factor_pred[topk_indices]
                current_tokens[0, f] = current_flat.view(T_clips, H, W)

                mask_flat = mask[0, f].flatten().clone()
                mask_flat[topk_indices] = False
                mask[0, f] = mask_flat.view(T_clips, H, W)

    return current_tokens


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare checkpoints using iterative decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare default checkpoints (every 10k steps):
  python compare_checkpoints_iterative.py \\
      --checkpoints_dir /path/to/checkpoints_rtx3060 \\
      --data_dir /path/to/val_v2.0

  # Compare specific checkpoints:
  python compare_checkpoints_iterative.py \\
      --checkpoints_dir /path/to/checkpoints \\
      --checkpoint_steps 10000 30000 60000 \\
      --data_dir /path/to/val_v2.0

  # Use more decoding iterations:
  python compare_checkpoints_iterative.py \\
      --checkpoints_dir /path/to/checkpoints \\
      --K 16 \\
      --data_dir /path/to/val_v2.0
        """
    )

    parser.add_argument("--checkpoints_dir", type=str, required=True,
                       help="Directory containing checkpoint folders")
    parser.add_argument("--checkpoint_steps", type=int, nargs="+", default=None,
                       help="Specific checkpoint steps to compare (default: auto-select)")
    parser.add_argument("--data_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0",
                       help="Path to dataset directory")
    parser.add_argument("--tokenizer_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer",
                       help="Path to Cosmos tokenizer directory")
    parser.add_argument("--output_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/videos/checkpoint_comparison_iterative",
                       help="Output directory for comparison videos")
    parser.add_argument("--K", type=int, default=16,
                       help="Number of decoding iterations (default: 16)")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Sample index to use")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--max_checkpoints", type=int, default=6,
                       help="Maximum number of checkpoints to compare (for video width)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(args.checkpoints_dir)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Decoding iterations (K): {args.K}")

    # Find checkpoints
    if args.checkpoint_steps:
        checkpoint_steps = args.checkpoint_steps
    else:
        # Auto-select: find all checkpoints and pick evenly spaced ones
        all_folders = sorted([
            d for d in checkpoints_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ], key=lambda x: int(x.name.split("-")[1]))

        if len(all_folders) <= args.max_checkpoints:
            checkpoint_steps = [int(d.name.split("-")[1]) for d in all_folders]
        else:
            # Pick evenly spaced + first and last
            indices = np.linspace(0, len(all_folders) - 1, args.max_checkpoints).astype(int)
            checkpoint_steps = [int(all_folders[i].name.split("-")[1]) for i in indices]

    print(f"\nSelected checkpoint steps: {checkpoint_steps}")

    # Verify checkpoints exist
    valid_checkpoints = []
    for step in checkpoint_steps:
        cp_path = checkpoints_dir / f"checkpoint-{step}" / "pytorch_model.bin"
        if cp_path.exists():
            valid_checkpoints.append((step, cp_path))
        else:
            print(f"  WARNING: checkpoint-{step} not found, skipping")

    if not valid_checkpoints:
        print("ERROR: No valid checkpoints found")
        return

    print(f"Valid checkpoints: {[s for s, _ in valid_checkpoints]}")

    # Load config from first checkpoint
    first_cp = torch.load(str(valid_checkpoints[0][1]), map_location=device, weights_only=False)
    if "config" in first_cp:
        config = first_cp["config"]
    else:
        config = MaskedHWMConfig()

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = HumanoidWorldModelDataset(
        data_dir=args.data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1
    )

    if len(dataset) == 0 or args.sample_idx >= len(dataset):
        print("ERROR: Dataset empty or sample index out of range")
        return

    # Load decoder
    print("Loading Cosmos decoder...")
    decoder = load_cosmos_decoder(args.tokenizer_dir, device)

    # Get sample
    sample = dataset[args.sample_idx]

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        return torch.from_numpy(x.copy())

    video_past = to_tensor(sample["video_past"]).unsqueeze(0).to(device)
    video_future_gt = to_tensor(sample["video_future"]).unsqueeze(0).to(device)
    actions_past = to_tensor(sample["actions_past"]).unsqueeze(0).to(device).to(torch.bfloat16)
    actions_future = to_tensor(sample["actions_future"]).unsqueeze(0).to(device).to(torch.bfloat16)

    # Reshape to model format
    video_past = video_past.permute(0, 2, 1, 3, 4)  # (B, 3, T_clips, H, W)
    video_future_gt = video_future_gt.permute(0, 2, 1, 3, 4)

    video_past = video_past % config.vocab_size
    video_future_gt = video_future_gt % config.vocab_size

    print(f"\nSample {args.sample_idx}:")
    print(f"  Future shape: {video_future_gt.shape}")

    # Decode ground truth
    print("\nDecoding ground truth...")
    gt_for_decode = video_future_gt.permute(0, 2, 1, 3, 4)  # (B, T_clips, 3, H, W)
    gt_video = decode_video_from_tokens(gt_for_decode[0], decoder, device)
    gt_video_np = tensor_to_numpy_video(gt_video)
    save_video(gt_video_np, output_dir / "ground_truth.mp4")

    all_videos = [gt_video_np]
    all_labels = ["GT"]

    # Process each checkpoint
    for step, cp_path in valid_checkpoints:
        print(f"\n{'='*50}")
        print(f"Processing checkpoint-{step}")
        print(f"{'='*50}")

        # Load checkpoint
        checkpoint = torch.load(str(cp_path), map_location=device, weights_only=False)
        if "config" in checkpoint:
            cp_config = checkpoint["config"]
        else:
            cp_config = config

        model = MaskedHWM(cp_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).to(torch.bfloat16)
        model.eval()

        actual_step = checkpoint.get("global_step", step)
        print(f"  Loaded model (step {actual_step})")

        # Generate with iterative decoding
        print(f"  Generating with K={args.K} iterations...")
        pred_tokens = iterative_decode(
            model=model,
            video_past=video_past,
            video_future_shape=video_future_gt.shape,
            actions_past=actions_past,
            actions_future=actions_future,
            config=cp_config,
            device=device,
            K=args.K,
            verbose=True
        )

        # Decode to video
        pred_for_decode = pred_tokens.permute(0, 2, 1, 3, 4)
        pred_video = decode_video_from_tokens(pred_for_decode[0], decoder, device)
        pred_video_np = tensor_to_numpy_video(pred_video)

        # Save individual video
        save_video(pred_video_np, output_dir / f"checkpoint_{step}.mp4")

        all_videos.append(pred_video_np)
        all_labels.append(f"Step {actual_step}")

        # Compute accuracy
        correct = (pred_tokens == video_future_gt).float().mean().item()
        print(f"  Token accuracy vs GT: {correct*100:.1f}%")

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Create comparison video
    print(f"\n{'='*50}")
    print("Creating comparison video...")
    print(f"{'='*50}")

    create_side_by_side_video(
        all_videos,
        all_labels,
        output_dir / "checkpoint_comparison.mp4"
    )

    print(f"\n{'='*60}")
    print("Comparison complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    # List generated videos
    print("\nGenerated videos:")
    for f in sorted(output_dir.glob("*.mp4")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
