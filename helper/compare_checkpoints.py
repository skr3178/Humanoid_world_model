#!/usr/bin/env python3
"""Generate side-by-side comparison videos of model predictions vs ground truth.

Compares multiple checkpoint predictions against ground truth videos
to visualize training progression.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from masked_hwm.config_test import MaskedHWMTestConfig
from masked_hwm.config import MaskedHWMConfig
from masked_hwm.model import MaskedHWM
from data.dataset import HumanoidWorldModelDataset
from data.collator import MaskedHWMCollator

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("WARNING: imageio not available. Install with: pip install imageio imageio-ffmpeg")


def load_checkpoint(checkpoint_path: str, config, device: str = "cuda") -> Tuple[MaskedHWM, dict]:
    """Load model from checkpoint."""
    print(f"  Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use config from checkpoint if available
    if "config" in checkpoint:
        config = checkpoint["config"]
    
    model = MaskedHWM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    step = checkpoint.get("global_step", "unknown")
    return model, {"step": step, "config": config}


def sample_tokens_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Sample tokens from logits for each factor.
    
    Args:
        logits: (num_factors, B, T_f, H, W, vocab_size)
        temperature: Sampling temperature
        
    Returns:
        tokens: (B, num_factors, T_f, H, W)
    """
    num_factors, B, T_f, H, W, vocab_size = logits.shape
    
    sampled_tokens = []
    for f in range(num_factors):
        factor_logits = logits[f]  # (B, T_f, H, W, vocab_size)
        logits_flat = rearrange(factor_logits, 'b t h w v -> (b t h w) v')
        
        # Sample with temperature
        probs = F.softmax(logits_flat / temperature, dim=-1)
        tokens_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        tokens = tokens_flat.view(B, T_f, H, W)
        sampled_tokens.append(tokens)
    
    return torch.stack(sampled_tokens, dim=1)


@torch.no_grad()
def generate_predictions(
    model: MaskedHWM,
    batch: Dict[str, torch.Tensor],
    config,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate predictions for a batch.
    
    Returns:
        predicted_tokens: (B, num_factors, T_f, H, W)
        actual_tokens: (B, num_factors, T_f, H, W)
    """
    # Prepare inputs
    video_past = batch["video_past"].to(device) % config.vocab_size
    video_future_input = batch["video_future"].to(device).clone()
    
    # Mask all future tokens for generation
    video_future_input.fill_(config.mask_token_id)
    
    # Forward pass
    logits = model(
        v_p_tokens=video_past,
        v_f_tokens=video_future_input,
        a_p=batch["actions_past"].to(device),
        a_f=batch["actions_future"].to(device),
    )
    
    # Sample tokens
    predicted_tokens = sample_tokens_from_logits(logits, temperature=1.0)
    actual_tokens = batch["video_future_labels"].to(device)
    
    return predicted_tokens, actual_tokens


def decode_tokens_to_video(
    tokens: torch.Tensor,
    decoder,
    device: str = "cuda"
) -> np.ndarray:
    """Decode factorized tokens to video frames.
    
    Args:
        tokens: (B, num_factors, T, H, W) - factorized tokens
        decoder: Cosmos decoder
        
    Returns:
        video: (B, T*17, H_img, W_img, 3) - decoded video as uint8 numpy array
    """
    from cosmos_tokenizer.utils import tensor2numpy
    
    B, num_factors, T, H, W = tokens.shape
    
    decoded_frames = []
    for t in range(T):
        # Get tokens for this temporal position: (B, 3, H, W)
        clip_tokens = tokens[:, :, t, :, :].to(device)
        
        with torch.no_grad():
            decoded_clip = decoder.decode(clip_tokens).float()
            decoded_frames.append(decoded_clip)
    
    # Concatenate: (B, 3, T*17, H_img, W_img)
    video = torch.cat(decoded_frames, dim=2)
    
    # Convert using official method: (B, T, H, W, 3) uint8
    video_np = tensor2numpy(video)
    
    return video_np


def create_side_by_side_video(
    gt_video: np.ndarray,
    pred_video: np.ndarray,
    output_path: str,
    fps: int = 30
):
    """Create side-by-side comparison video.
    
    Args:
        gt_video: (T, H, W, 3) ground truth video
        pred_video: (T, H, W, 3) predicted video
        output_path: Output path for video
        fps: Frames per second
    """
    if not HAS_IMAGEIO:
        print(f"  ERROR: Cannot save video - imageio not available")
        return
    
    T = min(gt_video.shape[0], pred_video.shape[0])
    H, W = gt_video.shape[1], gt_video.shape[2]
    
    # Create side-by-side frames
    combined_frames = []
    for t in range(T):
        # Stack horizontally: GT | Predicted
        combined = np.concatenate([gt_video[t], pred_video[t]], axis=1)
        combined_frames.append(combined)
    
    combined_frames = np.stack(combined_frames, axis=0)
    
    imageio.mimsave(
        str(output_path),
        combined_frames,
        fps=fps,
        codec='libx264',
        quality=8
    )
    print(f"  Saved: {output_path}")


def create_multi_checkpoint_comparison(
    gt_video: np.ndarray,
    checkpoint_videos: Dict[str, np.ndarray],
    output_path: str,
    fps: int = 30
):
    """Create video with ground truth and multiple checkpoint predictions.
    
    Layout: GT | Ckpt1 | Ckpt2 | ... (horizontally concatenated)
    
    Args:
        gt_video: (T, H, W, 3) ground truth video
        checkpoint_videos: Dict mapping checkpoint name to (T, H, W, 3) video
        output_path: Output path for video
        fps: Frames per second
    """
    if not HAS_IMAGEIO:
        print(f"  ERROR: Cannot save video - imageio not available")
        return
    
    T = gt_video.shape[0]
    H, W = gt_video.shape[1], gt_video.shape[2]
    
    # Ensure all videos have same length
    for name, video in checkpoint_videos.items():
        T = min(T, video.shape[0])
    
    combined_frames = []
    for t in range(T):
        # Start with ground truth
        frame_parts = [gt_video[t]]
        
        # Add each checkpoint prediction
        for name in sorted(checkpoint_videos.keys()):
            frame_parts.append(checkpoint_videos[name][t])
        
        # Concatenate horizontally
        combined = np.concatenate(frame_parts, axis=1)
        combined_frames.append(combined)
    
    combined_frames = np.stack(combined_frames, axis=0)
    
    imageio.mimsave(
        str(output_path),
        combined_frames,
        fps=fps,
        codec='libx264',
        quality=8
    )
    print(f"  Saved multi-checkpoint comparison: {output_path}")


def add_labels_to_video(
    video: np.ndarray,
    label: str,
    position: str = "top"
) -> np.ndarray:
    """Add text label to video frames.
    
    Args:
        video: (T, H, W, 3) video
        label: Text to add
        position: "top" or "bottom"
        
    Returns:
        Video with label bar added
    """
    from PIL import Image, ImageDraw, ImageFont
    
    T, H, W, C = video.shape
    label_height = 40
    
    # Create labeled frames
    labeled_frames = []
    for t in range(T):
        # Create image from frame
        frame = Image.fromarray(video[t])
        
        # Create new image with space for label
        if position == "top":
            new_frame = Image.new('RGB', (W, H + label_height), color='black')
            new_frame.paste(frame, (0, label_height))
            text_y = 10
        else:
            new_frame = Image.new('RGB', (W, H + label_height), color='black')
            new_frame.paste(frame, (0, 0))
            text_y = H + 10
        
        # Add text
        draw = ImageDraw.Draw(new_frame)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Center text
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (W - text_width) // 2
        draw.text((text_x, text_y), label, fill='white', font=font)
        
        labeled_frames.append(np.array(new_frame))
    
    return np.stack(labeled_frames, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoint predictions with ground truth")
    parser.add_argument("--checkpoints_dir", type=str, required=True,
                       help="Directory containing checkpoint folders (e.g., checkpoint-10, checkpoint-25, etc.)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./comparison_videos",
                       help="Output directory for comparison videos")
    parser.add_argument("--ground_truth_dir", type=str, default=None,
                       help="Directory with existing ground truth videos (optional)")
    parser.add_argument("--tokenizer_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer",
                       help="Path to Cosmos tokenizer directory")
    parser.add_argument("--num_samples", type=int, default=2,
                       help="Number of samples to compare")
    parser.add_argument("--use_test_config", action="store_true",
                       help="Use test config (for checkpoints trained with test config)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--add_labels", action="store_true", default=True,
                       help="Add text labels to videos")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(args.checkpoints_dir)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Find all checkpoints
    checkpoint_folders = sorted([
        d for d in checkpoints_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ], key=lambda x: (
        0 if "final" in x.name else int(x.name.split("-")[1])
    ))
    
    if not checkpoint_folders:
        print(f"ERROR: No checkpoint folders found in {checkpoints_dir}")
        return
    
    print(f"Found {len(checkpoint_folders)} checkpoints:")
    for cp in checkpoint_folders:
        print(f"  - {cp.name}")
    
    # Load config
    if args.use_test_config:
        config = MaskedHWMTestConfig()
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
    
    if len(dataset) == 0:
        print(f"ERROR: Dataset is empty")
        return
    
    collator = MaskedHWMCollator(config)
    
    # Load decoder
    print("Loading Cosmos decoder...")
    try:
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    except ImportError:
        print("ERROR: cosmos_tokenizer not found. Cannot decode video.")
        return
    
    decoder = CausalVideoTokenizer(
        checkpoint_dec=f"{args.tokenizer_dir}/decoder.jit",
        device=device,
        dtype="bfloat16"
    )
    
    # Process each sample
    for sample_idx in range(min(args.num_samples, len(dataset))):
        print(f"\n{'='*60}")
        print(f"Processing Sample {sample_idx}")
        print(f"{'='*60}")
        
        # Get sample and collate
        sample = dataset[sample_idx]
        batch = collator([sample])
        
        # Decode ground truth
        print("Decoding ground truth...")
        gt_tokens = batch["video_future_labels"]  # (1, num_factors, T_f, H, W)
        gt_video = decode_tokens_to_video(gt_tokens, decoder, device)[0]  # (T, H, W, 3)
        
        if args.add_labels:
            gt_video = add_labels_to_video(gt_video, "Ground Truth")
        
        # Generate predictions from each checkpoint
        checkpoint_videos = {}
        
        for cp_folder in checkpoint_folders:
            cp_name = cp_folder.name.replace("checkpoint-", "")
            cp_path = cp_folder / "pytorch_model.bin"
            
            if not cp_path.exists():
                print(f"  WARNING: {cp_path} not found, skipping...")
                continue
            
            print(f"\nGenerating from {cp_folder.name}...")
            
            # Load model
            model, cp_info = load_checkpoint(str(cp_path), config, device)
            
            # Generate predictions
            pred_tokens, _ = generate_predictions(model, batch, cp_info["config"], device)
            
            # Decode to video
            pred_video = decode_tokens_to_video(pred_tokens, decoder, device)[0]
            
            if args.add_labels:
                label = f"Step {cp_info['step']}" if cp_info['step'] != 'unknown' else f"Ckpt-{cp_name}"
                pred_video = add_labels_to_video(pred_video, label)
            
            checkpoint_videos[cp_name] = pred_video
            
            # Also save individual comparison video (GT vs this checkpoint)
            individual_output = output_dir / f"sample_{sample_idx}_gt_vs_ckpt{cp_name}.mp4"
            create_side_by_side_video(gt_video, pred_video, str(individual_output))
            
            # Clean up model to save memory
            del model
            torch.cuda.empty_cache()
        
        # Create multi-checkpoint comparison video
        if checkpoint_videos:
            multi_output = output_dir / f"sample_{sample_idx}_all_checkpoints.mp4"
            create_multi_checkpoint_comparison(gt_video, checkpoint_videos, str(multi_output))
    
    print(f"\n{'='*60}")
    print(f"Comparison complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.mp4")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
