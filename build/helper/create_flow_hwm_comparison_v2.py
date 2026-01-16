#!/usr/bin/env python3
"""Create side-by-side comparison video for Flow-HWM - using original tokens for GT.

This version uses the original discrete tokens from the dataset for ground truth,
bypassing the continuous latent conversion to avoid any conversion errors.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flow_hwm.config import FlowHWMConfig, FlowHWMConfigMedium
from flow_hwm.model import FlowHWM, create_flow_hwm
from flow_hwm.inference import (
    load_model_from_checkpoint,
    generate_video_latents,
)
from flow_hwm.dataset_latent import FlowHWMDataset

# Import base dataset to get original tokens
sys.path.insert(0, "/media/skr/storage/robot_world/humanoid_wm/build")
from data.dataset import HumanoidWorldModelDataset

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("ERROR: imageio not available. Install with: pip install imageio imageio-ffmpeg")
    sys.exit(1)


def latents_to_factorized_tokens(latents: torch.Tensor) -> torch.Tensor:
    """Convert continuous latents to factorized token format for Cosmos decoder.
    
    NOTE: This method (taking first 3 channels) was tested and found to be the best
    approach among multiple alternatives. Test results: Mean Absolute Error ~3859 vs ~5318
    for other methods. See: build/helper/test_latent_to_token_conversion.py
    """
    B, C, T, H, W = latents.shape
    
    # CRITICAL: Clip latents to [-1, 1] first!
    latents = torch.clamp(latents, -1.0, 1.0)
    
    # Take first 3 channels (best method based on testing)
    if C >= 3:
        latents_3ch = latents[:, :3, :, :, :]
    else:
        latents_3ch = latents.repeat(1, (3 // C + 1), 1, 1, 1)[:, :3, :, :, :]
    
    # Reverse normalization
    latents_normalized = (latents_3ch + 1.0) / 2.0
    latents_normalized = torch.clamp(latents_normalized, 0.0, 1.0)
    
    # Quantize
    tokens_float = latents_normalized * 65535.0
    tokens = tokens_float.round().long()
    tokens = torch.clamp(tokens, 0, 65535)
    
    return tokens


def decode_tokens_to_video(tokens, decoder, device="cuda"):
    """Decode factorized tokens to video frames."""
    from cosmos_tokenizer.utils import tensor2numpy
    
    B, num_factors, T, H, W = tokens.shape
    
    decoded_frames = []
    for t in range(T):
        clip_tokens = tokens[:, :, t, :, :].to(device)
        with torch.no_grad():
            decoded_clip = decoder.decode(clip_tokens).float()
            decoded_frames.append(decoded_clip)
    
    video = torch.cat(decoded_frames, dim=2)
    video_np = tensor2numpy(video)
    
    return video_np


def decode_latents_to_video(latents: torch.Tensor, decoder, device: str = "cuda") -> np.ndarray:
    """Decode continuous latents to video frames using Cosmos decoder."""
    B, C, T, H, W = latents.shape
    
    # Convert to factorized tokens
    tokens = latents_to_factorized_tokens(latents)
    
    # Decode tokens
    return decode_tokens_to_video(tokens, decoder, device)


def add_label_bar(frame: np.ndarray, label: str, bar_height: int = 64) -> np.ndarray:
    """Add a label bar at the top of a frame."""
    H, W, C = frame.shape
    
    bar = np.zeros((bar_height, W, C), dtype=np.uint8)
    bar_img = Image.fromarray(bar)
    draw = ImageDraw.Draw(bar_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (W - text_width) // 2
    text_y = (bar_height - (bbox[3] - bbox[1])) // 2
    draw.text((text_x, text_y), label, fill='white', font=font)
    
    bar = np.array(bar_img)
    return np.concatenate([bar, frame], axis=0)


def create_side_by_side_video(
    gt_video: np.ndarray,
    pred_video: np.ndarray,
    output_path: str,
    fps: int = 30,
    add_labels: bool = True
):
    """Create side-by-side comparison video (GT | Predicted)."""
    T = min(gt_video.shape[0], pred_video.shape[0])

    # Get target size (use GT size as reference)
    target_h, target_w = gt_video.shape[1], gt_video.shape[2]

    combined_frames = []
    for t in range(T):
        gt_frame = gt_video[t]
        pred_frame = pred_video[t]

        # Resize prediction to match GT if needed
        if pred_frame.shape[0] != target_h or pred_frame.shape[1] != target_w:
            pred_pil = Image.fromarray(pred_frame)
            pred_pil = pred_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            pred_frame = np.array(pred_pil)

        if add_labels:
            gt_frame = add_label_bar(gt_frame, "Ground Truth")
            pred_frame = add_label_bar(pred_frame, "Model Prediction")

        combined = np.concatenate([gt_frame, pred_frame], axis=1)
        combined_frames.append(combined)
    
    combined_frames = np.stack(combined_frames, axis=0)
    
    imageio.mimsave(
        str(output_path),
        combined_frames,
        fps=fps,
        codec='libx264',
        quality=8
    )
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side comparison video for Flow-HWM")
    parser.add_argument("--checkpoint", type=str, 
                       default="/media/skr/storage/robot_world/humanoid_wm/checkpoints_flow_hwm_medium/model-60000.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0",
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, 
                       default="/media/skr/storage/robot_world/humanoid_wm/comparison_videos",
                       help="Output directory for comparison videos")
    parser.add_argument("--tokenizer_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer",
                       help="Path to Cosmos tokenizer directory")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to compare")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--num_steps", type=int, default=100,
                       help="Number of Euler integration steps (try more steps)")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                       help="Classifier-free guidance scale (try 1.0 for no guidance)")
    parser.add_argument("--use_medium_config", action="store_true",
                       help="Use FlowHWMConfigMedium instead of default")
    parser.add_argument("--no_labels", action="store_true",
                       help="Don't add text labels to video")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load config
    use_medium = args.use_medium_config or ("medium" in args.checkpoint.lower())
    if use_medium:
        config = FlowHWMConfigMedium()
        print("Using FlowHWMConfigMedium")
    else:
        config = FlowHWMConfig()
        print("Using FlowHWMConfig")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, config, device)
    # Cast model to bfloat16 for consistency with dataset outputs
    model = model.to(torch.bfloat16)
    print(f"Model loaded: {model.get_num_params():,} parameters")
    
    # Load base dataset to get original tokens
    print(f"Loading base dataset from {args.data_dir}...")
    base_dataset = HumanoidWorldModelDataset(
        data_dir=args.data_dir,
        num_past_frames=config.num_past_clips,
        num_future_frames=config.num_future_clips,
        use_factored_tokens=True,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1
    )
    
    # Also load FlowHWM dataset for conditioning
    flow_dataset = FlowHWMDataset(
        data_dir=args.data_dir,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        latent_dim=config.latent_dim,
        max_shards=1
    )
    
    print(f"Dataset loaded: {len(base_dataset)} samples")
    
    # Load decoder
    print("Loading Cosmos decoder...")
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    
    decoder = CausalVideoTokenizer(
        checkpoint_dec=f"{args.tokenizer_dir}/decoder.jit",
        device=device,
        dtype="bfloat16"
    )
    
    # Process samples
    for sample_idx in range(min(args.num_samples, len(base_dataset))):
        print(f"\n{'='*50}")
        print(f"Processing Sample {sample_idx}")
        print(f"{'='*50}")
        
        # Get original tokens from base dataset
        base_sample = base_dataset[sample_idx]
        gt_tokens = base_sample["video_future"]  # (T, 3, H, W)
        gt_tokens = gt_tokens.unsqueeze(0)  # (1, T, 3, H, W)
        gt_tokens = gt_tokens.permute(0, 2, 1, 3, 4)  # (1, 3, T, H, W)
        
        # Get conditioning from flow dataset
        flow_sample = flow_dataset[sample_idx]
        v_p = flow_sample["latent_past"].unsqueeze(0).to(device, dtype=torch.bfloat16)
        a_p = flow_sample["actions_past"].unsqueeze(0).to(device, dtype=torch.bfloat16)
        a_f = flow_sample["actions_future"].unsqueeze(0).to(device, dtype=torch.bfloat16)
        
        print(f"Conditioning shapes:")
        print(f"  v_p: {v_p.shape}")
        print(f"  a_p: {a_p.shape}")
        print(f"  a_f: {a_f.shape}")
        
        # Decode ground truth using original tokens
        print("Decoding ground truth from original tokens...")
        gt_video_np = decode_tokens_to_video(gt_tokens, decoder, device)
        gt_video = gt_video_np[0]
        
        # Generate predictions
        print(f"Generating predictions with {args.num_steps} steps, CFG={args.cfg_scale}...")
        with torch.no_grad():
            predicted_latents = generate_video_latents(
                model, v_p, a_p, a_f,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                verbose=True,
            )
        
        print(f"Generated latents shape: {predicted_latents.shape}")
        
        # Decode predictions
        print("Decoding predictions...")
        pred_video_np = decode_latents_to_video(predicted_latents, decoder, device)
        pred_video = pred_video_np[0]
        
        # Create comparison video
        output_path = output_dir / f"flow_hwm_v2_sample_{sample_idx}_comparison.mp4"
        print(f"Creating comparison video...")
        create_side_by_side_video(
            gt_video, pred_video, 
            str(output_path),
            fps=30,
            add_labels=not args.no_labels
        )
    
    print(f"\n{'='*50}")
    print(f"Done! Videos saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
