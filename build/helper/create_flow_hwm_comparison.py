#!/usr/bin/env python3
"""Create side-by-side comparison video for Flow-HWM model.

Loads checkpoint, generates predictions using Euler ODE integration,
and creates comparison videos with ground truth.
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
    quantize_latents_to_tokens,
)
from flow_hwm.dataset_latent import FlowHWMDataset

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("ERROR: imageio not available. Install with: pip install imageio imageio-ffmpeg")
    sys.exit(1)


def latents_to_factorized_tokens(latents: torch.Tensor) -> torch.Tensor:
    """Convert continuous latents to factorized token format for Cosmos decoder.
    
    FlowHWM produces latents in format (B, C, T, H, W) where C=latent_dim (16).
    Cosmos decoder expects factorized tokens in format (B, 3, H, W) per clip.
    
    The dataset converts tokens to latents by:
    1. Normalizing tokens [0, 65535] -> [0, 1] -> [-1, 1]
    2. Expanding 3 factors to C channels by repeating
    
    To reverse:
    1. Take first 3 channels (they correspond to the original 3 factors)
    2. CLIP to [-1, 1] first (model may generate outside this range)
    3. Reverse normalization: [-1, 1] -> [0, 1] -> [0, 65535]
    4. Quantize to integers
    
    Args:
        latents: Continuous latents (B, C, T, H, W) - may be outside [-1, 1]
    
    Returns:
        Factorized tokens (B, 3, T, H, W) in range [0, 65535]
    """
    B, C, T, H, W = latents.shape
    
    # CRITICAL: Clip latents to [-1, 1] first!
    # The model may generate latents outside this range, which breaks token conversion
    latents = torch.clamp(latents, -1.0, 1.0)
    
    # Take first 3 channels (they correspond to the original 3 factors)
    # If C < 3, repeat channels; if C >= 3, take first 3
    if C >= 3:
        latents_3ch = latents[:, :3, :, :, :]  # (B, 3, T, H, W)
    else:
        # Repeat channels if we have fewer than 3
        latents_3ch = latents.repeat(1, (3 // C + 1), 1, 1, 1)[:, :3, :, :, :]
    
    # Reverse normalization: [-1, 1] -> [0, 1] -> [0, 65535]
    # The dataset does: tokens.float() / 65535.0 * 2 - 1
    # So reverse: (latents + 1) / 2 * 65535
    latents_normalized = (latents_3ch + 1.0) / 2.0
    latents_normalized = torch.clamp(latents_normalized, 0.0, 1.0)
    
    # Quantize to [0, 65535]
    tokens_float = latents_normalized * 65535.0
    tokens = tokens_float.round().long()
    tokens = torch.clamp(tokens, 0, 65535)
    
    return tokens


def decode_latents_to_video(latents: torch.Tensor, decoder, device: str = "cuda") -> np.ndarray:
    """Decode continuous latents to video frames using Cosmos decoder.
    
    Args:
        latents: Continuous latents (B, C, T, H, W)
        decoder: Cosmos CausalVideoTokenizer decoder
        device: Device to run on
    
    Returns:
        Video frames as numpy array (B, T*17, H, W, 3) in range [0, 255] uint8
    """
    from cosmos_tokenizer.utils import tensor2numpy
    
    B, C, T, H, W = latents.shape
    
    # Convert to factorized tokens
    tokens = latents_to_factorized_tokens(latents)  # (B, 3, T, H, W)
    
    decoded_frames = []
    for t in range(T):
        # Get tokens for this temporal position: (B, 3, H, W)
        clip_tokens = tokens[:, :, t, :, :].to(device)
        
        # Decode: input [B, 3, H, W] -> output [B, 3, 17, 256, 256]
        with torch.no_grad():
            decoded_clip = decoder.decode(clip_tokens).float()  # (B, 3, 17, 256, 256)
            decoded_frames.append(decoded_clip)
    
    # Concatenate along temporal dimension: (B, 3, T*17, 256, 256)
    video = torch.cat(decoded_frames, dim=2)
    
    # Convert to numpy: tensor2numpy converts (B, 3, T, H, W) -> (B, T, H, W, 3) uint8
    video_np = tensor2numpy(video)  # (B, T*17, 256, 256, 3) uint8
    
    return video_np


def add_label_bar(frame: np.ndarray, label: str, bar_height: int = 64) -> np.ndarray:
    """Add a label bar at the top of a frame."""
    H, W, C = frame.shape
    
    # Create label bar
    bar = np.zeros((bar_height, W, C), dtype=np.uint8)
    
    # Convert to PIL for text drawing
    bar_img = Image.fromarray(bar)
    draw = ImageDraw.Draw(bar_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Center text
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (W - text_width) // 2
    text_y = (bar_height - (bbox[3] - bbox[1])) // 2
    draw.text((text_x, text_y), label, fill='white', font=font)
    
    bar = np.array(bar_img)
    
    # Concatenate bar and frame
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
    
    combined_frames = []
    for t in range(T):
        gt_frame = gt_video[t]
        pred_frame = pred_video[t]
        
        if add_labels:
            gt_frame = add_label_bar(gt_frame, "Ground Truth")
            pred_frame = add_label_bar(pred_frame, "Model Prediction")
        
        # Concatenate horizontally
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
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of Euler integration steps")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--use_medium_config", action="store_true",
                       help="Use FlowHWMConfigMedium instead of default (default: True if checkpoint path contains 'medium')")
    parser.add_argument("--no_labels", action="store_true",
                       help="Don't add text labels to video")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load config - default to medium if checkpoint path contains 'medium'
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
    print(f"Model loaded: {model.get_num_params():,} parameters")
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = FlowHWMDataset(
        data_dir=args.data_dir,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        latent_dim=config.latent_dim,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1
    )
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty")
        return
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Load decoder
    print("Loading Cosmos decoder...")
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    
    decoder = CausalVideoTokenizer(
        checkpoint_dec=f"{args.tokenizer_dir}/decoder.jit",
        device=device,
        dtype="bfloat16"
    )
    
    # Process samples
    for sample_idx in range(min(args.num_samples, len(dataset))):
        print(f"\n{'='*50}")
        print(f"Processing Sample {sample_idx}")
        print(f"{'='*50}")
        
        # Get sample
        sample = dataset[sample_idx]
        v_p = sample["latent_past"].unsqueeze(0).to(device)  # (1, C, T_p, H, W)
        a_p = sample["actions_past"].unsqueeze(0).to(device)  # (1, T_p_frames, action_dim)
        a_f = sample["actions_future"].unsqueeze(0).to(device)  # (1, T_f_frames, action_dim)
        latent_future_gt = sample["latent_future"].unsqueeze(0).to(device)  # (1, C, T_f, H, W)
        
        print(f"Conditioning shapes:")
        print(f"  v_p: {v_p.shape}")
        print(f"  a_p: {a_p.shape}")
        print(f"  a_f: {a_f.shape}")
        
        # Decode ground truth
        print("Decoding ground truth...")
        gt_video_np = decode_latents_to_video(latent_future_gt, decoder, device)
        gt_video = gt_video_np[0]  # Remove batch dimension: (T*17, 256, 256, 3)
        
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
        pred_video = pred_video_np[0]  # Remove batch dimension: (T*17, 256, 256, 3)
        
        # Create comparison video
        output_path = output_dir / f"flow_hwm_sample_{sample_idx}_comparison.mp4"
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
