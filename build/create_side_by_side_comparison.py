#!/usr/bin/env python3
"""Create side-by-side comparison video of ground truth vs model predictions.

Uses the final checkpoint to generate predictions and compares them 
with ground truth videos.
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
sys.path.insert(0, str(Path(__file__).parent))

from masked_hwm.config_test import MaskedHWMTestConfig
from masked_hwm.model import MaskedHWM
from data.dataset import HumanoidWorldModelDataset
from data.collator import MaskedHWMCollator

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("ERROR: imageio not available. Install with: pip install imageio imageio-ffmpeg")
    sys.exit(1)


def load_checkpoint(checkpoint_path: str, config, device: str = "cuda"):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"  Using config from checkpoint (vocab_size={config.vocab_size})")
    else:
        print(f"  Using provided config (vocab_size={config.vocab_size})")
    
    # Warn if vocab size doesn't match Cosmos tokenizer
    if config.vocab_size != 65536:
        print(f"  WARNING: Model vocab_size={config.vocab_size} but Cosmos tokenizer expects 65536")
        print(f"  This may cause decoding issues. Consider retraining with vocab_size=65536")
    
    model = MaskedHWM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    step = checkpoint.get("global_step", "unknown")
    print(f"  Loaded model from step {step}")
    return model, config


@torch.no_grad()
def generate_predictions(model, batch, config, device="cuda"):
    """Generate predictions for a batch."""
    # Clamp past tokens to valid range (ground truth may have values > vocab_size)
    video_past = batch["video_past"].to(device)
    video_past = torch.clamp(video_past, 0, config.vocab_size - 1)
    
    video_future_input = batch["video_future"].to(device).clone()
    
    # Mask all future tokens
    video_future_input.fill_(config.mask_token_id)
    
    # Forward pass
    logits = model(
        v_p_tokens=video_past,
        v_f_tokens=video_future_input,
        a_p=batch["actions_past"].to(device),
        a_f=batch["actions_future"].to(device),
    )
    
    # Sample tokens from logits
    num_factors, B, T_f, H, W, vocab_size = logits.shape
    sampled_tokens = []
    for f in range(num_factors):
        factor_logits = logits[f]
        logits_flat = rearrange(factor_logits, 'b t h w v -> (b t h w) v')
        probs = F.softmax(logits_flat, dim=-1)
        tokens_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        tokens = tokens_flat.view(B, T_f, H, W)
        # Ensure tokens are in valid range [0, vocab_size-1]
        tokens = torch.clamp(tokens, 0, vocab_size - 1)
        sampled_tokens.append(tokens)
    
    predicted_tokens = torch.stack(sampled_tokens, dim=1)
    # Ground truth tokens - don't clamp, use as-is from dataset
    actual_tokens = batch["video_future_labels"].to(device)
    
    return predicted_tokens, actual_tokens


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
    parser = argparse.ArgumentParser(description="Create side-by-side comparison video")
    parser.add_argument("--checkpoint", type=str, 
                       default="/media/skr/storage/robot_world/humanoid_wm/build/checkpoints_test_subset/checkpoint-final/pytorch_model.bin",
                       help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0_test",
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, 
                       default="/media/skr/storage/robot_world/humanoid_wm/build/comparison_videos",
                       help="Output directory for comparison videos")
    parser.add_argument("--tokenizer_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer",
                       help="Path to Cosmos tokenizer directory")
    parser.add_argument("--num_samples", type=int, default=2,
                       help="Number of samples to compare")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--no_labels", action="store_true",
                       help="Don't add text labels to video")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load config and model
    config = MaskedHWMTestConfig()
    model, config = load_checkpoint(args.checkpoint, config, device)
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = HumanoidWorldModelDataset(
        data_dir=args.data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1
    )
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty")
        return
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    collator = MaskedHWMCollator(config)
    
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
        batch = collator([sample])
        
        # Decode ground truth
        print("Decoding ground truth...")
        gt_tokens = batch["video_future_labels"]
        gt_video = decode_tokens_to_video(gt_tokens, decoder, device)[0]
        
        # Generate predictions
        print("Generating predictions...")
        pred_tokens, _ = generate_predictions(model, batch, config, device)
        pred_video = decode_tokens_to_video(pred_tokens, decoder, device)[0]
        
        # Create comparison video
        output_path = output_dir / f"sample_{sample_idx}_comparison.mp4"
        print(f"Creating comparison video...")
        create_side_by_side_video(
            gt_video, pred_video, 
            str(output_path),
            add_labels=not args.no_labels
        )
    
    print(f"\n{'='*50}")
    print(f"Done! Videos saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
