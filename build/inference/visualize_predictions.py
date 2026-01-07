"""Inference and visualization script for Masked-HWM.

Loads a trained checkpoint and generates frame-by-frame comparisons
of actual vs predicted video frames.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from masked_hwm.config import MaskedHWMConfig
from masked_hwm.config_test import MaskedHWMTestConfig
from masked_hwm.model import MaskedHWM
from data.dataset import HumanoidWorldModelDataset
from data.collator import MaskedHWMCollator
from masked_hwm.tokenizer_wrapper import CosmosTokenizerWrapper


def load_checkpoint(checkpoint_path, config, device="cuda"):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config from checkpoint if available
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"Loaded config from checkpoint (step {checkpoint.get('global_step', 'unknown')})")
    
    model = MaskedHWM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (step {checkpoint.get('global_step', 'unknown')})")
    return model, config


def sample_tokens_from_logits(logits, temperature=1.0):
    """Sample tokens from logits for each factor.
    
    Args:
        logits: (num_factors, B, T_f, H, W, vocab_size)
        temperature: Sampling temperature
        
    Returns:
        tokens: (B, num_factors, T_f, H, W)
    """
    num_factors, B, T_f, H, W, vocab_size = logits.shape
    
    # Sample for each factor
    sampled_tokens = []
    for f in range(num_factors):
        factor_logits = logits[f]  # (B, T_f, H, W, vocab_size)
        
        # Flatten for sampling
        logits_flat = rearrange(factor_logits, 'b t h w v -> (b t h w) v')
        
        # Sample with temperature
        probs = F.softmax(logits_flat / temperature, dim=-1)
        tokens_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Reshape back
        tokens = tokens_flat.view(B, T_f, H, W)
        sampled_tokens.append(tokens)
    
    # Stack factors: (B, num_factors, T_f, H, W)
    return torch.stack(sampled_tokens, dim=1)


def decode_factorized_tokens(tokens, tokenizer, vocab_size=65536):
    """Decode factorized tokens to video frames.
    
    IMPORTANT: Cosmos decoder accepts [B, 3, H, W] format directly - pass all 3 factors!
    
    Args:
        tokens: (B, num_factors, T, H, W) - factorized tokens
        tokenizer: CosmosTokenizerWrapper
        vocab_size: Vocabulary size per factor (default: 65536 for Cosmos) - unused, kept for compatibility
        
    Returns:
        video: (B, 3, T, 256, 256) - decoded video frames
    """
    B, num_factors, T, H, W = tokens.shape
    
    # Cosmos DV decoder accepts [B, 3, H, W] format directly
    # Each clip token [3, 32, 32] decodes to 17 frames
    # Process each temporal position separately
    
    decoded_frames = []
    for t in range(T):
        # Get tokens for this temporal position: (B, 3, H, W)
        clip_tokens = tokens[:, :, t, :, :]  # (B, 3, H, W)
        
        # Pass all 3 factors directly to decoder
        # Cosmos decoder expects [B, 3, H, W] and outputs [B, 3, 17, 256, 256]
        try:
            # Note: tokenizer.decode expects [B, T, H, W] but we have [B, 3, H, W]
            # We need to use the Cosmos decoder directly
            from cosmos_tokenizer.video_lib import CausalVideoTokenizer
            import os
            
            # Get decoder path from tokenizer
            decoder_path = os.path.join(tokenizer.checkpoint_dir, "decoder.jit")
            decoder = CausalVideoTokenizer(
                checkpoint_dec=decoder_path,
                device=clip_tokens.device,
                dtype="bfloat16" if clip_tokens.dtype == torch.bfloat16 else "float32"
            )
            
            # Decode: input [B, 3, H, W] -> output [B, 3, 17, 256, 256]
            with torch.no_grad():
                video_clip = decoder.decode(clip_tokens)  # (B, 3, 17, 256, 256)
                decoded_frames.append(video_clip)
        except Exception as e:
            print(f"Warning: Tokenizer decode failed for temporal position {t}: {e}")
            print("Falling back to tokenizer wrapper method (may be incorrect)")
            # Fallback: try using tokenizer wrapper (but this may not work correctly)
            try:
                # This is incorrect but may work as fallback
                single_tokens = tokens[:, 0, t, :, :].unsqueeze(1)  # (B, 1, H, W)
                video_clip = tokenizer.decode(single_tokens)  # (B, 3, 1, 256, 256)
                decoded_frames.append(video_clip)
            except:
                print("Using token visualization instead of decoded video")
                return None
    
    # Concatenate along temporal dimension: (B, 3, T*17, 256, 256)
    video = torch.cat(decoded_frames, dim=2)
    return video


def tensor_to_image(tensor):
    """Convert tensor to PIL Image.
    
    Args:
        tensor: (C, H, W) or (H, W, C) in range [-1, 1] or [0, 1]
        
    Returns:
        PIL Image
    """
    if tensor.dim() == 3:
        if tensor.shape[0] == 3:  # (C, H, W)
            tensor = tensor.permute(1, 2, 0)  # (H, W, C)
    
    # Normalize to [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def create_comparison_grid(actual_frames, predicted_frames, output_path):
    """Create side-by-side comparison grid of actual vs predicted frames.
    
    Args:
        actual_frames: List of PIL Images (actual frames)
        predicted_frames: List of PIL Images (predicted frames)
        output_path: Path to save the comparison image
    """
    num_frames = len(actual_frames)
    cols = min(4, num_frames)
    rows = math.ceil(num_frames / cols) * 2  # 2 rows per frame (actual + predicted)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_frames):
        row_actual = (i // cols) * 2
        row_pred = row_actual + 1
        col = i % cols
        
        # Actual frame
        axes[row_actual, col].imshow(actual_frames[i])
        axes[row_actual, col].set_title(f"Actual Frame {i+1}", fontsize=10)
        axes[row_actual, col].axis('off')
        
        # Predicted frame
        axes[row_pred, col].imshow(predicted_frames[i])
        axes[row_pred, col].set_title(f"Predicted Frame {i+1}", fontsize=10)
        axes[row_pred, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_frames, cols * (rows // 2)):
        row_actual = (i // cols) * 2
        row_pred = row_actual + 1
        col = i % cols
        axes[row_actual, col].axis('off')
        axes[row_pred, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison grid to {output_path}")


@torch.no_grad()
def generate_predictions(model, batch, config, device="cuda"):
    """Generate predictions for a batch.
    
    Args:
        model: Trained MaskedHWM model
        batch: Batch from dataloader
        config: Model config
        device: Device to run on
        
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
    )  # (num_factors, B, T_f, H, W, vocab_size)
    
    # Sample tokens from logits
    predicted_tokens = sample_tokens_from_logits(logits, temperature=1.0)
    # predicted_tokens: (B, num_factors, T_f, H, W)
    
    # Get actual tokens
    actual_tokens = batch["video_future_labels"].to(device)  # (B, num_factors, T_f, H, W)
    
    return predicted_tokens, actual_tokens


def main():
    parser = argparse.ArgumentParser(description="Visualize actual vs predicted frames")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint (pytorch_model.bin)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to validation dataset directory")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to visualize")
    parser.add_argument("--tokenizer_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer",
                       help="Path to Cosmos tokenizer directory")
    parser.add_argument("--use_test_config", action="store_true",
                       help="Use test config (for checkpoints trained with test config)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--no_filter", action="store_true",
                       help="Disable segment filtering (useful for test subsets)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    if args.use_test_config:
        config = MaskedHWMTestConfig()
    else:
        config = MaskedHWMConfig()
    
    # Load checkpoint
    device = args.device if torch.cuda.is_available() else "cpu"
    model, config = load_checkpoint(args.checkpoint, config, device)
    
    # Load tokenizer
    print("Loading Cosmos tokenizer...")
    tokenizer = CosmosTokenizerWrapper(
        checkpoint_dir=args.tokenizer_dir,
        device=device
    )
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    filter_interrupts = not args.no_filter
    filter_overlaps = not args.no_filter
    dataset = HumanoidWorldModelDataset(
        data_dir=args.data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        filter_interrupts=filter_interrupts,
        filter_overlaps=filter_overlaps,
    )
    
    collator = MaskedHWMCollator(config)
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print(f"ERROR: Dataset is empty (0 sequences found)")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Please check:")
        print(f"    1. Dataset files exist in {args.data_dir}")
        print(f"    2. Run create_test_subsets.py to create test subsets")
        print(f"    3. Check metadata.json has num_shards > 0")
        return
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    
    print(f"Generating predictions for {args.num_samples} samples...")
    
    for sample_idx, batch in enumerate(dataloader):
        if sample_idx >= args.num_samples:
            break
        
        print(f"\nProcessing sample {sample_idx + 1}/{args.num_samples}...")
        
        # Generate predictions
        predicted_tokens, actual_tokens = generate_predictions(
            model, batch, config, device
        )
        
        # Decode tokens to video frames
        print("  Decoding tokens to video frames...")
        
        B = predicted_tokens.shape[0]
        T_f = predicted_tokens.shape[2]
        H, W = predicted_tokens.shape[3], predicted_tokens.shape[4]
        
        # Try to decode to actual video frames
        try:
            predicted_video = decode_factorized_tokens(
                predicted_tokens, tokenizer, vocab_size=config.vocab_size
            )
            actual_video = decode_factorized_tokens(
                actual_tokens, tokenizer, vocab_size=config.vocab_size
            )
            
            if predicted_video is not None and actual_video is not None:
                # Successfully decoded to video
                # predicted_video: (B, 3, T, 256, 256)
                # actual_video: (B, 3, T, 256, 256)
                print("  Successfully decoded tokens to video frames")
                
                actual_frames = []
                predicted_frames = []
                
                for t in range(T_f):
                    # Actual frame: (3, 256, 256)
                    actual_frame = tensor_to_image(actual_video[0, :, t])
                    actual_frames.append(actual_frame)
                    
                    # Predicted frame: (3, 256, 256)
                    pred_frame = tensor_to_image(predicted_video[0, :, t])
                    predicted_frames.append(pred_frame)
            else:
                raise ValueError("Decoding returned None")
                
        except Exception as e:
            print(f"  Warning: Video decoding failed ({e})")
            print("  Falling back to token visualization...")
            
            # Fallback: Visualize tokens directly
            pred_tokens_vis = predicted_tokens[0, 0]  # (T_f, H, W) - first factor
            actual_tokens_vis = actual_tokens[0, 0]  # (T_f, H, W) - first factor
            
            # Normalize tokens to [0, 1] for visualization
            pred_tokens_vis = (pred_tokens_vis.float() / config.vocab_size).clamp(0, 1)
            actual_tokens_vis = (actual_tokens_vis.float() / config.vocab_size).clamp(0, 1)
            
            # Convert to PIL Images
            actual_frames = []
            predicted_frames = []
            
            for t in range(T_f):
                # Actual frame (token visualization)
                actual_frame = tensor_to_image(actual_tokens_vis[t].unsqueeze(0).repeat(3, 1, 1))
                actual_frames.append(actual_frame)
                
                # Predicted frame (token visualization)
                pred_frame = tensor_to_image(pred_tokens_vis[t].unsqueeze(0).repeat(3, 1, 1))
                predicted_frames.append(pred_frame)
        
        # Create comparison grid
        output_path = output_dir / f"sample_{sample_idx + 1}_comparison.png"
        create_comparison_grid(actual_frames, predicted_frames, output_path)
        
        # Also save individual frames
        frames_dir = output_dir / f"sample_{sample_idx + 1}_frames"
        frames_dir.mkdir(exist_ok=True)
        
        for t in range(T_f):
            # Get frame dimensions
            frame_w, frame_h = actual_frames[t].size
            
            # Side-by-side frame
            side_by_side = Image.new('RGB', (frame_w * 2, frame_h))
            side_by_side.paste(actual_frames[t], (0, 0))
            side_by_side.paste(predicted_frames[t], (frame_w, 0))
            side_by_side.save(frames_dir / f"frame_{t:02d}_comparison.png")
            
            # Also save individual frames
            actual_frames[t].save(frames_dir / f"frame_{t:02d}_actual.png")
            predicted_frames[t].save(frames_dir / f"frame_{t:02d}_predicted.png")
        
        print(f"  Saved visualizations to {output_dir}")
    
    print(f"\nVisualization complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
