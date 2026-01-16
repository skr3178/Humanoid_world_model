#!/usr/bin/env python3
"""Debug Masked-HWM video decoding and generation.

This script provides debugging utilities for the Masked-HWM model to diagnose
pixelated video output issues.

NOTE: This file is SEPARATE from flow_hwm debugging utilities.
The masked model uses DISCRETE tokens directly, not continuous latents.

Debugging Steps:
1. Tokenizer Roundtrip Test - encode video -> decode -> compare
2. Ground Truth Token Decoding - decode dataset tokens directly (no model)
3. Decoder Format Verification - test decoder with various input formats
4. Mask Ratio Visualization - show predictions at different mask levels

Usage:
    python debug_masked_hwm_videos.py --step 2 --data_dir /path/to/data --tokenizer_dir /path/to/tokenizer
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
# HELPER FUNCTIONS (Masked-HWM specific - DO NOT share with flow_hwm)
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


def load_cosmos_encoder(tokenizer_dir: str, device: str = "cuda"):
    """Load Cosmos DV encoder for video encoding."""
    try:
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    except ImportError:
        raise ImportError(
            "cosmos_tokenizer not found. Please activate the 'cosmos-tokenizer' conda environment."
        )

    encoder_path = f"{tokenizer_dir}/encoder.jit"
    if not Path(encoder_path).exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    encoder = CausalVideoTokenizer(
        checkpoint_enc=encoder_path,
        device=device,
        dtype="bfloat16" if device == "cuda" else "float32"
    )
    return encoder


def decode_factorized_clip(clip_tokens: torch.Tensor, decoder, device: str) -> torch.Tensor:
    """Decode a single clip of factorized tokens.

    Args:
        clip_tokens: (3, H, W) or (B, 3, H, W) - factorized discrete tokens
        decoder: Cosmos DV decoder
        device: Device to use

    Returns:
        video: (B, 3, 17, 256, 256) - decoded video clip (17 frames per clip)
    """
    if clip_tokens.dim() == 3:
        clip_tokens = clip_tokens.unsqueeze(0)  # (1, 3, H, W)

    clip_tokens = clip_tokens.to(device).long()

    with torch.no_grad():
        decoded = decoder.decode(clip_tokens).float()  # (B, 3, 17, 256, 256)

    return decoded


def decode_video_from_tokens(tokens: torch.Tensor, decoder, device: str) -> torch.Tensor:
    """Decode multiple clips of factorized tokens to a full video.

    Args:
        tokens: (T_clips, 3, H, W) or (B, T_clips, 3, H, W) - factorized tokens
        decoder: Cosmos DV decoder
        device: Device to use

    Returns:
        video: (B, 3, T_clips*17, 256, 256) - full decoded video
    """
    if tokens.dim() == 4:
        tokens = tokens.unsqueeze(0)  # (1, T_clips, 3, H, W)

    T_clips = tokens.shape[1]

    decoded_clips = []
    for t in range(T_clips):
        clip_tokens = tokens[:, t]  # (B, 3, H, W)
        decoded_clip = decode_factorized_clip(clip_tokens, decoder, device)
        decoded_clips.append(decoded_clip)

    # Concatenate along temporal dimension
    video = torch.cat(decoded_clips, dim=2)  # (B, 3, T_clips*17, 256, 256)
    return video


def tensor_to_numpy_video(video: torch.Tensor) -> np.ndarray:
    """Convert tensor video to numpy array for saving.

    Args:
        video: (B, C, T, H, W) or (C, T, H, W) in range [-1, 1]

    Returns:
        numpy array (T, H, W, C) in range [0, 255] uint8
    """
    try:
        from cosmos_tokenizer.utils import tensor2numpy
        return tensor2numpy(video)[0] if video.dim() == 5 else tensor2numpy(video.unsqueeze(0))[0]
    except ImportError:
        pass

    # Fallback manual conversion
    if video.dim() == 5:
        video = video[0]  # Remove batch dim

    # (C, T, H, W) -> (T, H, W, C)
    video = video.permute(1, 2, 3, 0)

    # Normalize to [0, 1]
    if video.min() < 0:
        video = (video + 1) / 2

    video = torch.clamp(video, 0, 1)
    video_np = (video.cpu().numpy() * 255).astype(np.uint8)

    return video_np


def save_video(frames: np.ndarray, output_path: Path, fps: int = 30) -> None:
    """Save numpy video array to mp4."""
    if not HAS_IMAGEIO:
        print(f"ERROR: Cannot save video - imageio not available")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(
        str(output_path),
        frames,
        fps=fps,
        codec='libx264',
        quality=8
    )
    print(f"Saved: {output_path}")


def add_label_bar(frame: np.ndarray, label: str, bar_height: int = 40) -> np.ndarray:
    """Add a label bar at the top of a frame."""
    from PIL import Image, ImageDraw, ImageFont

    _, W, C = frame.shape
    bar = np.zeros((bar_height, W, C), dtype=np.uint8)
    bar_img = Image.fromarray(bar)
    draw = ImageDraw.Draw(bar_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (W - text_width) // 2
    text_y = (bar_height - (bbox[3] - bbox[1])) // 2
    draw.text((text_x, text_y), label, fill="white", font=font)

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


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two videos."""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# =============================================================================
# STEP 1: TOKENIZER ROUNDTRIP TEST
# =============================================================================

def test_tokenizer_roundtrip(
    tokenizer_dir: str,
    output_dir: Path,
    device: str = "cuda",
    num_frames: int = 17
) -> Dict[str, Any]:
    """Test tokenizer encode -> decode roundtrip with synthetic video.

    Creates a test pattern video, encodes it to tokens, decodes back,
    and compares the original vs reconstructed.

    Args:
        tokenizer_dir: Path to Cosmos tokenizer
        output_dir: Directory to save outputs
        device: Device to use
        num_frames: Number of frames in test video (must be 17 for single clip)

    Returns:
        Dict with PSNR metric and paths to saved videos
    """
    print("\n" + "="*60)
    print("STEP 1: Tokenizer Roundtrip Test")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder and decoder
    print("Loading Cosmos encoder and decoder...")
    encoder = load_cosmos_encoder(tokenizer_dir, device)
    decoder = load_cosmos_decoder(tokenizer_dir, device)

    # Create synthetic test video with gradient pattern
    print(f"Creating test video ({num_frames} frames, 256x256)...")
    T, H, W = num_frames, 256, 256

    # Create a gradient + moving pattern that's easy to verify visually
    video = torch.zeros(1, 3, T, H, W, device=device)
    for t in range(T):
        # Red channel: horizontal gradient
        video[0, 0, t] = torch.linspace(-1, 1, W).unsqueeze(0).expand(H, W)
        # Green channel: vertical gradient
        video[0, 1, t] = torch.linspace(-1, 1, H).unsqueeze(1).expand(H, W)
        # Blue channel: temporal gradient (moving bar)
        bar_pos = int((t / T) * W)
        bar_width = W // 8
        video[0, 2, t, :, max(0, bar_pos-bar_width):min(W, bar_pos+bar_width)] = 1.0

    print(f"  Video shape: {video.shape}")
    print(f"  Video range: [{video.min().item():.3f}, {video.max().item():.3f}]")

    # Encode to tokens
    print("Encoding video to discrete tokens...")
    with torch.no_grad():
        # Cosmos encoder expects (B, C, T, H, W)
        output = encoder.encode(video.to(torch.bfloat16))

        if isinstance(output, tuple):
            tokens = output[0]  # (B, T', H', W') or (B, 3, H', W') for factorized
        else:
            tokens = output

    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Tokens dtype: {tokens.dtype}")
    print(f"  Tokens range: [{tokens.min().item()}, {tokens.max().item()}]")

    # Decode back to video
    print("Decoding tokens back to video...")
    with torch.no_grad():
        # For factorized format (B, 3, H, W), decode directly
        if tokens.dim() == 4 and tokens.shape[1] == 3:
            reconstructed = decoder.decode(tokens.long()).float()
        else:
            # For non-factorized, need to handle differently
            reconstructed = decoder.decode(tokens.long()).float()

    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Reconstructed range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")

    # Convert to numpy for comparison
    original_np = tensor_to_numpy_video(video.cpu())
    reconstructed_np = tensor_to_numpy_video(reconstructed.cpu())

    # Compute PSNR
    # Truncate to same length
    min_t = min(original_np.shape[0], reconstructed_np.shape[0])
    psnr = compute_psnr(original_np[:min_t], reconstructed_np[:min_t])
    print(f"\n  PSNR: {psnr:.2f} dB")

    # Save videos
    save_video(original_np, output_dir / "roundtrip_original.mp4")
    save_video(reconstructed_np, output_dir / "roundtrip_reconstructed.mp4")

    # Create side-by-side comparison
    create_side_by_side_video(
        [original_np[:min_t], reconstructed_np[:min_t]],
        ["Original", "Reconstructed"],
        output_dir / "roundtrip_comparison.mp4"
    )

    # Interpretation
    print("\n" + "-"*40)
    print("INTERPRETATION:")
    if psnr > 30:
        print("  PASS - Tokenizer roundtrip is working well (PSNR > 30 dB)")
    elif psnr > 25:
        print("  OK - Some quality loss expected from compression (PSNR 25-30 dB)")
    else:
        print("  WARNING - Significant quality loss (PSNR < 25 dB)")
        print("  Check: Cosmos tokenizer installation, encoder/decoder paths")
    print("-"*40)

    return {
        "psnr": psnr,
        "original_path": str(output_dir / "roundtrip_original.mp4"),
        "reconstructed_path": str(output_dir / "roundtrip_reconstructed.mp4"),
        "comparison_path": str(output_dir / "roundtrip_comparison.mp4"),
    }


# =============================================================================
# STEP 2: GROUND TRUTH TOKEN DECODING
# =============================================================================

def decode_ground_truth_tokens(
    data_dir: str,
    tokenizer_dir: str,
    output_dir: Path,
    sample_indices: List[int],
    device: str = "cuda",
    config: Optional[MaskedHWMConfig] = None
) -> Dict[str, Any]:
    """Decode ground truth tokens from dataset directly (no model inference).

    This isolates tokenizer/data issues from model issues.

    Args:
        data_dir: Path to dataset directory
        tokenizer_dir: Path to Cosmos tokenizer
        output_dir: Directory to save outputs
        sample_indices: Which samples to decode
        device: Device to use
        config: Model config (for dataset parameters)

    Returns:
        Dict with decoded video paths and token statistics
    """
    print("\n" + "="*60)
    print("STEP 2: Ground Truth Token Decoding")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = MaskedHWMConfig()

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = HumanoidWorldModelDataset(
        data_dir=data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1
    )

    if len(dataset) == 0:
        print("ERROR: Dataset is empty")
        return {"error": "Empty dataset"}

    print(f"Dataset loaded: {len(dataset)} samples")

    # Load decoder
    print("Loading Cosmos decoder...")
    decoder = load_cosmos_decoder(tokenizer_dir, device)

    results = []

    for idx in sample_indices:
        if idx >= len(dataset):
            print(f"Sample {idx} out of range, skipping")
            continue

        print(f"\nDecoding sample {idx}...")

        # Get sample from dataset
        sample = dataset[idx]
        video_future = sample["video_future"]  # (T_clips, 3, 32, 32)

        print(f"  Token shape: {video_future.shape}")

        # Check token statistics
        if isinstance(video_future, np.ndarray):
            video_future = torch.from_numpy(video_future.copy())

        print(f"  Token dtype: {video_future.dtype}")
        print(f"  Token range: [{video_future.min().item()}, {video_future.max().item()}]")

        # Check for invalid tokens
        max_valid = 65535
        invalid_mask = video_future > max_valid
        if invalid_mask.any():
            num_invalid = invalid_mask.sum().item()
            print(f"  WARNING: {num_invalid} tokens > {max_valid} (out of vocab)")

        # Decode
        video = decode_video_from_tokens(video_future, decoder, device)
        print(f"  Decoded video shape: {video.shape}")
        print(f"  Decoded video range: [{video.min().item():.3f}, {video.max().item():.3f}]")

        # Convert to numpy and save
        video_np = tensor_to_numpy_video(video)

        output_path = output_dir / f"gt_decoded_sample_{idx}.mp4"
        save_video(video_np, output_path)

        results.append({
            "sample_idx": idx,
            "token_shape": tuple(video_future.shape),
            "token_range": (video_future.min().item(), video_future.max().item()),
            "video_path": str(output_path)
        })

    # Interpretation
    print("\n" + "-"*40)
    print("INTERPRETATION:")
    print("  If videos look CORRECT -> Tokenizer and data are fine")
    print("    -> Problem is in model predictions")
    print("  If videos look PIXELATED -> Tokenizer or data issue")
    print("    -> Run Step 1 to test tokenizer, check data integrity")
    print("-"*40)

    return {"samples": results}


# =============================================================================
# STEP 3: DECODER FORMAT VERIFICATION
# =============================================================================

def verify_decoder_format(
    tokenizer_dir: str,
    output_dir: Path,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Verify decoder accepts correct input format and handles edge cases.

    Tests:
    a) Valid tokens [B, 3, H, W] in range [0, 65535]
    b) Boundary values (0, 65535)
    c) Out-of-range tokens (should fail gracefully)
    d) Wrong dtype (float vs long)

    Args:
        tokenizer_dir: Path to Cosmos tokenizer
        output_dir: Directory to save outputs
        device: Device to use

    Returns:
        Dict with test results
    """
    print("\n" + "="*60)
    print("STEP 3: Decoder Format Verification")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load decoder
    print("Loading Cosmos decoder...")
    decoder = load_cosmos_decoder(tokenizer_dir, device)

    results = {}

    # Test a) Valid random tokens
    print("\nTest A: Valid random tokens [0, 65535]...")
    try:
        tokens = torch.randint(0, 65536, (1, 3, 32, 32), device=device, dtype=torch.long)
        with torch.no_grad():
            output = decoder.decode(tokens)
        print(f"  PASS - Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        results["valid_tokens"] = "PASS"
    except Exception as e:
        print(f"  FAIL - {e}")
        results["valid_tokens"] = f"FAIL: {e}"

    # Test b) Boundary values
    print("\nTest B: Boundary values (0 and 65535)...")
    try:
        # All zeros
        tokens_zero = torch.zeros(1, 3, 32, 32, device=device, dtype=torch.long)
        with torch.no_grad():
            output_zero = decoder.decode(tokens_zero)
        print(f"  All zeros -> Output range: [{output_zero.min().item():.3f}, {output_zero.max().item():.3f}]")

        # All max
        tokens_max = torch.full((1, 3, 32, 32), 65535, device=device, dtype=torch.long)
        with torch.no_grad():
            output_max = decoder.decode(tokens_max)
        print(f"  All 65535 -> Output range: [{output_max.min().item():.3f}, {output_max.max().item():.3f}]")

        results["boundary_values"] = "PASS"
    except Exception as e:
        print(f"  FAIL - {e}")
        results["boundary_values"] = f"FAIL: {e}"

    # Test c) Out-of-range tokens
    print("\nTest C: Out-of-range tokens (>65535)...")
    try:
        tokens_invalid = torch.full((1, 3, 32, 32), 70000, device=device, dtype=torch.long)
        with torch.no_grad():
            output_invalid = decoder.decode(tokens_invalid)
        print(f"  Accepted (may have undefined behavior)")
        print(f"  Output shape: {output_invalid.shape}")
        results["out_of_range"] = "ACCEPTED (no error)"
    except Exception as e:
        print(f"  Rejected with error: {e}")
        results["out_of_range"] = f"REJECTED: {e}"

    # Test d) Float dtype (should work but may warn)
    print("\nTest D: Float dtype input...")
    try:
        tokens_float = torch.rand(1, 3, 32, 32, device=device) * 65535
        with torch.no_grad():
            _ = decoder.decode(tokens_float.long())  # Cast to long
        print("  PASS (after casting to long)")
        results["float_dtype"] = "PASS (needs .long() cast)"
    except Exception as e:
        print(f"  FAIL - {e}")
        results["float_dtype"] = f"FAIL: {e}"

    # Summary
    print("\n" + "-"*40)
    print("SUMMARY:")
    for test, result in results.items():
        status = "PASS" if "PASS" in result else "FAIL" if "FAIL" in result else "?"
        print(f"  {test}: {status}")
    print("-"*40)

    return results


# =============================================================================
# STEP 4: MASK RATIO VISUALIZATION
# =============================================================================

def create_mask_ratio_visualization(
    checkpoint_path: str,
    data_dir: str,
    tokenizer_dir: str,
    output_dir: Path,
    sample_idx: int = 0,
    mask_ratios: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    device: str = "cuda",
    config: Optional[MaskedHWMConfig] = None,
    use_test_config: bool = False
) -> Dict[str, Any]:
    """Create visualization showing model predictions at different mask ratios.

    At 0% mask (no noise) -> output should be IDENTICAL to ground truth
    At higher mask ratios -> gradual degradation expected

    This helps identify if the model is corrupting input tokens.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to dataset
        tokenizer_dir: Path to Cosmos tokenizer
        output_dir: Directory to save outputs
        sample_idx: Which sample to use
        mask_ratios: List of mask ratios to test
        device: Device to use
        config: Model config
        use_test_config: Use test config instead of default

    Returns:
        Dict with output paths
    """
    print("\n" + "="*60)
    print("STEP 4: Mask Ratio Visualization")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if config is None:
        config = MaskedHWMTestConfig() if use_test_config else MaskedHWMConfig()

    # Load checkpoint
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"  Using config from checkpoint")

    model = MaskedHWM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).to(torch.bfloat16)  # FlashAttention requires bf16/fp16
    model.eval()
    print(f"  Model loaded (step {checkpoint.get('global_step', 'unknown')})")

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = HumanoidWorldModelDataset(
        data_dir=data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1
    )

    if len(dataset) == 0 or sample_idx >= len(dataset):
        print("ERROR: Dataset empty or sample index out of range")
        return {"error": "Invalid sample"}

    # Load decoder
    print("Loading Cosmos decoder...")
    decoder = load_cosmos_decoder(tokenizer_dir, device)

    # Get sample
    sample = dataset[sample_idx]

    # Prepare inputs - handle both tensor and numpy array
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        return torch.from_numpy(x.copy())

    video_past = to_tensor(sample["video_past"]).unsqueeze(0).to(device)
    video_future_gt = to_tensor(sample["video_future"]).unsqueeze(0).to(device)
    actions_past = to_tensor(sample["actions_past"]).unsqueeze(0).to(device).to(torch.bfloat16)
    actions_future = to_tensor(sample["actions_future"]).unsqueeze(0).to(device).to(torch.bfloat16)

    # Reshape tokens to model format: (B, num_factors, T_clips, H, W)
    # Dataset returns: (T_clips, 3, H, W)
    video_past = video_past.permute(0, 2, 1, 3, 4)  # (B, 3, T_clips, H, W)
    video_future_gt = video_future_gt.permute(0, 2, 1, 3, 4)  # (B, 3, T_clips, H, W)

    print(f"\nSample {sample_idx}:")
    print(f"  Past video shape: {video_past.shape}")
    print(f"  Future video shape (GT): {video_future_gt.shape}")
    print(f"  Actions past shape: {actions_past.shape}")
    print(f"  Actions future shape: {actions_future.shape}")

    # Ensure tokens are in valid range
    video_past = video_past % config.vocab_size
    video_future_gt = video_future_gt % config.vocab_size

    # First decode ground truth for comparison
    print("\nDecoding ground truth...")
    gt_tokens_for_decode = video_future_gt.permute(0, 2, 1, 3, 4)  # (B, T_clips, 3, H, W)
    gt_video = decode_video_from_tokens(gt_tokens_for_decode[0], decoder, device)
    gt_video_np = tensor_to_numpy_video(gt_video)

    save_video(gt_video_np, output_dir / "mask_ratio_gt.mp4")

    # Generate predictions at each mask ratio
    all_videos = [gt_video_np]
    all_labels = ["GT"]

    B, num_factors, T_clips, H, W = video_future_gt.shape
    num_tokens = T_clips * H * W

    for mask_ratio in mask_ratios:
        print(f"\nMask ratio: {mask_ratio*100:.0f}%")

        # Create deterministic mask (same positions for reproducibility)
        torch.manual_seed(42)
        mask_flat = torch.rand(num_tokens) < mask_ratio
        mask = mask_flat.view(T_clips, H, W).unsqueeze(0).unsqueeze(0)  # (1, 1, T_clips, H, W)
        mask = mask.expand(B, num_factors, T_clips, H, W).to(device)

        num_masked = mask[0, 0].sum().item()
        print(f"  Masking {num_masked}/{num_tokens} tokens ({num_masked/num_tokens*100:.1f}%)")

        # Apply mask to future tokens
        video_future_input = video_future_gt.clone()
        video_future_input[mask] = config.mask_token_id

        # Forward pass
        with torch.no_grad():
            logits = model(
                v_p_tokens=video_past,
                v_f_tokens=video_future_input,
                a_p=actions_past,
                a_f=actions_future
            )  # (num_factors, B, T_clips, H, W, vocab_size)

        # Sample tokens using temperature-scaled categorical sampling (like MaskGiT)
        # This prevents mode collapse to few tokens when logits are peaked
        temperature = 1.2  # >1.0 for more diversity, tune as needed (1.0-1.5)
        logits_scaled = logits / temperature

        # Categorical sampling: sample from probability distribution
        original_shape = logits_scaled.shape[:-1]  # (num_factors, B, T_clips, H, W)
        vocab_size = logits_scaled.shape[-1]
        flat_logits = logits_scaled.reshape(-1, vocab_size)  # (N, vocab_size)
        probs = torch.softmax(flat_logits, dim=-1)
        predicted_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (N,)
        predicted_tokens = predicted_flat.reshape(original_shape)  # (num_factors, B, T_clips, H, W)
        predicted_tokens = predicted_tokens.permute(1, 0, 2, 3, 4)  # (B, num_factors, T_clips, H, W)

        # Replace ONLY masked positions with predictions
        output_tokens = video_future_gt.clone()
        output_tokens[mask] = predicted_tokens[mask]

        # Check how many tokens differ from GT
        diff_mask = output_tokens != video_future_gt
        num_diff = diff_mask.sum().item()
        print(f"  Tokens differing from GT: {num_diff}")

        # Decode
        output_for_decode = output_tokens.permute(0, 2, 1, 3, 4)  # (B, T_clips, 3, H, W)
        output_video = decode_video_from_tokens(output_for_decode[0], decoder, device)
        output_video_np = tensor_to_numpy_video(output_video)

        # Save individual video
        save_video(output_video_np, output_dir / f"mask_ratio_{int(mask_ratio*100)}.mp4")

        all_videos.append(output_video_np)
        all_labels.append(f"{int(mask_ratio*100)}%")

    # Create comparison video
    create_side_by_side_video(
        all_videos,
        all_labels,
        output_dir / "mask_ratio_comparison.mp4"
    )

    # Interpretation
    print("\n" + "-"*40)
    print("INTERPRETATION:")
    print("  Check mask_ratio_comparison.mp4")
    print("  - At 0% mask: Should be IDENTICAL to GT")
    print("    If different -> Model corrupts input tokens (bug)")
    print("  - At higher ratios: Gradual degradation expected")
    print("    If 0% is good but others pixelated -> Model prediction quality issue")
    print("-"*40)

    return {
        "comparison_path": str(output_dir / "mask_ratio_comparison.mp4"),
        "individual_videos": [str(output_dir / f"mask_ratio_{int(r*100)}.mp4") for r in mask_ratios]
    }


# =============================================================================
# STEP 5: FACTORIZED TOKEN HANDLING STRATEGIES
# =============================================================================

def test_factorized_token_strategies(
    data_dir: str,
    tokenizer_dir: str,
    output_dir: Path,
    sample_idx: int = 0,
    device: str = "cuda",
    config: Optional[MaskedHWMConfig] = None
) -> Dict[str, Any]:
    """Test different strategies for handling factorized tokens.

    The Cosmos DV tokenizer uses 3 factorized tokens per position.
    This tests whether different handling strategies affect output quality:
    - Strategy A: Pass all 3 factors directly (current approach)
    - Strategy B: Sum the factors before decoding
    - Strategy C: Use only factor 0 (first codebook)
    - Strategy D: Average the factors

    Args:
        data_dir: Path to dataset
        tokenizer_dir: Path to Cosmos tokenizer
        output_dir: Directory to save outputs
        sample_idx: Which sample to use
        device: Device to use
        config: Model config

    Returns:
        Dict with comparison results
    """
    print("\n" + "="*60)
    print("STEP 5: Factorized Token Handling Strategies")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = MaskedHWMConfig()

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = HumanoidWorldModelDataset(
        data_dir=data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1
    )

    if len(dataset) == 0 or sample_idx >= len(dataset):
        print("ERROR: Dataset empty or sample index out of range")
        return {"error": "Invalid sample"}

    # Load decoder
    print("Loading Cosmos decoder...")
    decoder = load_cosmos_decoder(tokenizer_dir, device)

    # Get sample
    sample = dataset[sample_idx]
    video_future = sample["video_future"]  # (T_clips, 3, 32, 32)

    if isinstance(video_future, np.ndarray):
        video_future = torch.from_numpy(video_future.copy())

    print(f"\nSample {sample_idx}:")
    print(f"  Token shape: {video_future.shape}")
    print(f"  Token range: [{video_future.min().item()}, {video_future.max().item()}]")

    T_clips = video_future.shape[0]
    results = {}
    all_videos = []
    all_labels = []

    # Strategy A: Pass all 3 factors directly (current/default approach)
    print("\nStrategy A: Pass all 3 factors directly to decoder...")
    try:
        decoded_clips_a = []
        for t in range(T_clips):
            clip_tokens = video_future[t].unsqueeze(0).to(device)  # (1, 3, 32, 32)
            with torch.no_grad():
                decoded = decoder.decode(clip_tokens.long()).float()
            decoded_clips_a.append(decoded)
        video_a = torch.cat(decoded_clips_a, dim=2)
        video_a_np = tensor_to_numpy_video(video_a)
        save_video(video_a_np, output_dir / "strategy_a_direct.mp4")
        all_videos.append(video_a_np)
        all_labels.append("A: Direct")
        results["strategy_a"] = "SUCCESS"
        print(f"  Output shape: {video_a.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["strategy_a"] = f"FAILED: {e}"

    # Strategy B: Sum the factors
    print("\nStrategy B: Sum factors before decoding...")
    try:
        decoded_clips_b = []
        for t in range(T_clips):
            clip_tokens = video_future[t]  # (3, 32, 32)
            # Sum across factor dimension
            summed = clip_tokens.sum(dim=0, keepdim=True)  # (1, 32, 32)
            # Clamp to valid range
            summed = torch.clamp(summed, 0, 65535)
            # Expand back to 3 factors (decoder may require it)
            summed_3 = summed.expand(3, 32, 32).unsqueeze(0).to(device)  # (1, 3, 32, 32)
            with torch.no_grad():
                decoded = decoder.decode(summed_3.long()).float()
            decoded_clips_b.append(decoded)
        video_b = torch.cat(decoded_clips_b, dim=2)
        video_b_np = tensor_to_numpy_video(video_b)
        save_video(video_b_np, output_dir / "strategy_b_summed.mp4")
        all_videos.append(video_b_np)
        all_labels.append("B: Summed")
        results["strategy_b"] = "SUCCESS"
        print(f"  Output shape: {video_b.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["strategy_b"] = f"FAILED: {e}"

    # Strategy C: Use only factor 0
    print("\nStrategy C: Use only factor 0 (first codebook)...")
    try:
        decoded_clips_c = []
        for t in range(T_clips):
            clip_tokens = video_future[t]  # (3, 32, 32)
            # Use only first factor, repeat for decoder
            factor0 = clip_tokens[0:1].expand(3, 32, 32).unsqueeze(0).to(device)  # (1, 3, 32, 32)
            with torch.no_grad():
                decoded = decoder.decode(factor0.long()).float()
            decoded_clips_c.append(decoded)
        video_c = torch.cat(decoded_clips_c, dim=2)
        video_c_np = tensor_to_numpy_video(video_c)
        save_video(video_c_np, output_dir / "strategy_c_factor0.mp4")
        all_videos.append(video_c_np)
        all_labels.append("C: Factor0")
        results["strategy_c"] = "SUCCESS"
        print(f"  Output shape: {video_c.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["strategy_c"] = f"FAILED: {e}"

    # Strategy D: Average the factors
    print("\nStrategy D: Average factors before decoding...")
    try:
        decoded_clips_d = []
        for t in range(T_clips):
            clip_tokens = video_future[t].float()  # (3, 32, 32)
            # Average across factor dimension
            averaged = clip_tokens.mean(dim=0, keepdim=True)  # (1, 32, 32)
            averaged = averaged.round().long()
            averaged = torch.clamp(averaged, 0, 65535)
            # Expand back to 3 factors
            averaged_3 = averaged.expand(3, 32, 32).unsqueeze(0).to(device)  # (1, 3, 32, 32)
            with torch.no_grad():
                decoded = decoder.decode(averaged_3.long()).float()
            decoded_clips_d.append(decoded)
        video_d = torch.cat(decoded_clips_d, dim=2)
        video_d_np = tensor_to_numpy_video(video_d)
        save_video(video_d_np, output_dir / "strategy_d_averaged.mp4")
        all_videos.append(video_d_np)
        all_labels.append("D: Averaged")
        results["strategy_d"] = "SUCCESS"
        print(f"  Output shape: {video_d.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["strategy_d"] = f"FAILED: {e}"

    # Create comparison video
    if len(all_videos) > 1:
        create_side_by_side_video(
            all_videos,
            all_labels,
            output_dir / "strategy_comparison.mp4"
        )

    # Interpretation
    print("\n" + "-"*40)
    print("INTERPRETATION:")
    print("  Compare strategy_comparison.mp4")
    print("  - Strategy A (Direct): Current approach - should be correct for Cosmos DV")
    print("  - Strategy B (Summed): If better, factors need combining")
    print("  - Strategy C (Factor0): If better, other factors may be corrupted")
    print("  - Strategy D (Averaged): Alternative combining method")
    print("  If A is best -> Factorization is handled correctly")
    print("-"*40)

    return results


# =============================================================================
# STEP 5b: FACTOR INDEPENDENCE DIAGNOSTIC
# =============================================================================

def diagnose_factor_independence(
    checkpoint_path: str,
    data_dir: str,
    tokenizer_dir: str,
    output_dir: Path,
    sample_idx: int = 0,
    device: str = "cuda",
    config: Optional[MaskedHWMConfig] = None,
    use_test_config: bool = False
) -> Dict[str, Any]:
    """Diagnose whether model predicts independent vs identical factors.
    
    If all 3 factors are predicted identically, the decoder will produce
    output similar to "averaged" tokens (Strategy D from Step 5).
    
    This is a TRAINING BUG indicator - factors should be independent.
    """
    print("\n" + "="*60)
    print("STEP 5b: Factor Independence Diagnostic")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if config is None:
        from masked_hwm.config_test import MaskedHWMTestConfig
        config = MaskedHWMTestConfig() if use_test_config else MaskedHWMConfig()

    # Load checkpoint
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        config = checkpoint["config"]
        print("  Using config from checkpoint")

    model = MaskedHWM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).to(torch.bfloat16)  # FlashAttention requires bf16/fp16
    model.eval()
    print(f"  Model loaded (step {checkpoint.get('global_step', 'unknown')})")

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = HumanoidWorldModelDataset(
        data_dir=data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1
    )

    if len(dataset) == 0 or sample_idx >= len(dataset):
        print("ERROR: Dataset empty or sample index out of range")
        return {"error": "Invalid sample"}

    # Get sample
    sample = dataset[sample_idx]

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        return torch.from_numpy(x.copy())

    video_past = to_tensor(sample["video_past"]).unsqueeze(0).to(device)
    video_future_gt = to_tensor(sample["video_future"]).unsqueeze(0).to(device)
    actions_past = to_tensor(sample["actions_past"]).unsqueeze(0).to(device).to(torch.bfloat16)
    actions_future = to_tensor(sample["actions_future"]).unsqueeze(0).to(device).to(torch.bfloat16)

    # Reshape to model format
    video_past = video_past.permute(0, 2, 1, 3, 4) % config.vocab_size
    video_future_gt = video_future_gt.permute(0, 2, 1, 3, 4) % config.vocab_size

    # Mask all future tokens
    video_future_input = torch.full_like(video_future_gt, config.mask_token_id)

    # Forward pass
    with torch.no_grad():
        logits = model(
            v_p_tokens=video_past,
            v_f_tokens=video_future_input,
            a_p=actions_past,
            a_f=actions_future
        )  # (num_factors, B, T_clips, H, W, vocab_size)

    # Get predictions for each factor using temperature-scaled categorical sampling
    num_factors = logits.shape[0]

    # Temperature-scaled categorical sampling (like MaskGiT)
    temperature = 1.2  # >1.0 for more diversity
    logits_scaled = logits / temperature
    original_shape = logits_scaled.shape[:-1]  # (num_factors, B, T_clips, H, W)
    vocab_size = logits_scaled.shape[-1]
    flat_logits = logits_scaled.reshape(-1, vocab_size)
    probs = torch.softmax(flat_logits, dim=-1)
    predicted_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
    predicted_per_factor = predicted_flat.reshape(original_shape)  # (num_factors, B, T_clips, H, W)

    # Also compute argmax predictions for comparison
    argmax_predictions = logits.argmax(dim=-1)

    print(f"\n=== Factor Independence Analysis (T={temperature}) ===")
    
    # Check pairwise agreement between factors
    results = {}
    for i in range(num_factors):
        for j in range(i+1, num_factors):
            same_mask = predicted_per_factor[i] == predicted_per_factor[j]
            agreement_rate = same_mask.float().mean().item()
            results[f"factor_{i}_vs_{j}_agreement"] = agreement_rate
            print(f"  Factor {i} vs Factor {j}: {agreement_rate*100:.1f}% identical predictions")

    # Check if all 3 are identical
    all_same = (predicted_per_factor[0] == predicted_per_factor[1]) & \
               (predicted_per_factor[1] == predicted_per_factor[2])
    all_same_rate = all_same.float().mean().item()
    results["all_three_identical"] = all_same_rate
    print(f"\n  ALL 3 factors identical: {all_same_rate*100:.1f}%")

    # Compare with ground truth
    print(f"\n=== Ground Truth Factor Comparison ===")
    gt_factors = video_future_gt  # (B, num_factors, T_clips, H, W)
    for i in range(num_factors):
        for j in range(i+1, num_factors):
            gt_same = gt_factors[:, i] == gt_factors[:, j]
            gt_agreement = gt_same.float().mean().item()
            print(f"  GT Factor {i} vs Factor {j}: {gt_agreement*100:.1f}% identical")

    gt_all_same = (gt_factors[:, 0] == gt_factors[:, 1]) & \
                  (gt_factors[:, 1] == gt_factors[:, 2])
    print(f"  GT ALL 3 factors identical: {gt_all_same.float().mean().item()*100:.1f}%")

    # Token distribution analysis
    print(f"\n=== Token Distribution Analysis ===")

    # Get all predicted tokens (sampled with temperature)
    pred_all = predicted_per_factor.flatten().float()
    gt_all = video_future_gt.flatten().float()
    argmax_all = argmax_predictions.flatten().float()

    print(f"  Predicted tokens (T={temperature} sampling):")
    print(f"    Mean: {pred_all.mean().item():.1f}, Std: {pred_all.std().item():.1f}")
    print(f"    Min: {pred_all.min().item():.0f}, Max: {pred_all.max().item():.0f}")
    print(f"    Unique values: {len(pred_all.unique())} / {len(pred_all)}")

    print(f"  Predicted tokens (argmax, for comparison):")
    print(f"    Mean: {argmax_all.mean().item():.1f}, Std: {argmax_all.std().item():.1f}")
    print(f"    Min: {argmax_all.min().item():.0f}, Max: {argmax_all.max().item():.0f}")
    print(f"    Unique values: {len(argmax_all.unique())} / {len(argmax_all)}")
    
    print(f"  Ground truth tokens:")
    print(f"    Mean: {gt_all.mean().item():.1f}, Std: {gt_all.std().item():.1f}")
    print(f"    Min: {gt_all.min().item():.0f}, Max: {gt_all.max().item():.0f}")
    print(f"    Unique values: {len(gt_all.unique())} / {len(gt_all)}")
    
    # Check if predictions are collapsed to a narrow range
    pred_range = pred_all.max() - pred_all.min()
    gt_range = gt_all.max() - gt_all.min()
    range_ratio = pred_range / gt_range if gt_range > 0 else 0
    
    results["pred_mean"] = pred_all.mean().item()
    results["pred_std"] = pred_all.std().item()
    results["gt_mean"] = gt_all.mean().item()
    results["gt_std"] = gt_all.std().item()
    results["pred_unique"] = len(pred_all.unique())
    results["gt_unique"] = len(gt_all.unique())
    
    # Token accuracy
    correct = (predicted_per_factor.permute(1, 0, 2, 3, 4) == video_future_gt).float()
    token_accuracy = correct.mean().item()
    print(f"\n  Token accuracy vs GT: {token_accuracy*100:.2f}%")
    results["token_accuracy"] = token_accuracy
    
    # Per-factor accuracy
    for f in range(num_factors):
        factor_acc = (predicted_per_factor[f] == video_future_gt[:, f]).float().mean().item()
        print(f"    Factor {f}: {factor_acc*100:.2f}%")
        results[f"factor_{f}_accuracy"] = factor_acc

    # Interpretation
    print("\n" + "-"*40)
    print("INTERPRETATION:")
    if all_same_rate > 0.5:
        print("  *** WARNING: Model predicts near-identical tokens for all 3 factors! ***")
        print("  This causes 'averaged' appearance when decoded.")
        print("\n  LIKELY CAUSES:")
        print("  1. Loss function averages across factors - model learns 'average' token")
        print("  2. Embedding sum loses factor identity - model can't distinguish factors")
        print("  3. Single shared output projection - all factors share same weights")
        print("\n  RECOMMENDED FIXES:")
        print("  1. Ensure each factor has INDEPENDENT output projection (check output_projs)")
        print("  2. Consider factor-specific positional encoding")
        print("  3. Train with larger/separate factor gradients")
    elif token_accuracy < 0.01:  # Less than 1% accuracy
        print("  *** Model has very low token accuracy (<1%) ***")
        print("  The model is NOT predicting correct tokens yet.")
        print("\n  LIKELY CAUSES:")
        print("  1. Under-training: 13K steps may not be enough for 65536 vocab")
        print("  2. Learning rate too low or model capacity too small")
        print("  3. Data preprocessing issue")
        print("\n  RECOMMENDATIONS:")
        print("  1. Continue training for more steps (50K-100K+)")
        print("  2. Check training loss curve - is it still decreasing?")
        print("  3. Random baseline accuracy = 1/65536 = 0.0015%")
    elif results["pred_std"] < results["gt_std"] * 0.5:
        print("  *** Predicted tokens have COLLAPSED distribution ***")
        print(f"  Pred std ({results['pred_std']:.1f}) << GT std ({results['gt_std']:.1f})")
        print("  Model predicting from narrow token range → blurry output")
    else:
        print("  Factor independence: OK (not the main issue)")
        print(f"  Token accuracy: {token_accuracy*100:.2f}%")
        if token_accuracy < 0.1:
            print("  Low accuracy suggests model needs more training")
        print("  Check training loss curve for convergence")
    print("-"*40)

    return results


# =============================================================================
# STEP 6: MASKGIT ITERATIVE DECODING TEST
# =============================================================================

def test_iterative_decoding(
    checkpoint_path: str,
    data_dir: str,
    tokenizer_dir: str,
    output_dir: Path,
    sample_idx: int = 0,
    k_values: List[int] = [2, 4, 8, 16],  # K=2 is paper default, start with that
    device: str = "cuda",
    config: Optional[MaskedHWMConfig] = None,
    use_test_config: bool = False
) -> Dict[str, Any]:
    """Test MaskGIT iterative decoding with different K values.

    MaskGIT uses iterative parallel decoding:
    1. Start with all tokens masked
    2. Predict all tokens, keep top-K% most confident
    3. Re-mask low confidence tokens
    4. Repeat K times

    This tests if iterative refinement improves output quality.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to dataset
        tokenizer_dir: Path to Cosmos tokenizer
        output_dir: Directory to save outputs
        sample_idx: Which sample to use
        k_values: List of K (iteration) values to test
        device: Device to use
        config: Model config
        use_test_config: Use test config

    Returns:
        Dict with comparison results
    """
    print("\n" + "="*60)
    print("STEP 6: MaskGIT Iterative Decoding Test")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if config is None:
        config = MaskedHWMTestConfig() if use_test_config else MaskedHWMConfig()

    # Load checkpoint
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        config = checkpoint["config"]
        print("  Using config from checkpoint")

    model = MaskedHWM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).to(torch.bfloat16)  # FlashAttention requires bf16/fp16
    model.eval()
    print(f"  Model loaded (step {checkpoint.get('global_step', 'unknown')})")

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = HumanoidWorldModelDataset(
        data_dir=data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=1
    )

    if len(dataset) == 0 or sample_idx >= len(dataset):
        print("ERROR: Dataset empty or sample index out of range")
        return {"error": "Invalid sample"}

    # Load decoder
    print("Loading Cosmos decoder...")
    decoder = load_cosmos_decoder(tokenizer_dir, device)

    # Get sample
    sample = dataset[sample_idx]

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

    B, num_factors, T_clips, H, W = video_future_gt.shape
    num_tokens = T_clips * H * W

    print(f"\nSample {sample_idx}:")
    print(f"  Future tokens: {num_factors} factors x {T_clips} clips x {H}x{W}")
    print(f"  Total tokens per factor: {num_tokens}")

    # Decode ground truth
    print("\nDecoding ground truth...")
    gt_for_decode = video_future_gt.permute(0, 2, 1, 3, 4)  # (B, T_clips, 3, H, W)
    gt_video = decode_video_from_tokens(gt_for_decode[0], decoder, device)
    gt_video_np = tensor_to_numpy_video(gt_video)
    save_video(gt_video_np, output_dir / "iterative_gt.mp4")

    all_videos = [gt_video_np]
    all_labels = ["GT"]
    results = {}

    # MaskGIT cosine schedule for unmasking
    def cosine_schedule(ratio):
        """Cosine schedule: gamma(r) = cos(r * pi/2)"""
        return np.cos(ratio * np.pi / 2)

    for K in k_values:
        print(f"\n--- K={K} iterations ---")

        # Start with all tokens masked
        current_tokens = torch.full_like(video_future_gt, config.mask_token_id)
        mask = torch.ones(B, num_factors, T_clips, H, W, dtype=torch.bool, device=device)

        for iteration in range(K):
            # Ratio of tokens to unmask this iteration
            ratio = (iteration + 1) / K
            num_to_unmask = int(num_tokens * (1 - cosine_schedule(ratio)))

            print(f"  Iteration {iteration+1}/{K}: unmasking {num_to_unmask}/{num_tokens} tokens")

            # Forward pass
            with torch.no_grad():
                logits = model(
                    v_p_tokens=video_past,
                    v_f_tokens=current_tokens,
                    a_p=actions_past,
                    a_f=actions_future
                )  # (num_factors, B, T_clips, H, W, vocab_size)

            # Get predictions using categorical sampling (like MaskGiT)
            # Temperature annealing: start high (diverse), decrease over iterations
            temperature = 1.2 * (1.0 - ratio) + 0.8 * ratio  # 1.2 -> 0.8 over iterations
            logits_scaled = logits / max(temperature, 0.1)
            probs = torch.softmax(logits_scaled, dim=-1)

            # Categorical sampling for predictions
            original_shape = logits_scaled.shape[:-1]  # (num_factors, B, T_clips, H, W)
            vocab_size = logits_scaled.shape[-1]
            flat_logits = logits_scaled.reshape(-1, vocab_size)
            flat_probs = torch.softmax(flat_logits, dim=-1)
            predicted_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            predicted = predicted_flat.reshape(original_shape)

            # Get confidence = probability of the sampled token
            confidence = torch.gather(probs, dim=-1, index=predicted.unsqueeze(-1)).squeeze(-1)

            # Add Gumbel noise to confidence for stochastic masking (like MaskGiT)
            choice_temperature = 4.5 * (1.0 - ratio)  # Anneal noise over iterations
            if choice_temperature > 0:
                gumbel_noise = torch.distributions.gumbel.Gumbel(0, 1).sample(confidence.shape).to(device)
                confidence = torch.log(confidence + 1e-8) + choice_temperature * gumbel_noise

            # For each factor, unmask the most confident predictions
            for f in range(num_factors):
                factor_mask = mask[0, f].flatten().clone()  # (T_clips * H * W)
                factor_conf = confidence[f, 0].flatten()  # (T_clips * H * W)
                factor_pred = predicted[f, 0].flatten()  # (T_clips * H * W)

                # Only consider masked positions
                masked_indices = factor_mask.nonzero(as_tuple=True)[0]

                if len(masked_indices) == 0:
                    continue

                # Get confidence at masked positions
                masked_conf = factor_conf[masked_indices]

                # Select top-k most confident
                num_unmask_this = min(num_to_unmask, len(masked_indices))
                if num_unmask_this > 0:
                    _, topk_local = masked_conf.topk(num_unmask_this)
                    topk_indices = masked_indices[topk_local]

                    # Unmask these positions - clone to avoid in-place modification issues
                    current_flat = current_tokens[0, f].flatten().clone()
                    current_flat[topk_indices] = factor_pred[topk_indices]
                    current_tokens[0, f] = current_flat.view(T_clips, H, W)

                    mask_flat = mask[0, f].flatten().clone()
                    mask_flat[topk_indices] = False
                    mask[0, f] = mask_flat.view(T_clips, H, W)

        # Decode final output
        output_for_decode = current_tokens.permute(0, 2, 1, 3, 4)  # (B, T_clips, 3, H, W)
        output_video = decode_video_from_tokens(output_for_decode[0], decoder, device)
        output_video_np = tensor_to_numpy_video(output_video)

        save_video(output_video_np, output_dir / f"iterative_k{K}.mp4")
        all_videos.append(output_video_np)
        all_labels.append(f"K={K}")

        # Compute accuracy vs GT
        correct = (current_tokens == video_future_gt).float().mean().item()
        results[f"k{K}_accuracy"] = f"{correct*100:.1f}%"
        print(f"  Token accuracy vs GT: {correct*100:.1f}%")

    # Create comparison video
    create_side_by_side_video(
        all_videos,
        all_labels,
        output_dir / "iterative_comparison.mp4"
    )

    # Interpretation
    print("\n" + "-"*40)
    print("INTERPRETATION:")
    print("  Compare iterative_comparison.mp4")
    print("  - K=1: Single pass (no refinement)")
    print("  - K=2: Paper default (2 iterations)")
    print("  - K=4,8: More iterations")
    print("  If higher K improves quality -> Model benefits from refinement")
    print("  If K=1 is already good -> Single pass sufficient")
    print("  If all K values pixelated -> Model quality issue, not algorithm")
    print("-"*40)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Debug Masked-HWM video decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  - Tokenizer roundtrip test (encode -> decode)
  2  - Ground truth token decoding (no model)
  3  - Decoder format verification
  4  - Mask ratio visualization (0%, 25%, 50%, 75%, 100%)
  5  - Factorized token handling strategies (direct, sum, factor0, average)
  5b - Factor independence diagnostic (checks if model predicts same token for all factors)
  6  - MaskGIT iterative decoding (K=2,4,8,16 iterations, K=2 is paper default)

Examples:
  # Run Step 2 (recommended first):
  python debug_masked_hwm_videos.py --step 2

  # Run Step 5 (factorized token strategies):
  python debug_masked_hwm_videos.py --step 5

  # Run Step 5b (IMPORTANT if model output looks like Strategy D):
  python debug_masked_hwm_videos.py --step 5b

  # Run Step 6 (iterative decoding with different K):
  python debug_masked_hwm_videos.py --step 6

  # Run all steps:
  python debug_masked_hwm_videos.py --step all
        """
    )

    parser.add_argument("--step", type=str, required=True,
                       choices=["1", "2", "3", "4", "5", "5b", "6", "all"],
                       help="Which debugging step to run (1-6, 5b for factor diagnosis, or 'all')")
    parser.add_argument("--data_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0",
                       help="Path to dataset directory")
    parser.add_argument("--tokenizer_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer",
                       help="Path to Cosmos tokenizer directory (DV discrete, not CV continuous)")
    parser.add_argument("--checkpoint", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/build/checkpoints_minimal/checkpoint-13500/pytorch_model.bin",
                       help="Path to model checkpoint (required for Step 4)")
    parser.add_argument("--output_dir", type=str,
                       default="/media/skr/storage/robot_world/humanoid_wm/videos/debug_masked_hwm",
                       help="Output directory for videos")
    parser.add_argument("--sample_indices", type=str, default="0,1,2",
                       help="Comma-separated sample indices for Step 2")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--use_test_config", action="store_true",
                       help="Use test config instead of default")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    steps_to_run = ["1", "2", "3", "4", "5", "5b", "6"] if args.step == "all" else [args.step]

    for step in steps_to_run:
        if step == "1":
            test_tokenizer_roundtrip(
                tokenizer_dir=args.tokenizer_dir,
                output_dir=output_dir / "step1_roundtrip",
                device=device
            )

        elif step == "2":
            sample_indices = [int(x) for x in args.sample_indices.split(",")]
            decode_ground_truth_tokens(
                data_dir=args.data_dir,
                tokenizer_dir=args.tokenizer_dir,
                output_dir=output_dir / "step2_gt_decode",
                sample_indices=sample_indices,
                device=device
            )

        elif step == "3":
            verify_decoder_format(
                tokenizer_dir=args.tokenizer_dir,
                output_dir=output_dir / "step3_decoder_format",
                device=device
            )

        elif step == "4":
            if not args.checkpoint:
                print("ERROR: --checkpoint required for Step 4")
                continue

            create_mask_ratio_visualization(
                checkpoint_path=args.checkpoint,
                data_dir=args.data_dir,
                tokenizer_dir=args.tokenizer_dir,
                output_dir=output_dir / "step4_mask_ratios",
                sample_idx=int(args.sample_indices.split(",")[0]),
                device=device,
                use_test_config=args.use_test_config
            )

        elif step == "5":
            test_factorized_token_strategies(
                data_dir=args.data_dir,
                tokenizer_dir=args.tokenizer_dir,
                output_dir=output_dir / "step5_factorized_strategies",
                sample_idx=int(args.sample_indices.split(",")[0]),
                device=device
            )

        elif step == "5b":
            if not args.checkpoint:
                print("ERROR: --checkpoint required for Step 5b")
                continue

            diagnose_factor_independence(
                checkpoint_path=args.checkpoint,
                data_dir=args.data_dir,
                tokenizer_dir=args.tokenizer_dir,
                output_dir=output_dir / "step5b_factor_diagnosis",
                sample_idx=int(args.sample_indices.split(",")[0]),
                device=device,
                use_test_config=args.use_test_config
            )

        elif step == "6":
            if not args.checkpoint:
                print("ERROR: --checkpoint required for Step 6")
                continue

            test_iterative_decoding(
                checkpoint_path=args.checkpoint,
                data_dir=args.data_dir,
                tokenizer_dir=args.tokenizer_dir,
                output_dir=output_dir / "step6_iterative_decoding",
                sample_idx=int(args.sample_indices.split(",")[0]),
                device=device,
                use_test_config=args.use_test_config
            )

    print(f"\n{'='*60}")
    print("DEBUGGING COMPLETE")
    print(f"Outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
