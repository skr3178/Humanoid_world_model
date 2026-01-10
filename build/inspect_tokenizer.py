#!/usr/bin/env python3
"""Inspect Cosmos DV 8×8×8 tokenizer: show indices, shapes, and latent values.

This script demonstrates the tokenization pipeline for the current implementation:
- Input: RGB video (B, 3, T, 256, 256) in range [-1, 1]
- Output: Discrete tokens with factorized format [B, 3, H, W] where H=W=32
- Each clip token represents 17 frames (8× temporal compression)
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
except ImportError:
    print("ERROR: cosmos_tokenizer not found. Please activate the 'cosmos-tokenizer' conda environment.")
    sys.exit(1)

from masked_hwm.config import MaskedHWMConfig


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_tensor_info(name, tensor, show_stats=True):
    """Print detailed information about a tensor."""
    print(f"\n{name}:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    
    if show_stats:
        if tensor.dtype in [torch.int32, torch.int64, torch.long]:
            # Integer tensor (indices)
            print(f"  Min value: {tensor.min().item()}")
            print(f"  Max value: {tensor.max().item()}")
            print(f"  Unique values: {torch.unique(tensor).numel()}")
            if tensor.numel() > 0:
                print(f"  Value range: [{tensor.min().item()} .. {tensor.max().item()}]")
        else:
            # Float tensor
            print(f"  Min value: {tensor.min().item():.6f}")
            print(f"  Max value: {tensor.max().item():.6f}")
            print(f"  Mean: {tensor.mean().item():.6f}")
            print(f"  Std: {tensor.std().item():.6f}")
    
    # Show a sample of values
    if tensor.numel() <= 100:
        print(f"  Values:\n    {tensor.flatten()[:20]}")
    else:
        print(f"  Sample values (first 20):\n    {tensor.flatten()[:20]}")


def main():
    # Load config to get tokenizer path
    config = MaskedHWMConfig()
    tokenizer_dir = Path(config.tokenizer_checkpoint_dir)
    
    print_section("Cosmos DV 8×8×8 Tokenizer Inspection")
    print(f"\nTokenizer checkpoint directory: {tokenizer_dir}")
    print(f"Expected model: Cosmos DV 8×8×8 (spatial: 8×8, temporal: 8×)")
    
    # Check if checkpoint files exist
    encoder_path = tokenizer_dir / "encoder.jit"
    decoder_path = tokenizer_dir / "decoder.jit"
    
    if not encoder_path.exists():
        print(f"\nERROR: Encoder checkpoint not found: {encoder_path}")
        print("\nThe tokenizer models need to be downloaded from HuggingFace.")
        print("To download the Cosmos DV 8×8×8 tokenizer, run:")
        print(f"  huggingface-cli download nvidia/Cosmos-0.1-Tokenizer-DV8x8x8 \\")
        print(f"    --local-dir {tokenizer_dir}/")
        print("\nOr use the setup script:")
        print("  ./setup_server.sh  # This will download the models automatically")
        sys.exit(1)
    
    if not decoder_path.exists():
        print(f"\nERROR: Decoder checkpoint not found: {decoder_path}")
        print("\nThe tokenizer models need to be downloaded from HuggingFace.")
        print("To download the Cosmos DV 8×8×8 tokenizer, run:")
        print(f"  huggingface-cli download nvidia/Cosmos-0.1-Tokenizer-DV8x8x8 \\")
        print(f"    --local-dir {tokenizer_dir}/")
        sys.exit(1)
    
    print(f"✓ Encoder found: {encoder_path}")
    print(f"✓ Decoder found: {decoder_path}")
    
    # Create test input
    # For v2.0 dataset: input is (B, 3, T, 256, 256) in range [-1, 1]
    # Using 17 frames (1 clip) to match the dataset format
    batch_size = 1
    num_frames = 17  # 1 clip after temporal compression
    height, width = 256, 256
    
    print_section("Input Video")
    input_tensor = torch.randn(batch_size, 3, num_frames, height, width).to('cuda').to(torch.bfloat16)
    # Normalize to [-1, 1] range
    input_tensor = torch.clamp(input_tensor, -1.0, 1.0)
    print_tensor_info("Input video", input_tensor)
    print(f"\n  Expected output spatial size: {height // 8} × {width // 8} = 32 × 32")
    print(f"  Temporal compression: {num_frames} frames → 1 clip")
    
    # Load encoder
    print_section("Encoding")
    print("Loading encoder...")
    encoder = CausalVideoTokenizer(
        checkpoint_enc=str(encoder_path),
        device="cuda",
        dtype="bfloat16"
    )
    print("✓ Encoder loaded")
    
    # Encode
    print("\nEncoding video...")
    with torch.no_grad():
        output = encoder.encode(input_tensor)
    
    # Parse output (discrete tokenizer returns (indices, codes))
    if isinstance(output, tuple):
        indices, codes = output
        print(f"✓ Encoder returned tuple: (indices, codes)")
    else:
        # Single tensor output (shouldn't happen for discrete, but handle it)
        indices = output
        codes = None
        print(f"⚠ Encoder returned single tensor (unexpected for discrete tokenizer)")
    
    print_section("Encoded Indices (Discrete Tokens)")
    print_tensor_info("Indices", indices)
    
    # Check expected shape for DV 8×8×8
    # According to Cosmos docs: for DV, indices format depends on model
    # DV4x8x8: (B, 3, H, W) - factorized format with 3 factors
    # DV8x8x8: May be (B, T, H, W) or (B, 3, H, W) depending on implementation
    expected_spatial = height // 8  # 32
    print(f"\n  Expected spatial dimensions: {expected_spatial} × {expected_spatial}")
    
    # Determine format based on shape
    if len(indices.shape) == 4:
        if indices.shape[1] == 3:
            print(f"  ✓ Factorized format detected: [B, 3, H, W] = {tuple(indices.shape)}")
            print(f"    - Factor 0: shape {tuple(indices[0, 0].shape)}, range [{indices[0, 0].min().item()}..{indices[0, 0].max().item()}]")
            print(f"    - Factor 1: shape {tuple(indices[0, 1].shape)}, range [{indices[0, 1].min().item()}..{indices[0, 1].max().item()}]")
            print(f"    - Factor 2: shape {tuple(indices[0, 2].shape)}, range [{indices[0, 2].min().item()}..{indices[0, 2].max().item()}]")
            print(f"    - Each factor has vocab_size = 65536")
            print(f"    - Total possible combinations = 65536³ ≈ 2.8×10¹⁴")
        else:
            print(f"  Standard temporal format: [B, T, H, W] = {tuple(indices.shape)}")
            print(f"    - Temporal dimension: {indices.shape[1]} clips/frames")
            print(f"    - Spatial: {indices.shape[2]} × {indices.shape[3]}")
            print(f"    - This format represents tokens over time")
    else:
        print(f"  ⚠ Unexpected shape: {tuple(indices.shape)}")
    
    # Check value range
    if indices.dtype in [torch.int32, torch.int64, torch.long]:
        min_val = indices.min().item()
        max_val = indices.max().item()
        print(f"\n  Value range: [{min_val} .. {max_val}]")
        if max_val <= 65536:
            print(f"  ✓ Values are within expected range [0..65536] for vocab_size=65536")
        else:
            print(f"  ⚠ Values exceed expected vocab_size=65536")
    
    if codes is not None:
        print_section("Pre-quantization Codes (Continuous Latents)")
        print_tensor_info("Codes", codes)
        print(f"\n  Codes represent the continuous latents before quantization")
        print(f"  Shape indicates: {codes.shape[1]} FSQ levels × {codes.shape[2]} factors")
    
    # Load decoder
    print_section("Decoding")
    print("Loading decoder...")
    decoder = CausalVideoTokenizer(
        checkpoint_dec=str(decoder_path),
        device="cuda",
        dtype="bfloat16"
    )
    print("✓ Decoder loaded")
    
    # Decode
    print("\nDecoding indices back to video...")
    print(f"  Input to decoder: {tuple(indices.shape)}")
    
    with torch.no_grad():
        reconstructed_tensor = decoder.decode(indices)
    
    print_section("Reconstructed Video")
    print_tensor_info("Reconstructed", reconstructed_tensor)
    
    # Verify shapes match
    print_section("Shape Verification")
    print(f"Input shape:     {tuple(input_tensor.shape)}")
    print(f"Indices shape:   {tuple(indices.shape)}")
    if codes is not None:
        print(f"Codes shape:      {tuple(codes.shape)}")
    print(f"Output shape:    {tuple(reconstructed_tensor.shape)}")
    
    # Check if shapes match (accounting for temporal compression)
    input_spatial = input_tensor.shape[-2:]
    output_spatial = reconstructed_tensor.shape[-2:]
    
    if input_spatial == output_spatial:
        print(f"\n✓ Spatial dimensions match: {input_spatial}")
    else:
        print(f"\n⚠ Spatial dimensions differ:")
        print(f"  Input:  {input_spatial}")
        print(f"  Output: {output_spatial}")
    
    # Temporal dimension check
    input_temporal = input_tensor.shape[2]
    output_temporal = reconstructed_tensor.shape[2]
    print(f"\nTemporal dimensions:")
    print(f"  Input:  {input_temporal} frames")
    print(f"  Output: {output_temporal} frames")
    if output_temporal == 17:
        print(f"  ✓ Output has 17 frames per clip (matches dataset format)")
    
    # Compute reconstruction error
    print_section("Reconstruction Quality")
    # Align temporal dimensions for comparison
    min_temporal = min(input_tensor.shape[2], reconstructed_tensor.shape[2])
    input_aligned = input_tensor[:, :, :min_temporal, :, :]
    output_aligned = reconstructed_tensor[:, :, :min_temporal, :, :]
    
    # Resize if spatial dimensions differ
    if input_aligned.shape[-2:] != output_aligned.shape[-2:]:
        output_aligned = torch.nn.functional.interpolate(
            output_aligned.view(-1, 3, *output_aligned.shape[-2:]),
            size=input_aligned.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).view(*output_aligned.shape[:-2], 3, *input_aligned.shape[-2:])
        output_aligned = output_aligned.permute(0, 2, 1, 3, 4)  # Adjust if needed
    
    mse = torch.nn.functional.mse_loss(input_aligned.float(), output_aligned.float())
    psnr = -10 * torch.log10(mse + 1e-10)
    
    print(f"MSE:  {mse.item():.6f}")
    print(f"PSNR: {psnr.item():.2f} dB")
    
    # Show sample values
    print_section("Sample Values")
    print("\nSample indices (first batch, top-left 5×5):")
    if indices.shape[1] == 3:
        # Factorized format
        print("  Factor 0:")
        print(indices[0, 0, :5, :5].cpu().numpy())
        print("  Factor 1:")
        print(indices[0, 1, :5, :5].cpu().numpy())
        print("  Factor 2:")
        print(indices[0, 2, :5, :5].cpu().numpy())
    else:
        # Temporal format
        print("  First temporal slice:")
        print(indices[0, 0, :5, :5].cpu().numpy())
    
    # Show how indices are used in the model
    print_section("Usage in Model Pipeline")
    print("\n1. Dataset Loading:")
    print(f"   - Tokens stored as: [num_clips, 3, 32, 32] in binary format")
    print(f"   - Each clip represents {num_frames} frames (8× temporal compression)")
    
    print("\n2. Model Input Format:")
    if indices.shape[1] == 3:
        print(f"   - Factorized format: [B, 3, H, W] = {tuple(indices.shape)}")
        print(f"   - Each of 3 factors embedded separately")
        print(f"   - Embeddings summed: embed_factor0 + embed_factor1 + embed_factor2")
    else:
        print(f"   - Temporal format: [B, T, H, W] = {tuple(indices.shape)}")
        print(f"   - Transposed to [B, T, H, W] for model input")
    
    print("\n3. Decoding:")
    print(f"   - Decoder input: {tuple(indices.shape)}")
    if indices.shape[1] == 3:
        print(f"   - Pass all 3 factors directly: decoder.decode([B, 3, H, W])")
        print(f"   - Output: [B, 3, 17, 256, 256] (17 frames per clip)")
    else:
        print(f"   - Decoder processes temporal dimension")
        print(f"   - Output: [B, 3, T, 256, 256]")
    
    print("\n✓ Tokenizer inspection complete!")
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Input shape:     (B={batch_size}, C=3, T={num_frames}, H={height}, W={width})")
    print(f"  Indices shape:   {tuple(indices.shape)} (discrete tokens)")
    if codes is not None:
        print(f"  Codes shape:      {tuple(codes.shape)} (pre-quantization latents)")
    print(f"  Output shape:    {tuple(reconstructed_tensor.shape)} (reconstructed video)")
    print(f"  Spatial compression: {height}×{width} → {indices.shape[-1]}×{indices.shape[-2]} = {height//indices.shape[-1]}×")
    print(f"  Temporal compression: {num_frames} frames → 1 clip")
    if indices.shape[1] == 3:
        print(f"  Format: Factorized (3 factors per spatial position)")
        print(f"  Vocab per factor: 65536")
    else:
        print(f"  Format: Temporal (tokens over time)")
    print("=" * 80)


if __name__ == "__main__":
    main()
