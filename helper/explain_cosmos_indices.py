#!/usr/bin/env python3
"""
Comprehensive explanation of Cosmos Tokenizer DV 8×8×8 transformation:
- Input video structure and values
- Encoding process step-by-step
- FSQ quantization with levels [8,8,8,5,5,5]
- What indices represent
- How factorized tokens work
"""

import torch
import numpy as np
from pathlib import Path
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

print("="*80)
print("COSMOS TOKENIZER DV 8×8×8: COMPLETE TRANSFORMATION EXPLANATION")
print("="*80)

# ============================================================================
# STEP 1: UNDERSTANDING THE INPUT
# ============================================================================
print("\n" + "="*80)
print("STEP 1: INPUT VIDEO STRUCTURE")
print("="*80)

print("\n1.1 Input Video Format:")
print("   - Shape: [B, C, T, H, W]")
print("   - B = Batch size (e.g., 1)")
print("   - C = Channels = 3 (RGB: Red, Green, Blue)")
print("   - T = Temporal frames (e.g., 9 frames)")
print("   - H = Height (e.g., 512 pixels)")
print("   - W = Width (e.g., 512 pixels)")

# Create example input
example_input = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)
# Clamp to valid RGB range [-1, 1] (Cosmos expects normalized values)
example_input = torch.clamp(example_input, -1.0, 1.0)

print(f"\n1.2 Example Input:")
print(f"   Shape: {example_input.shape}")
print(f"   Value range: [{example_input.min():.3f}, {example_input.max():.3f}]")
print(f"   Dtype: {example_input.dtype}")
print("\n   What each value represents:")
print("   - Each value is a normalized pixel intensity in range [-1, 1]")
print("   - -1.0 = minimum intensity (dark)")
print("   - +1.0 = maximum intensity (bright)")
print("   - 0.0 = middle intensity")
print(f"   - Total pixels per frame: {512 * 512} = {512*512:,}")
print(f"   - Total values in video: {np.prod(example_input.shape):,}")

# ============================================================================
# STEP 2: ENCODING PROCESS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: ENCODING PROCESS (Video -> Latent -> Tokens)")
print("="*80)

encoder_path = Path("/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer/encoder.jit")
if not encoder_path.exists():
    print(f"\n⚠️  Encoder not found at {encoder_path}")
    print("   Cannot demonstrate encoding. Using theoretical explanation.")
else:
    encoder = CausalVideoTokenizer(checkpoint_enc=str(encoder_path), device="cuda", dtype="bfloat16")
    
    print("\n2.1 Encoder Architecture:")
    print("   The encoder performs:")
    print("   a) Haar Wavelet Transform (2-level)")
    print("      - Down-samples by 4× in spatial dimensions")
    print("      - Down-samples by 4× in temporal dimension")
    print("      - 512×512 → 128×128 (spatial)")
    print("      - 9 frames → ~2-3 frames (temporal)")
    
    print("\n   b) Convolutional Downsampling")
    print("      - Additional spatial compression: 16× total")
    print("      - 512×512 → 32×32 (spatial)")
    print("      - Temporal compression: 8× total")
    print("      - 9 frames → ~1-2 frames (temporal)")
    
    print("\n   c) Feature Extraction")
    print("      - Extracts high-level features")
    print("      - Output channels: 16 (z_channels)")
    
    with torch.no_grad():
        (indices, codes) = encoder.encode(example_input)
    
    print(f"\n2.2 Encoder Output:")
    print(f"   codes.shape: {codes.shape}")
    print(f"   indices.shape: {indices.shape}")
    print(f"\n   codes = Pre-quantization continuous latent")
    print(f"   - Shape: [B, 6, T_compressed, H_compressed, W_compressed]")
    print(f"   - 6 channels = 6 FSQ quantization levels")
    print(f"   - Values are continuous (float)")
    
    print(f"\n   indices = Quantized discrete tokens")
    print(f"   - Shape: [B, T_compressed, H_compressed, W_compressed]")
    print(f"   - Values are integers (indices into codebook)")
    print(f"   - Range: [0, codebook_size-1] where codebook_size = 8×8×8×5×5×5 = {8*8*8*5*5*5:,}")
    
    # Show actual values
    print(f"\n2.3 Sample Values:")
    print(f"   codes min/max: [{codes.min():.3f}, {codes.max():.3f}]")
    print(f"   indices min/max: [{indices.min()}, {indices.max()}]")
    print(f"   indices unique values: {len(torch.unique(indices))} different indices")
    print(f"   Sample indices (first 5×5 spatial region, first temporal frame):")
    print(f"   {indices[0, 0, :5, :5].cpu().numpy()}")

# ============================================================================
# STEP 3: FSQ QUANTIZATION EXPLAINED
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FSQ (Finite Scalar Quantization) - The '8×8×8' Part")
print("="*80)

print("\n3.1 What is FSQ?")
print("   FSQ = Finite Scalar Quantization")
print("   - Instead of learning a codebook, uses fixed quantization levels")
print("   - More stable and simpler than VQ-VAE")
print("   - No codebook collapse issues")

print("\n3.2 Quantization Levels: [8, 8, 8, 5, 5, 5]")
print("   The 'DV 8×8×8' name refers to the first 3 levels, but full config is:")
print("   - Level 1: 8 possible values")
print("   - Level 2: 8 possible values")
print("   - Level 3: 8 possible values")
print("   - Level 4: 5 possible values")
print("   - Level 5: 5 possible values")
print("   - Level 6: 5 possible values")
print(f"   - Total combinations: 8×8×8×5×5×5 = {8*8*8*5*5*5:,} possible tokens")

print("\n3.3 How Quantization Works:")
print("   For each spatial position (x, y) at time t:")
print("   1. Encoder outputs 6 continuous values (one per FSQ level)")
print("   2. Each value is quantized to nearest discrete level:")
print("      - Levels 1-3: quantized to {0, 1, 2, 3, 4, 5, 6, 7} (8 options)")
print("      - Levels 4-6: quantized to {0, 1, 2, 3, 4} (5 options)")
print("   3. These 6 quantized values are combined into a single index:")
print("      index = level1 + level2×8 + level3×8² + level4×8³ + level5×8³×5 + level6×8³×5²")
print(f"      This gives unique index in range [0, {8*8*8*5*5*5-1}]")

print("\n3.4 What the Index Represents:")
print("   Each index is a unique combination of 6 quantized values.")
print("\n   Example Calculation:")
print("   - If quantized levels = [3, 5, 2, 1, 4, 0]")
print("   - Index = 3 + 5×8 + 2×64 + 1×512 + 4×2560 + 0×12800")
print("   - Index = 3 + 40 + 128 + 512 + 10240 + 0 = 10,923")
print("   - This index uniquely identifies this combination")
print("\n   Reverse Calculation (Index → Levels):")
print("   - Given index = 10,923")
print("   - level1 = 10923 % 8 = 3")
print("   - level2 = (10923 // 8) % 8 = 5")
print("   - level3 = (10923 // 64) % 8 = 2")
print("   - level4 = (10923 // 512) % 5 = 1")
print("   - level5 = (10923 // 2560) % 5 = 4")
print("   - level6 = (10923 // 12800) % 5 = 0")
print("   - Result: [3, 5, 2, 1, 4, 0] ✓")

# Demonstrate with actual index from encoding if available
if encoder_path.exists() and 'indices' in locals():
    print("\n3.5 Real Example from Encoding:")
    sample_idx = indices[0, 0, 0, 0].item()
    print(f"   Sample index: {sample_idx}")
    # Decode index to levels
    level1 = sample_idx % 8
    level2 = (sample_idx // 8) % 8
    level3 = (sample_idx // 64) % 8
    level4 = (sample_idx // 512) % 5
    level5 = (sample_idx // 2560) % 5
    level6 = (sample_idx // 12800) % 5
    print(f"   Decoded to FSQ levels: [{level1}, {level2}, {level3}, {level4}, {level5}, {level6}]")
    print(f"   These 6 levels represent the quantized feature vector")
    print(f"   at this spatial position and time step.")

# ============================================================================
# STEP 4: FACTORIZED TOKENS (Dataset Format)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: FACTORIZED TOKENS (How Dataset Stores Tokens)")
print("="*80)

print("\n4.1 The Problem with Direct Storage:")
print("   - Direct index storage: [B, T, H, W] with values 0 to 64,000")
print("   - Very large vocabulary size (64K)")
print("   - Hard to model with transformers")

print("\n4.2 Dataset Storage Format:")
print("   The dataset stores tokens as [num_clips, 3, 32, 32]")
print("   - The '3' dimension represents 3 separate token streams")
print("   - Each value is a full index (0 to 64,000)")
print("   - These are NOT small factorized values, but full indices")
print("   - The decoder expects this format and processes all 3 streams")

print("\n4.3 Why 3 Channels?")
print("   The decoder is designed to handle 3 separate token maps:")
print("   - This allows modeling different aspects of the video")
print("   - Each channel can represent different temporal or semantic information")
print("   - The decoder combines them during reconstruction")

print("\n4.4 Index Values:")
print("   - Each index is an integer from 0 to 64,000")
print("   - Represents a unique combination of 6 FSQ quantization levels")
print("   - Formula: index = level1 + level2×8 + level3×64 + level4×512 + level5×2560 + level6×12800")
print("   - Example: index 19113 represents a specific quantized feature combination")

# ============================================================================
# STEP 5: DECODING PROCESS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: DECODING PROCESS (Tokens -> Video)")
print("="*80)

print("\n5.1 Decoder Input:")
print("   - For dataset tokens: [B, 3, H, W] (3 factorized tokens)")
print("   - Decoder converts factors back to full indices")
print("   - Then converts indices back to 6 FSQ levels")
print("   - Then converts levels to continuous latent codes")

print("\n5.2 Decoder Architecture:")
print("   a) Inverse Quantization (inv_quant)")
print("      - Converts indices → 6 FSQ levels → continuous codes")
print("   b) Post-quantization convolution")
print("      - Processes the continuous codes")
print("   c) Decoder network")
print("      - Upsamples spatially: 32×32 → 256×256")
print("      - Upsamples temporally: 1-2 frames → 17 frames")
print("   d) Inverse Haar Wavelet")
print("      - Final upsampling to match input resolution")

# ============================================================================
# STEP 6: COMPRESSION RATIOS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: COMPRESSION SUMMARY")
print("="*80)

input_size = np.prod([1, 3, 9, 512, 512])
print(f"\n6.1 Input Size:")
print(f"   Values: {input_size:,} (1×3×9×512×512)")

# Calculate compressed size
spatial_comp = 16  # 512/32
temporal_comp = 8  # approximate
compressed_spatial = 512 // spatial_comp
compressed_temporal = 9 // temporal_comp
compressed_size = np.prod([1, compressed_temporal, compressed_spatial, compressed_spatial])
print(f"\n6.2 Compressed Latent Size:")
print(f"   Spatial: 512×512 → {compressed_spatial}×{compressed_spatial} ({spatial_comp}× compression)")
print(f"   Temporal: 9 frames → ~{compressed_temporal} frames ({temporal_comp}× compression)")
print(f"   Values: ~{compressed_size:,}")

print(f"\n6.3 Token Size (after quantization):")
print(f"   Same spatial/temporal dimensions as compressed latent")
print(f"   But each value is an integer index (much smaller than float)")

compression_ratio = input_size / compressed_size
print(f"\n6.4 Overall Compression:")
print(f"   Ratio: ~{compression_ratio:.1f}×")
print(f"   {input_size:,} values → ~{compressed_size:,} tokens")

# ============================================================================
# STEP 7: PRACTICAL EXAMPLE
# ============================================================================
print("\n" + "="*80)
print("STEP 7: PRACTICAL EXAMPLE WITH REAL DATA")
print("="*80)

try:
    # Load from dataset
    data_dir = Path("/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0")
    import json
    metadata_path = data_dir / "metadata" / "metadata_0.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        video_path = data_dir / "videos" / "video_0.bin"
        if video_path.exists():
            import math
            total_frames = metadata["shard_num_frames"]
            num_clips = math.ceil(total_frames / 17)
            video_tokens = np.memmap(video_path, dtype=np.int32, mode="r", 
                                   shape=(num_clips, 3, 32, 32))
            
            sample_clip = np.array(video_tokens[0])  # [3, 32, 32]
            
            print(f"\n7.1 Dataset Sample:")
            print(f"   Shape: {sample_clip.shape}")
            print(f"   Factor 0 (first 2 FSQ levels):")
            print(f"     Shape: {sample_clip[0].shape}")
            print(f"     Value range: [{sample_clip[0].min()}, {sample_clip[0].max()}]")
            print(f"     Sample values (5×5):")
            print(f"     {sample_clip[0, :5, :5]}")
            
            print(f"\n   Factor 1 (next 2 FSQ levels):")
            print(f"     Value range: [{sample_clip[1].min()}, {sample_clip[1].max()}]")
            print(f"     Sample values (5×5):")
            print(f"     {sample_clip[1, :5, :5]}")
            
            print(f"\n   Factor 2 (last 2 FSQ levels):")
            print(f"     Value range: [{sample_clip[2].min()}, {sample_clip[2].max()}]")
            print(f"     Sample values (5×5):")
            print(f"     {sample_clip[2, :5, :5]}")
            
            print(f"\n7.2 What These Values Mean:")
            print(f"   At position (0,0):")
            print(f"     Factor 0 = {sample_clip[0, 0, 0]}")
            print(f"     Factor 1 = {sample_clip[1, 0, 0]}")
            print(f"     Factor 2 = {sample_clip[2, 0, 0]}")
            print(f"\n   NOTE: The dataset appears to store full indices (0-64K range)")
            print(f"   rather than small factorized values. The decoder handles this")
            print(f"   by interpreting the 3 channels as separate token streams.")
            print(f"   Each value is a full index into the 64K codebook.")
            print(f"   The '3' dimension represents 3 separate token maps that")
            print(f"   together encode the video content at each spatial position.")
            
except Exception as e:
    print(f"\n⚠️  Could not load dataset example: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
1. INPUT: RGB video [B, 3, T, H, W] with pixel values in [-1, 1]

2. ENCODING:
   - Spatial compression: 512×512 → 32×32 (16×)
   - Temporal compression: 9 frames → ~2 frames (8×)
   - Feature extraction: 3 RGB channels → 16 feature channels

3. QUANTIZATION (FSQ):
   - 6 continuous values → 6 discrete levels
   - Levels: [8, 8, 8, 5, 5, 5]
   - Combined into single index: 0 to 64,000

4. FACTORIZATION (Dataset Format):
   - Large index (0-64K) → 3 smaller factors
   - Factor 0: 0-63 (first 2 levels)
   - Factor 1: 0-39 (next 2 levels)
   - Factor 2: 0-24 (last 2 levels)

5. DECODING:
   - Factors → indices → FSQ levels → continuous codes → video
   - Upsamples: 32×32 → 256×256, ~2 frames → 17 frames

Each index value represents a unique combination of 6 quantized feature levels
that encode the visual content at a specific spatial position and time step.
""")

print("="*80)
