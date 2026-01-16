import torch
import numpy as np
import json
import math
from pathlib import Path
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

# Dataset path
data_dir = Path("/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0")
shard_idx = 0

# Load metadata for the shard
metadata_path = data_dir / "metadata" / f"metadata_{shard_idx}.json"
with open(metadata_path, "r") as f:
    metadata_shard = json.load(f)

total_frames = metadata_shard["shard_num_frames"]
num_clips = math.ceil(total_frames / 17)

# Load video tokens from dataset (already factorized: [num_clips, 3, 32, 32])
video_path = data_dir / "videos" / f"video_{shard_idx}.bin"
video_tokens = np.memmap(video_path, dtype=np.int32, mode="r", shape=(num_clips, 3, 32, 32))

print(f"Loaded video tokens from shard {shard_idx}")
print(f"  Shape: {video_tokens.shape}")  # [num_clips, 3, 32, 32]
print(f"  Total clips: {num_clips}")
print(f"  Total frames represented: {num_clips * 17}")

# Select a sample clip (first clip)
sample_clip = np.array(video_tokens[0])  # [3, 32, 32] - copy to make writable
print(f"\nSample clip shape: {sample_clip.shape}")

# Convert to torch tensor and add batch dimension
sample_tokens = torch.from_numpy(sample_clip).long().cuda().unsqueeze(0)  # [1, 3, 32, 32]
print(f"Sample tokens tensor shape: {sample_tokens.shape}")

# Initialize decoder
decoder_path = Path("/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer/decoder.jit")
decoder = CausalVideoTokenizer(checkpoint_dec=str(decoder_path), device="cuda", dtype="bfloat16")

# Decode the sample tokens
print("\nDecoding sample tokens...")
with torch.no_grad():
    reconstructed_video = decoder.decode(sample_tokens)  # [1, 3, 17, 256, 256]

print(f"Reconstructed video shape: {reconstructed_video.shape}")
print(f"Expected: [1, 3, 17, 256, 256]")
torch.testing.assert_close(
    reconstructed_video.shape, 
    torch.Size([1, 3, 17, 256, 256]),
    msg="Reconstructed video shape mismatch"
)

print("\n✓ Successfully decoded video tokens from dataset!")
print(f"  Input tokens shape: {sample_tokens.shape}")  # [1, 3, 32, 32]
print(f"  Output video shape: {reconstructed_video.shape}")  # [1, 3, 17, 256, 256]

print("\n" + "="*80)
print("COMPARISON: Dataset tokens vs Encoding output")
print("="*80)

print("\n1. DATASET TOKENS (from train_v2.0):")
print(f"   Shape: {sample_tokens.shape}")  # [1, 3, 32, 32]
print("   The '3' dimension = 3 FACTORIZED TOKENS")
print("   - These are 3 separate token maps (factors) per spatial position")
print("   - Each factor represents part of the quantized representation")
print("   - All 3 factors must be passed together to the decoder")
print("   - This is the format stored in the v2.0 dataset")

print("\n2. ENCODING OUTPUT (from encoder.encode()):")
print("   Testing encoding to see the difference...")
try:
    encoder_path = Path("/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer/encoder.jit")
    if encoder_path.exists():
        encoder = CausalVideoTokenizer(checkpoint_enc=str(encoder_path), device="cuda", dtype="bfloat16")
        # Encode a sample video (9 frames, 512x512)
        test_input = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
        print(f"   Input video shape: {test_input.shape}")  # [1, 3, 9, 512, 512]
        
        (indices, codes) = encoder.encode(test_input)
        print(f"   indices.shape: {indices.shape}")
        print(f"   codes.shape: {codes.shape}")
        print(f"   The second dimension in indices.shape = TEMPORAL FRAMES (compressed from input)")
        print(f"   - Input had {test_input.shape[2]} frames, compressed to {indices.shape[1]} frames")
        print("   - This represents time steps in the compressed temporal dimension")
        print("   - The decoder can decode this back to the original video")
        
        print("\n   KEY DIFFERENCE:")
        print("   - Dataset [1, 3, 32, 32]: 3 = FACTORIZED TOKENS (spatial factors)")
        print(f"   - Encoding {indices.shape}: second dim = TEMPORAL FRAMES (time dimension)")
        print("   - Dataset '3' = 3 separate token maps per spatial position (factors)")
        print("   - Encoding second dim = number of temporal frames (varies with input)")
        print("   - They represent completely different things!")
    else:
        print(f"   Encoder not found at {encoder_path}")
        print("   Skipping encoding test")
except Exception as e:
    print(f"   Could not test encoding: {e}")

print("\n" + "="*80)







# Example: Encoding a video (for comparison)
# This shows what happens when you encode a video from scratch:
# 
# encoder_path = Path("/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer/encoder.jit")
# encoder = CausalVideoTokenizer(checkpoint_enc=str(encoder_path), device="cuda", dtype="bfloat16")
# input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
# (indices, codes) = encoder.encode(input_tensor)
# # indices.shape = (1, 3, 64, 64) where 3 = TEMPORAL FRAMES
# # codes.shape = (1, 6, 3, 64, 64) where 6 = FSQ levels, 3 = temporal frames