# Masked-HWM: Masked Humanoid World Model with Shared Parameters

Implementation of Masked-HWM (Masked Humanoid World Model) with shared parameters, following the specifications from the Humanoid World Models paper.

## Architecture

- **Model**: Masked-HWM with modality-based parameter sharing
- **Tokenizer**: Cosmos DV 8×8×8 (256×256 → 32×32 spatial), factorized vocab (3 × 65536)
  - **Decoding**: Pass all 3 factors directly as `[B, 3, H, W]` to decoder - do NOT use only Factor 0
- **Input**: 9 past frames + 9 past actions + 8 future actions (downsampled to clips)
- **Output**: 8 future frames (predicted in latent space at clip level)
- **Transformer**: 24 layers, 8 heads, 512 dim, 2048 MLP hidden
- **Parameter Sharing**: First 4 layers unshared, layers 5-24 modality-shared
- **Dataset**: v2.0 format with 25-dim actions (matches paper's R^25)

### Key Features

1. **4-Stream Architecture**: Past video (v_p), future video (v_f), past actions (a_p), future actions (a_f)
2. **Factorized Attention with RoPE**: 
   - Spatial attention: Applied only to video tokens (per-frame) with **2D RoPE**
   - Temporal attention: Joint across all 4 streams with **1D RoPE**
3. **Parameter Sharing**:
   - Layers 0-3: No sharing (separate params per stream)
   - Layers 4-23: Modality sharing (video streams share, action streams share)
4. **Action Conditioning**: 25-dim action vectors embedded to 512-dim tokens (paper's R^25)
5. **Masked Video Generation**: MaskGIT-style iterative parallel decoding
6. **Factorized Token Embeddings**: 3 separate embeddings per spatial position, summed

### Implementation Design Choices

This implementation makes several practical engineering choices that differ from a strict literal reading of the paper figure:

1. **Spatial Mean-Pooling Before Temporal Attention** (lines 196-198 in `transformer.py`):
   - Video tokens are spatially averaged before joint temporal attention
   - Reduces sequence length from `(T_p + T_f) × H×W` to `(T_p + T_f)` for computational efficiency
   - Output is broadcast back to spatial dimensions after temporal attention
   - **Rationale**: Makes training/inference feasible with 32×32 spatial tokens

2. **Single Shared QKV Projection** (line 155 in `transformer.py`):
   - Temporal attention uses one QKV projection for all streams
   - Stream identity is implicit via position in concatenated sequence `[v_p, v_f, a_p, a_f]`
   - **Optional Enhancement**: Set `use_stream_type_emb=True` in config to add explicit stream-type embeddings
   - **Rationale**: Simpler, fewer parameters; stream identity learned via position + RoPE

3. **Progressive Parameter Sharing**:
   - First 4 layers: No sharing (separate params per stream) - matches paper figure
   - Remaining layers: Modality sharing (video streams share, action streams share)
   - **Rationale**: Documented in build2.md as improving stability while preserving low-level features

## v2.0 Dataset Format

The v2.0 dataset uses Cosmos DV 8×8×8 tokenizer with temporal compression:

- **Video tokens**: Shape `[num_clips, 3, 32, 32]` with int32 dtype
  - Each clip contains 17 temporally-compressed frames
  - 3 factorized tokens per spatial position (vocab ~65536 each)
  - **Important**: The 3 factors must be passed directly to the Cosmos decoder as `[B, 3, H, W]` - do NOT combine them into a single token
- **States**: Shape `[num_frames, 25]` with float32 dtype
- **Segment indices**: Shape `[num_frames]` with int32 dtype - indicates independent video segments/clips

### Clip-based Processing

Since video is stored at clip level (17 frames per clip), the model:
1. Works with clips as the temporal unit for video
2. Downsamples frame-level actions to match clip rate (via averaging)

### Understanding Clips and Segments

**Clips** are defined by the Cosmos tokenizer:
- Each clip token `[3, 32, 32]` decodes to 17 frames of video
- Clips are stored sequentially in the dataset
- Clip boundaries occur every 17 frames (frames 0, 17, 34, 51, ...)

**Segments** are independent video episodes:
- Defined by `segment_idx_{shard}.bin` - each frame has a segment ID
- Segment boundaries indicate where one independent episode ends and another begins
- In practice, clip boundaries align with segment boundaries (each clip typically contains one complete segment)

**Important**: When decoding multiple clips, the resulting video contains fragments of continuous clips - each clip is continuous within itself, but there are jumps/discontinuities between clips because they represent different independent segments.

## Action Format (v2.0 - matches paper's R^25)

25-dimensional state vector:
- Indices 0-20: Joint positions (21 dims)
  - HIP (yaw, roll, pitch), KNEE (pitch), ANKLE (roll, pitch)
  - LEFT: SHOULDER (pitch, roll, yaw), ELBOW (pitch, yaw), WRIST (pitch, roll)
  - RIGHT: SHOULDER (pitch, roll, yaw), ELBOW (pitch, yaw), WRIST (pitch, roll)
  - NECK (pitch)
- Index 21: Left hand closure (0=open, 1=closed)
- Index 22: Right hand closure (0=open, 1=closed)
- Index 23: Linear Velocity
- Index 24: Angular Velocity

## Training

Per the paper:
- **Optimizer**: AdamW
- **Learning Rate**: 3e-5 with cosine schedule
- **Warmup**: 100 steps
- **Batch Size**: 128
- **Steps**: 60,000

## Usage

```python
from masked_hwm.config import MaskedHWMConfig
from masked_hwm.model import MaskedHWM
from data.dataset import HumanoidWorldModelDataset

# Load config and model
config = MaskedHWMConfig()

# Optional: Enable stream-type embeddings for explicit stream identity
# config.use_stream_type_emb = True

model = MaskedHWM(config)

# Load dataset (v2.0 format)
dataset = HumanoidWorldModelDataset(config.train_data_dir)

# Get a sample
sample = dataset[0]
v_p = sample['video_past']  # (num_clips, 3, 32, 32)
v_f = sample['video_future']  # (num_clips, 3, 32, 32)
a_p = sample['actions_past']  # (T_p_frames, 25)
a_f = sample['actions_future']  # (T_f_frames, 25)

# Prepare for model (need batch dim and factors first)
v_p = v_p.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, 3, num_clips, H, W)
v_f = v_f.unsqueeze(0).permute(0, 2, 1, 3, 4)
a_p = a_p.unsqueeze(0)
a_f = a_f.unsqueeze(0)

# Forward pass
logits = model(v_p, v_f, a_p, a_f)  # (3, B, T_f_clips, H, W, vocab_size)
```

## Decoding Video Tokens

**CRITICAL**: When decoding factorized tokens from the v2.0 dataset, you must pass all 3 factors directly to the Cosmos decoder:

```python
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
import torch

# Load tokens from dataset: shape [num_clips, 3, 32, 32]
factored_tokens = torch.from_numpy(video_tokens[clip_idx]).long().cuda()  # [3, 32, 32]

# Add batch dimension
factored_tokens = factored_tokens.unsqueeze(0)  # [1, 3, 32, 32]

# Decode - pass ALL 3 factors directly (not just Factor 0!)
decoder = CausalVideoTokenizer(
    checkpoint_dec="cosmos_tokenizer/decoder.jit",
    device="cuda",
    dtype="bfloat16"
)
video = decoder.decode(factored_tokens)  # [1, 3, 17, 256, 256] - 17 frames per clip
```

**DO NOT**:
- Use only Factor 0: `decoder.decode(factored_tokens[0])` ❌
- Combine factors into single token ❌

**DO**:
- Pass all 3 factors: `decoder.decode(factored_tokens)` where `factored_tokens.shape = [B, 3, H, W]` ✅

See `1xgpt/data/cosmos_video_decoder.py` for the official reference implementation.

## Data Paths

- **Tokenizer**: `/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer/`
- **Dataset** (v2.0): `/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/`
  - `train_v2.0/`: Training data (100 shards, ~11M frames)
  - `val_v2.0/`: Validation data
- **Source**: [HuggingFace - 1x-technologies/world_model_tokenized_data](https://huggingface.co/datasets/1x-technologies/world_model_tokenized_data)

## Quick Test

```bash
# Test with reduced config (fast, uses less memory)
python test_model.py

# Test with full config (requires ~32GB+ RAM)
python test_model.py --full
```

## Training and Testing Pipeline

### Quick Test on Subset

Train on the test subset and automatically generate comparison videos:

```bash
./train_test_subset.sh
```

This will:
1. Train the model (50 steps on test subset)
2. Save checkpoints
3. Automatically run video generation test
4. Create side-by-side comparison videos: `checkpoints_test_subset/test_videos/`
   - `sample_*_comparison.mp4` - Side-by-side: Ground Truth | Predicted

### Manual Video Generation Test

After training, test video generation from a checkpoint:

```bash
python3 test_generate_video.py \
    --checkpoint checkpoints_test_subset/checkpoint-final/pytorch_model.bin \
    --data_dir /media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0_test \
    --output_dir ./test_output \
    --tokenizer_dir /media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer \
    --num_samples 3 \
    --use_test_config
```

See `README_TESTING.md` for more details on the testing pipeline.

## References

- [Humanoid World Models Paper](https://arxiv.org/abs/2505.07840)
- [1X World Model Challenge](https://github.com/1x-technologies/1xgpt)
- [Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer)
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864)


Initial inputs
    ↓
[Learned token embeddings + initial positional information]
    ↓
          ┌──────────────────────────────┐
          │           Layer 1            │   ← one full "Base Block" (the entire Figure 2)
          │   (Spatial → Joint Temporal → 4× specialized MLPs)   │
          └──────────────────────────────┘
                   │   (black bold residual paths)
                   ▼
          ┌──────────────────────────────┐
          │           Layer 2            │   ← exactly the same block design again
          └──────────────────────────────┘
                   │
                   ▼
          ┌──────────────────────────────┐
          │           Layer 3            │
          └──────────────────────────────┘
                   │
                   ▼
                   ⋮
                   ▼
          ┌──────────────────────────────┐
          │          Layer 24            │   ← or however many layers the model has (12, 24, 36…)
          └──────────────────────────────┘
                   │
                   ▼
Final layer norm (usually)
                   ↓
Model heads / predictions
    (next video token logits, action predictions, etc.)


After each layer finishes (i.e. after the four specialized MLPs + their residuals), you get back exactly the same four groups of tokens, but now refined/updated:

Past Video Tokens (cleaned up, more contextualized)
Predicted Future Video Tokens (better denoising / prediction of masked/noisy future frames)
Past Actions (processed, now aware of more context)
Future Actions (better future action prediction)