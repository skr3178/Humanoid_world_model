"""Dataset loader for Humanoid World Model (v2.0 format).

v2.0 format:
- Video: Cosmos DV 8×8×8 tokens, shape [num_clips, 3, 32, 32]
  - Each clip contains 17 temporally-compressed frames
  - 3 factorized tokens per spatial position (vocab ~65536 each)
  - **Decoding**: Pass all 3 factors directly as [B, 3, H, W] to Cosmos decoder
- States: np.float32, shape [num_frames, 25]
- Segment indices: np.int32, shape [num_frames] - indicates independent video segments
  - Clip boundaries typically align with segment boundaries (each clip = one segment)
- 25 dimensions match paper's R^25 specification

State Index Definition (v2.0):
 0: HIP_YAW, 1: HIP_ROLL, 2: HIP_PITCH
 3: KNEE_PITCH
 4: ANKLE_ROLL, 5: ANKLE_PITCH
 6-8: LEFT_SHOULDER (PITCH, ROLL, YAW)
 9-10: LEFT_ELBOW (PITCH, YAW)
 11-12: LEFT_WRIST (PITCH, ROLL)
 13-15: RIGHT_SHOULDER (PITCH, ROLL, YAW)
 16-17: RIGHT_ELBOW (PITCH, YAW)
 18-19: RIGHT_WRIST (PITCH, ROLL)
 20: NECK_PITCH
 21: Left hand closure (0=open, 1=closed)
 22: Right hand closure (0=open, 1=closed)
 23: Linear Velocity
 24: Angular Velocity
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset


# Number of frames per video clip in Cosmos DV 8×8×8 tokenizer
FRAMES_PER_CLIP = 17
# Number of factorized tokens per spatial position
NUM_FACTORED_TOKENS = 3
# Spatial size of tokens
SPATIAL_SIZE = 32
# Per-factor vocabulary size (total vocab = 512^3 or similar)
FACTORED_VOCAB_SIZE = 512


def factorize_to_single_token(factored_tokens: np.ndarray) -> np.ndarray:
    """Convert factored tokens [3, H, W] to single token [H, W].
    
    Uses vocab_size = FACTORED_VOCAB_SIZE^3 mapping.
    """
    # factored_tokens: [3, H, W]
    return (factored_tokens[0] * FACTORED_VOCAB_SIZE * FACTORED_VOCAB_SIZE + 
            factored_tokens[1] * FACTORED_VOCAB_SIZE + 
            factored_tokens[2])


class HumanoidWorldModelDataset(Dataset):
    """Dataset for loading video tokens and states (v2.0 format).
    
    v2.0 format:
    - Video clips: Cosmos DV 8×8×8 tokens, [num_clips, 3, 32, 32]
      - Each clip token [3, 32, 32] decodes to 17 frames
      - Clips are independent segments (boundaries align with segment boundaries)
    - States: [num_frames, 25]
    - Segment indices: [num_frames] - identifies independent video segments
    
    The dataset maps temporal indices to clip indices and handles
    the 17:1 temporal compression ratio.
    
    Note: When decoding, pass all 3 factors directly to Cosmos decoder as [B, 3, H, W].
    Do NOT combine factors into a single token.
    """
    
    def __init__(
        self,
        data_dir: str,
        num_past_frames: int = 9,
        num_future_frames: int = 8,
        stride: int = 1,
        filter_interrupts: bool = True,
        filter_overlaps: bool = False,
        max_shards: Optional[int] = None,
        use_factored_tokens: bool = True,
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing v2.0 sharded data
            num_past_frames: Number of past frames (default: 9)
            num_future_frames: Number of future frames (default: 8)
            stride: Frame skip (default: 1)
            filter_interrupts: Filter sequences that span multiple segments
            filter_overlaps: Filter overlapping sequences
            max_shards: Maximum number of shards to load (for testing)
            use_factored_tokens: If True, return [3, H, W] factored tokens.
                                 If False, return [H, W] single tokens (combined).
        """
        self.data_dir = Path(data_dir)
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.stride = stride
        self.filter_interrupts = filter_interrupts
        self.filter_overlaps = filter_overlaps
        self.use_factored_tokens = use_factored_tokens
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {data_dir}")
            
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        self.num_shards = self.metadata.get("num_shards", 1)
        self.hz = self.metadata.get("hz", 30)
        self.total_frames_all = self.metadata.get("num_images", 0)
        
        if max_shards is not None:
            self.num_shards = min(self.num_shards, max_shards)
        
        # Load per-shard data
        self.shard_data = []
        self.valid_start_inds = []
        
        total_frames = 0
        total_frames_in_sample = self.num_past_frames + self.num_future_frames
        
        for shard_idx in range(self.num_shards):
            shard_info = self._load_shard(shard_idx, total_frames)
            if shard_info is None:
                continue
            
            self.shard_data.append(shard_info)
            shard_data_idx = len(self.shard_data) - 1
            
            # For v2.0, we work at clip level since video is temporally compressed
            # Each clip corresponds to 17 frames, but we sample at clip boundaries
            num_clips = shard_info["num_clips"]
            num_frames = shard_info["num_frames"]
            
            # We need enough clips for past + future at the clip level
            # Since each clip = 17 frames, we need ceil((total_frames_in_sample) / 17) clips
            clips_needed = math.ceil(total_frames_in_sample / FRAMES_PER_CLIP)
            
            for start_clip in range(num_clips - clips_needed):
                # Check segment boundaries if filtering
                if self.filter_interrupts and shard_info["segment_ids"] is not None:
                    start_frame = start_clip * FRAMES_PER_CLIP
                    end_frame = min((start_clip + clips_needed) * FRAMES_PER_CLIP, num_frames - 1)
                    if shard_info["segment_ids"][start_frame] != shard_info["segment_ids"][end_frame]:
                        continue
                
                self.valid_start_inds.append({
                    "shard_data_idx": shard_data_idx,
                    "start_clip": start_clip,
                    "global_start": total_frames + start_clip * FRAMES_PER_CLIP,
                })
            
            total_frames += num_frames
        
        print(f"Dataset: {len(self.valid_start_inds)} valid sequences from {total_frames} frames across {len(self.shard_data)} shards")
        print(f"Action dimension: 25 (matches paper's R^25)")
        print(f"Using {'factored' if use_factored_tokens else 'combined'} token format")
    
    def _load_shard(self, shard_idx: int, global_start: int) -> Optional[Dict]:
        """Load a single shard."""
        # Try sharded format first (train_v2.0: metadata/metadata_0.json, videos/video_0.bin)
        shard_metadata_path = self.data_dir / "metadata" / f"metadata_{shard_idx}.json"
        video_path = self.data_dir / "videos" / f"video_{shard_idx}.bin"
        states_path = self.data_dir / "robot_states" / f"states_{shard_idx}.bin"
        segment_path = self.data_dir / "segment_indices" / f"segment_idx_{shard_idx}.bin"
        
        # If not found, try flat format (val_v2.0: metadata_0.json, video_0.bin in root)
        if not shard_metadata_path.exists():
            shard_metadata_path = self.data_dir / f"metadata_{shard_idx}.json"
            video_path = self.data_dir / f"video_{shard_idx}.bin"
            states_path = self.data_dir / f"states_{shard_idx}.bin"
            segment_path = self.data_dir / f"segment_idx_{shard_idx}.bin"
        
        if not shard_metadata_path.exists():
            return None
            
        with open(shard_metadata_path) as f:
            shard_metadata = json.load(f)
        
        num_frames = shard_metadata.get("shard_num_frames", 0)
        if num_frames == 0:
            return None
        
        num_clips = math.ceil(num_frames / FRAMES_PER_CLIP)
        
        if not video_path.exists() or not states_path.exists():
            return None
        
        # Load video tokens: shape [num_clips, 3, 32, 32]
        video_tokens = np.memmap(
            video_path,
            dtype=np.int32,
            mode="r",
            shape=(num_clips, NUM_FACTORED_TOKENS, SPATIAL_SIZE, SPATIAL_SIZE)
        )
        
        # Load states: shape [num_frames, 25]
        states = np.memmap(
            states_path,
            dtype=np.float32,
            mode="r",
            shape=(num_frames, 25)
        )
        
        # Load segment indices if available
        segment_ids = None
        if segment_path.exists():
            segment_ids = np.memmap(
                segment_path,
                dtype=np.int32,
                mode="r",
                shape=(num_frames,)
            )
        
        return {
            "shard_idx": shard_idx,
            "num_frames": num_frames,
            "num_clips": num_clips,
            "start_frame": global_start,
            "video_tokens": video_tokens,
            "states": states,
            "segment_ids": segment_ids,
        }
    
    def __len__(self) -> int:
        return len(self.valid_start_inds)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        For v2.0 format, each clip represents 17 temporally-compressed frames.
        We return:
        - video_past: (num_past_clips, [3], 32, 32) - past clip tokens
        - video_future: (num_future_clips, [3], 32, 32) - future clip tokens
        - actions_past: (T_p, 25) - past states sampled at frame level
        - actions_future: (T_f, 25) - future states sampled at frame level
        """
        idx_info = self.valid_start_inds[idx]
        shard_info = self.shard_data[idx_info["shard_data_idx"]]
        
        start_clip = idx_info["start_clip"]
        total_frames = self.num_past_frames + self.num_future_frames
        total_clips = math.ceil(total_frames / FRAMES_PER_CLIP)
        
        # Get clip indices
        past_clips = math.ceil(self.num_past_frames / FRAMES_PER_CLIP)
        future_clips = math.ceil(self.num_future_frames / FRAMES_PER_CLIP)
        
        # Extract video clip tokens
        clip_inds = list(range(start_clip, start_clip + past_clips + future_clips))
        video_clips = shard_info["video_tokens"][clip_inds]  # (num_clips, 3, 32, 32)
        
        video_past = video_clips[:past_clips]  # Past clips
        video_future = video_clips[past_clips:past_clips + future_clips]  # Future clips
        
        # Extract states at frame level
        start_frame = start_clip * FRAMES_PER_CLIP
        frame_inds = [start_frame + i * self.stride for i in range(total_frames)]
        # Clamp to valid range
        frame_inds = [min(i, shard_info["num_frames"] - 1) for i in frame_inds]
        
        states = shard_info["states"][frame_inds]  # (total_frames, 25)
        actions_past = states[:self.num_past_frames]  # (T_p, 25)
        actions_future = states[self.num_past_frames:]  # (T_f, 25)
        
        # Convert to tensors
        if self.use_factored_tokens:
            # Return factored format: (num_clips, 3, 32, 32)
            video_past_tensor = torch.from_numpy(video_past.copy()).long()
            video_future_tensor = torch.from_numpy(video_future.copy()).long()
        else:
            # Combine factored tokens: (num_clips, 32, 32)
            video_past_combined = np.array([factorize_to_single_token(v) for v in video_past])
            video_future_combined = np.array([factorize_to_single_token(v) for v in video_future])
            video_past_tensor = torch.from_numpy(video_past_combined).long()
            video_future_tensor = torch.from_numpy(video_future_combined).long()
        
        return {
            "video_past": video_past_tensor,
            "video_future": video_future_tensor,
            "actions_past": torch.from_numpy(actions_past.copy()).float(),
            "actions_future": torch.from_numpy(actions_future.copy()).float(),
        }
