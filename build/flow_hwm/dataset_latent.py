"""Dataset adapter for continuous video latents in Flow-HWM.

Flow-HWM operates in continuous latent space (pre-quantization), unlike
Masked-HWM which uses discrete tokens. This module provides:

1. FlowHWMDataset: Wraps existing HumanoidWorldModelDataset and converts
   discrete tokens to approximate continuous latents via dequantization.

2. FlowHWMLatentDataset: Loads pre-computed continuous latents directly
   (for when raw latents are available).

The continuous latents have shape (B, latent_dim, T, H, W) where:
- latent_dim: Number of latent channels (e.g., 16 for Cosmos)
- T: Number of temporal tokens (clips * tokens per clip)
- H, W: Spatial dimensions (16x16 for Cosmos CV 8×16×16)
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Import the existing dataset
import sys
sys.path.insert(0, "/media/skr/storage/robot_world/humanoid_wm/build")
from data.dataset import HumanoidWorldModelDataset, FRAMES_PER_CLIP, SPATIAL_SIZE


def infer_cv_temporal_tokens_per_clip(
    cv_tokenizer_dir: str,
    dv_tokenizer_dir: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> int:
    """Infer temporal tokens per clip from the CV encoder."""
    try:
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    except ImportError as exc:
        raise ImportError(
            "cosmos_tokenizer package not found. "
            "Install it before using automatic temporal token detection."
        ) from exc

    enc_path = Path(cv_tokenizer_dir) / "encoder.jit"
    dec_path = Path(dv_tokenizer_dir) / "decoder.jit"
    if not enc_path.exists():
        raise FileNotFoundError(f"CV encoder.jit not found at {enc_path}")
    if not dec_path.exists():
        raise FileNotFoundError(f"DV decoder.jit not found at {dec_path}")

    cv_encoder = CausalVideoTokenizer(
        checkpoint_enc=str(enc_path),
        device=device,
        dtype=dtype,
    )
    dv_decoder = CausalVideoTokenizer(
        checkpoint_dec=str(dec_path),
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        # DV tokens: (B=1, 3, 32, 32) -> decoded to (1, 3, 17, 256, 256)
        tokens = torch.randint(
            low=0,
            high=65536,
            size=(1, 3, SPATIAL_SIZE, SPATIAL_SIZE),
            device=device,
            dtype=torch.long,
        )
        decoded = dv_decoder.decode(tokens)
        latents = cv_encoder.encode(decoded)
        if isinstance(latents, tuple):
            latents = latents[0]

    if latents.ndim != 5:
        raise ValueError(f"Unexpected latent shape from CV encoder: {latents.shape}")

    return int(latents.shape[2])


class FlowHWMDataset(Dataset):
    """Dataset wrapper that provides continuous latents for Flow-HWM.

    Wraps HumanoidWorldModelDataset and converts discrete tokens to
    continuous latents using Cosmos CV 8×16×16 (or a fallback approximation).

    For training, we use the discrete tokens as indices into a learned
    embedding table, producing pseudo-continuous representations. This
    is a practical approximation when true continuous latents are not
    available.

    Args:
        data_dir: Directory containing v2.0 sharded data
        num_past_clips: Number of past video clips for context
        num_future_clips: Number of future clips to predict
        latent_dim: Dimension of continuous latents (default: 16)
        embed_dim: If using embedding lookup, dimension per factor
        use_cv_tokenizer: If True, decode DV tokens to RGB and re-encode with
            Cosmos CV 8×16×16 encoder (continuous latents).
        cv_tokenizer_dir: Directory with CV 8×16×16 encoder checkpoints.
        dv_tokenizer_dir: Directory with DV 8×8×8 decoder checkpoints.
        cv_temporal_tokens_per_clip: Temporal tokens per 17-frame clip (default: 3)
        cv_device: Device for tokenizer inference (e.g., "cuda")
        cv_dtype: Torch dtype for tokenizer inference (e.g., "bfloat16")
        use_embedding: If True, use embedding lookup for continuous latents
                      If False, use normalized token indices as latents
        **kwargs: Additional arguments passed to HumanoidWorldModelDataset
    """

    def __init__(
        self,
        data_dir: str,
        num_past_clips: int = 2,
        num_future_clips: int = 1,
        latent_dim: int = 16,
        use_cv_tokenizer: bool = True,
        cv_tokenizer_dir: str = "/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer/Continuous_video",
        dv_tokenizer_dir: str = "/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer",
        cv_temporal_tokens_per_clip: Optional[int] = 3,
        cv_device: str = "cuda",
        cv_dtype: str = "bfloat16",
        use_embedding: bool = False,
        **kwargs,
    ):
        self.num_past_clips = num_past_clips
        self.num_future_clips = num_future_clips
        self.latent_dim = latent_dim
        self.use_cv_tokenizer = use_cv_tokenizer
        self.cv_tokenizer_dir = Path(cv_tokenizer_dir)
        self.dv_tokenizer_dir = Path(dv_tokenizer_dir)
        self.cv_temporal_tokens_per_clip = cv_temporal_tokens_per_clip
        self.cv_device = cv_device
        self.cv_dtype = cv_dtype
        self.use_embedding = use_embedding
        self._cv_encoder = None
        self._dv_decoder = None

        # Create underlying discrete token dataset
        self.base_dataset = HumanoidWorldModelDataset(
            data_dir=data_dir,
            num_past_clips=num_past_clips,
            num_future_clips=num_future_clips,
            use_factored_tokens=True,  # We need factored format
            **kwargs,
        )

        if self.use_cv_tokenizer and self.use_embedding:
            raise ValueError("use_embedding is incompatible with use_cv_tokenizer.")

        # If using embedding, create embedding tables
        # 3 factors, each with vocab size ~65536
        # Handle case where latent_dim is not divisible by 3
        if use_embedding:
            embed_dim_per_factor = latent_dim // 3
            self.embeddings = torch.nn.ModuleList([
                torch.nn.Embedding(65536, embed_dim_per_factor)
                for _ in range(3)
            ])
            # Initialize with small random values
            for emb in self.embeddings:
                torch.nn.init.normal_(emb.weight, std=0.02)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _ensure_cv_tokenizers(self) -> None:
        if self._cv_encoder is not None and self._dv_decoder is not None:
            return

        enc_path = self.cv_tokenizer_dir / "encoder.jit"
        dec_path = self.dv_tokenizer_dir / "decoder.jit"
        if not enc_path.exists():
            raise FileNotFoundError(f"CV encoder.jit not found at {enc_path}")
        if not dec_path.exists():
            raise FileNotFoundError(f"DV decoder.jit not found at {dec_path}")

        try:
            from cosmos_tokenizer.video_lib import CausalVideoTokenizer
        except ImportError as exc:
            raise ImportError(
                "cosmos_tokenizer package not found. "
                "Install it before using use_cv_tokenizer."
            ) from exc

        self._cv_encoder = CausalVideoTokenizer(
            checkpoint_enc=str(enc_path),
            device=self.cv_device,
            dtype=self.cv_dtype,
        )
        self._dv_decoder = CausalVideoTokenizer(
            checkpoint_dec=str(dec_path),
            device=self.cv_device,
            dtype=self.cv_dtype,
        )

    def _tokens_to_cv_latents(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert DV tokens to continuous CV latents via decode+encode."""
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError(
                "use_cv_tokenizer requires num_workers=0 to avoid CUDA in workers."
            )

        self._ensure_cv_tokenizers()

        # tokens: (T, 3, H, W) -> decoded: (T, 3, 17, 256, 256)
        tokens = tokens.to(device=self.cv_device)
        decoded = self._dv_decoder.decode(tokens)

        # Encode each clip independently (batch = num_clips)
        latents = self._cv_encoder.encode(decoded)
        if isinstance(latents, tuple):
            latents = latents[0]

        if latents.ndim != 5:
            raise ValueError(f"Unexpected latent shape from CV encoder: {latents.shape}")
        if latents.shape[1] != self.latent_dim:
            raise ValueError(
                f"CV encoder channels {latents.shape[1]} != latent_dim {self.latent_dim}"
            )
        if self.cv_temporal_tokens_per_clip is None:
            self.cv_temporal_tokens_per_clip = latents.shape[2]
        if latents.shape[2] != self.cv_temporal_tokens_per_clip:
            raise ValueError(
                "CV temporal tokens per clip mismatch: "
                f"expected {self.cv_temporal_tokens_per_clip}, got {latents.shape[2]}"
            )

        # Merge clip and temporal token dimensions: (T_clips, C, t, H, W) -> (C, T_clips*t, H, W)
        latents = latents.permute(1, 0, 2, 3, 4).contiguous()
        C, T_clips, t, H, W = latents.shape
        latents = latents.view(C, T_clips * t, H, W)
        return latents

    def _tokens_to_latents(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert discrete tokens to continuous latents.

        Args:
            tokens: Discrete tokens (T, 3, H, W) - int64

        Returns:
            latents: Continuous latents (latent_dim, T, H, W) - float32
        """
        T, num_factors, H, W = tokens.shape

        if self.use_embedding:
            # Use embedding lookup
            # tokens: (T, 3, H, W) -> embeddings: (T, 3, H, W, latent_dim//3)
            # Handle case where latent_dim is not divisible by 3
            embed_dim_per_factor = self.latent_dim // 3
            remainder = self.latent_dim % 3
            
            embeds = []
            for i in range(num_factors):
                factor_tokens = tokens[:, i]  # (T, H, W)
                factor_embed = self.embeddings[i](factor_tokens)  # (T, H, W, embed_dim_per_factor)
                embeds.append(factor_embed)

            # Concatenate along last dimension: (T, H, W, embed_dim_per_factor*3)
            latents = torch.cat(embeds, dim=-1)  # (T, H, W, embed_dim_per_factor*3)
            
            # If there's a remainder, pad with zeros or repeat first factor
            if remainder > 0:
                # Pad with zeros to reach latent_dim
                padding_shape = list(latents.shape)
                padding_shape[-1] = remainder
                padding = torch.zeros(padding_shape, dtype=latents.dtype, device=latents.device)
                latents = torch.cat([latents, padding], dim=-1)  # (T, H, W, latent_dim)

            # Rearrange to (latent_dim, T, H, W)
            latents = latents.permute(3, 0, 1, 2)
        else:
            # Simple approach: normalize token indices to [-1, 1] range
            # This provides a basic continuous representation
            # tokens: (T, 3, H, W) int64 in [0, 65535]

            # Normalize to [0, 1]
            latents = tokens.float() / 65535.0

            # Scale to [-1, 1] for better distribution
            latents = latents * 2 - 1

            # Expand to full latent_dim by repeating/interleaving
            # (T, 3, H, W) -> (T, latent_dim, H, W) -> (latent_dim, T, H, W)
            if self.latent_dim == 3:
                # Direct mapping
                latents = latents.permute(1, 0, 2, 3)  # (3, T, H, W)
            else:
                # Repeat each factor to fill latent_dim
                # Handle case where latent_dim is not divisible by 3
                repeats = self.latent_dim // 3
                remainder = self.latent_dim % 3
                
                # Repeat each of the 3 factors
                expanded = latents.repeat_interleave(repeats, dim=1)  # (T, repeats*3, H, W)
                
                # If there's a remainder, pad with the first factor(s)
                if remainder > 0:
                    padding = latents[:, :remainder].repeat_interleave(1, dim=1)  # (T, remainder, H, W)
                    expanded = torch.cat([expanded, padding], dim=1)  # (T, latent_dim, H, W)
                
                latents = expanded.permute(1, 0, 2, 3)  # (latent_dim, T, H, W)

        return latents.contiguous()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with continuous latents.

        Returns:
            latent_past: (latent_dim, T_p, H, W) past continuous latents
            latent_future: (latent_dim, T_f, H, W) future continuous latents (target X_1)
            actions_past: (T_p * 17, action_dim) past actions at frame level
            actions_future: (T_f * 17, action_dim) future actions at frame level
        """
        # Get discrete token sample from base dataset
        sample = self.base_dataset[idx]

        # Convert tokens to continuous latents
        # video_past: (T_p, 3, H, W) -> latent_past: (latent_dim, T_p, H, W)
        if self.use_cv_tokenizer:
            latent_past = self._tokens_to_cv_latents(sample["video_past"])
            latent_future = self._tokens_to_cv_latents(sample["video_future"])
        else:
            latent_past = self._tokens_to_latents(sample["video_past"])
            latent_future = self._tokens_to_latents(sample["video_future"])

        return {
            "latent_past": latent_past,
            "latent_future": latent_future,
            "actions_past": sample["actions_past"],
            "actions_future": sample["actions_future"],
        }


class FlowHWMLatentDataset(Dataset):
    """Dataset for loading pre-computed continuous latents.

    Use this when you have access to the actual continuous latents from
    the Cosmos encoder (before quantization). This provides the most
    accurate representation for flow matching.

    Expected data format:
    - latents_{shard_idx}.bin: float32 array of shape
      [num_clips * temporal_tokens_per_clip, latent_dim, H, W]
    - states_{shard_idx}.bin: float32 array of shape [num_frames, 25]
    - segment_idx_{shard_idx}.bin: int32 array of shape [num_frames]
    - metadata.json: Contains num_shards, latent_dim, etc.

    Args:
        data_dir: Directory containing pre-computed latent data
        num_past_clips: Number of past video clips
        num_future_clips: Number of future clips to predict
        latent_dim: Dimension of continuous latents
        temporal_tokens_per_clip: Temporal tokens per clip
        filter_interrupts: Filter sequences spanning multiple segments
        max_shards: Maximum shards to load (for testing)
    """

    def __init__(
        self,
        data_dir: str,
        num_past_clips: int = 2,
        num_future_clips: int = 1,
        latent_dim: int = 16,
        spatial_size: int = 16,
        temporal_tokens_per_clip: int = 3,
        filter_interrupts: bool = True,
        max_shards: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.num_past_clips = num_past_clips
        self.num_future_clips = num_future_clips
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        self.temporal_tokens_per_clip = temporal_tokens_per_clip
        self.filter_interrupts = filter_interrupts

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {data_dir}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.num_shards = self.metadata.get("num_shards", 1)
        if max_shards is not None:
            self.num_shards = min(self.num_shards, max_shards)

        # Load shards
        self.shard_data = []
        self.valid_start_inds = []
        clips_needed = num_past_clips + num_future_clips

        for shard_idx in range(self.num_shards):
            shard_info = self._load_shard(shard_idx)
            if shard_info is None:
                continue

            self.shard_data.append(shard_info)
            shard_data_idx = len(self.shard_data) - 1

            num_clips = shard_info["num_clips"]
            num_frames = shard_info["num_frames"]

            for start_clip in range(num_clips - clips_needed + 1):
                if self.filter_interrupts and shard_info["segment_ids"] is not None:
                    start_frame = start_clip * FRAMES_PER_CLIP
                    end_frame = min((start_clip + clips_needed) * FRAMES_PER_CLIP - 1, num_frames - 1)
                    if shard_info["segment_ids"][start_frame] != shard_info["segment_ids"][end_frame]:
                        continue

                self.valid_start_inds.append({
                    "shard_data_idx": shard_data_idx,
                    "start_clip": start_clip,
                })

        print(f"FlowHWMLatentDataset: {len(self.valid_start_inds)} valid sequences")

    def _load_shard(self, shard_idx: int) -> Optional[Dict]:
        """Load a single shard of pre-computed latents."""
        latent_path = self.data_dir / f"latents_{shard_idx}.bin"
        states_path = self.data_dir / f"states_{shard_idx}.bin"
        segment_path = self.data_dir / f"segment_idx_{shard_idx}.bin"
        metadata_path = self.data_dir / f"metadata_{shard_idx}.json"

        if not latent_path.exists() or not states_path.exists():
            return None

        # Load shard metadata
        if metadata_path.exists():
            with open(metadata_path) as f:
                shard_metadata = json.load(f)
            num_clips = shard_metadata.get("num_clips", 0)
            num_frames = shard_metadata.get("num_frames", num_clips * FRAMES_PER_CLIP)
        else:
            # Infer from file size
            file_size = os.path.getsize(latent_path)
            elements_per_clip = (
                self.latent_dim
                * self.spatial_size
                * self.spatial_size
                * self.temporal_tokens_per_clip
            )
            num_clips = file_size // (4 * elements_per_clip)  # 4 bytes per float32
            num_frames = num_clips * FRAMES_PER_CLIP

        if num_clips == 0:
            return None

        # Load latents: shape [num_clips * temporal_tokens_per_clip, latent_dim, H, W]
        latents = np.memmap(
            latent_path,
            dtype=np.float32,
            mode="r",
            shape=(
                num_clips * self.temporal_tokens_per_clip,
                self.latent_dim,
                self.spatial_size,
                self.spatial_size,
            ),
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
            "num_clips": num_clips,
            "num_frames": num_frames,
            "latents": latents,
            "states": states,
            "segment_ids": segment_ids,
        }

    def __len__(self) -> int:
        return len(self.valid_start_inds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with continuous latents.

        Returns:
            latent_past: (latent_dim, T_p, H, W) past continuous latents
            latent_future: (latent_dim, T_f, H, W) future continuous latents
            actions_past: (T_p * 17, action_dim) past actions
            actions_future: (T_f * 17, action_dim) future actions
        """
        idx_info = self.valid_start_inds[idx]
        shard_info = self.shard_data[idx_info["shard_data_idx"]]
        start_clip = idx_info["start_clip"]

        total_clips = self.num_past_clips + self.num_future_clips

        # Extract latents: (total_clips * temporal_tokens_per_clip, latent_dim, H, W)
        clip_inds = list(range(start_clip, start_clip + total_clips))
        token_inds = []
        for clip_idx in clip_inds:
            start = clip_idx * self.temporal_tokens_per_clip
            end = start + self.temporal_tokens_per_clip
            token_inds.extend(range(start, end))
        latents = shard_info["latents"][token_inds]

        # Reshape to (total_clips, temporal_tokens_per_clip, latent_dim, H, W)
        latents = latents.reshape(
            total_clips,
            self.temporal_tokens_per_clip,
            self.latent_dim,
            self.spatial_size,
            self.spatial_size,
        )

        latent_past = latents[:self.num_past_clips]  # (T_p, t, C, H, W)
        latent_future = latents[self.num_past_clips:]  # (T_f, t, C, H, W)

        # Reshape to (latent_dim, T, H, W)
        latent_past = np.transpose(latent_past, (2, 0, 1, 3, 4))
        latent_future = np.transpose(latent_future, (2, 0, 1, 3, 4))
        latent_past = latent_past.reshape(
            self.latent_dim,
            self.num_past_clips * self.temporal_tokens_per_clip,
            self.spatial_size,
            self.spatial_size,
        )
        latent_future = latent_future.reshape(
            self.latent_dim,
            self.num_future_clips * self.temporal_tokens_per_clip,
            self.spatial_size,
            self.spatial_size,
        )

        # Extract states at frame level
        start_frame = start_clip * FRAMES_PER_CLIP
        past_frames = self.num_past_clips * FRAMES_PER_CLIP
        future_frames = self.num_future_clips * FRAMES_PER_CLIP
        total_frames = past_frames + future_frames

        frame_inds = list(range(start_frame, start_frame + total_frames))
        frame_inds = [min(i, shard_info["num_frames"] - 1) for i in frame_inds]

        states = shard_info["states"][frame_inds]
        actions_past = states[:past_frames]
        actions_future = states[past_frames:past_frames + future_frames]

        return {
            "latent_past": torch.from_numpy(latent_past.copy()).float(),
            "latent_future": torch.from_numpy(latent_future.copy()).float(),
            "actions_past": torch.from_numpy(actions_past.copy()).float(),
            "actions_future": torch.from_numpy(actions_future.copy()).float(),
        }


def create_flow_hwm_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_past_clips: int = 2,
    num_future_clips: int = 1,
    latent_dim: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    use_precomputed_latents: bool = False,
    use_cv_tokenizer: bool = True,
    cv_temporal_tokens_per_clip: int = 3,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for Flow-HWM training.

    Args:
        data_dir: Data directory
        batch_size: Batch size
        num_past_clips: Number of past clips
        num_future_clips: Number of future clips
        latent_dim: Latent dimension
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
        use_precomputed_latents: If True, use FlowHWMLatentDataset
        cv_temporal_tokens_per_clip: Temporal tokens per clip for CV encoder
        **kwargs: Additional dataset arguments

    Returns:
        DataLoader for Flow-HWM
    """
    if use_precomputed_latents:
        dataset = FlowHWMLatentDataset(
            data_dir=data_dir,
            num_past_clips=num_past_clips,
            num_future_clips=num_future_clips,
            latent_dim=latent_dim,
            temporal_tokens_per_clip=cv_temporal_tokens_per_clip,
            **kwargs,
        )
    else:
        dataset = FlowHWMDataset(
            data_dir=data_dir,
            num_past_clips=num_past_clips,
            num_future_clips=num_future_clips,
            latent_dim=latent_dim,
            use_cv_tokenizer=use_cv_tokenizer,
            cv_temporal_tokens_per_clip=cv_temporal_tokens_per_clip,
            **kwargs,
        )

    if use_cv_tokenizer and num_workers != 0 and not use_precomputed_latents:
        num_workers = 0

    pin_memory = not use_cv_tokenizer

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
