"""FVD (Frechet Video Distance) metric for video generation evaluation

Wrapper around the frechet_video_distance implementation for easier use with Genie models.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
import sys

# Add the frechet_video_distance directory to path
FVD_DIR = Path(__file__).parent / "frechet_video_distance-pytorch-"
if str(FVD_DIR) not in sys.path:
    sys.path.insert(0, str(FVD_DIR))

try:
    from frechet_video_distance import (
        preprocess,
        get_activations,
        calculate_fvd_from_activations,
        batch_generator,
    )
    from pytorch_i3d_model.pytorch_i3d import InceptionI3d
except ImportError:
    print("Warning: Could not import FVD dependencies. FVD metric will not be available.")
    print(f"Please ensure the I3D model is available in {FVD_DIR}")
    InceptionI3d = None


class FVDMetric:
    """Calculate FVD (Frechet Video Distance) for video generation"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 10,
    ):
        """
        Args:
            model_path: Path to I3D model weights (if None, will try to download)
            device: Device to run on
            batch_size: Batch size for processing videos
        """
        self.device = device
        self.batch_size = batch_size
        
        if InceptionI3d is None:
            raise ImportError("FVD dependencies not available. Please install required packages.")
        
        # Load I3D model
        if model_path is None:
            # Try to find model in the directory (check both locations)
            model_path = FVD_DIR / "pytorch_i3d_model" / "rgb_imagenet.pt"
            if not model_path.exists():
                # Try models subdirectory
                model_path = FVD_DIR / "pytorch_i3d_model" / "models" / "rgb_imagenet.pt"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"I3D model not found. Tried:\n"
                    f"  - {FVD_DIR / 'pytorch_i3d_model' / 'rgb_imagenet.pt'}\n"
                    f"  - {FVD_DIR / 'pytorch_i3d_model' / 'models' / 'rgb_imagenet.pt'}\n"
                    "Please download the model weights or specify model_path."
                )
        
        self.i3d = InceptionI3d(400, in_channels=3)
        self.i3d.load_state_dict(torch.load(model_path, map_location=device))
        self.i3d.to(device)
        self.i3d.eval()
    
    @torch.no_grad()
    def compute(
        self,
        real_videos: torch.Tensor,
        generated_videos: torch.Tensor,
    ) -> float:
        """
        Compute FVD between real and generated videos.
        
        Args:
            real_videos: Real videos of shape (N, T, H, W, C) or (N, T, C, H, W)
            generated_videos: Generated videos of same shape as real_videos
        
        Returns:
            FVD value (lower is better)
        """
        # Convert to (N, T, H, W, C) format if needed
        if real_videos.shape[-1] != 3:
            # Assume (N, T, C, H, W) format
            real_videos = real_videos.permute(0, 1, 3, 4, 2)
        if generated_videos.shape[-1] != 3:
            generated_videos = generated_videos.permute(0, 1, 3, 4, 2)
        
        # Ensure videos are in [0, 255] range
        if real_videos.max() <= 1.0:
            real_videos = real_videos * 255.0
        if generated_videos.max() <= 1.0:
            generated_videos = generated_videos * 255.0
        
        # Preprocess videos
        real_preprocessed = preprocess(real_videos, (224, 224)).to(self.device)
        gen_preprocessed = preprocess(generated_videos, (224, 224)).to(self.device)
        
        # Get activations
        print("Calculating activations for real videos...")
        real_activations = self._get_activations(real_preprocessed)
        
        print("Calculating activations for generated videos...")
        gen_activations = self._get_activations(gen_preprocessed)
        
        # Calculate FVD
        fvd = calculate_fvd_from_activations(real_activations, gen_activations)
        
        return float(fvd)
    
    @torch.no_grad()
    def _get_activations(self, videos: torch.Tensor) -> np.ndarray:
        """Get I3D activations for videos"""
        activations = []
        for batch in batch_generator(videos, self.batch_size):
            batch = batch.to(self.device)
            act = self.i3d(batch).cpu().numpy()  # (batch, time, classes) or (batch, classes)
            # Ensure activations are 2D: (batch, features)
            # I3D returns (batch, time, classes) - we need to flatten time and classes
            if act.ndim == 3:
                # (batch, time, classes) -> (batch, time*classes)
                act = act.reshape(act.shape[0], -1)
            elif act.ndim == 2:
                # Already (batch, features) - good
                pass
            elif act.ndim == 1:
                # Single sample - add batch dimension
                act = act[np.newaxis, :]
            else:
                # Flatten all dimensions except batch
                act = act.reshape(act.shape[0], -1)
            activations.append(act)
        result = np.vstack(activations)
        # Final check: ensure result is 2D (samples, features)
        if result.ndim > 2:
            result = result.reshape(result.shape[0], -1)
        return result


def batch_generator(data: torch.Tensor, batch_size: int):
    """Generate batches from data"""
    n = data.size()[0]
    indices = np.random.permutation(n)
    
    for i in range(0, n, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield data[batch_indices]
