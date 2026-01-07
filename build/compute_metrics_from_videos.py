#!/usr/bin/env python3
"""Compute FVD and PSNR metrics from comparison videos.

This script loads side-by-side comparison videos (GT | Predicted) and computes
FVD (Fréchet Video Distance) and PSNR (Peak Signal-to-Noise Ratio) metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("ERROR: imageio not available. Install with: pip install imageio imageio-ffmpeg")
    sys.exit(1)

# Add Metrics directory to path
sys.path.insert(0, str(Path(__file__).parent / "Metrics" / "FVD"))
sys.path.insert(0, str(Path(__file__).parent / "Metrics" / "PSNR"))

try:
    from fvd_metric import FVDMetric
except ImportError:
    print("WARNING: Could not import FVDMetric. FVD computation will be skipped.")
    FVDMetric = None

try:
    from psnr_metric import psnr
except ImportError:
    print("WARNING: Could not import psnr. PSNR computation will be skipped.")
    psnr = None


def load_video(video_path: Path) -> np.ndarray:
    """Load video from file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Video array of shape (T, H, W, 3) in uint8 format
    """
    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()
    
    video = np.stack(frames, axis=0)  # (T, H, W, 3)
    return video


def split_comparison_video(
    comparison_video: np.ndarray,
    remove_label_bars: bool = True,
    label_bar_height: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Split side-by-side comparison video into GT and predicted halves.
    
    Args:
        comparison_video: (T, H, W, 3) side-by-side video (GT | Predicted)
        remove_label_bars: Whether to remove label bars at the top (if present)
        label_bar_height: Height of label bars to remove (if present)
        
    Returns:
        gt_video: (T, H', W//2, 3) ground truth video (left half)
        pred_video: (T, H', W//2, 3) predicted video (right half)
    """
    T, H, W, C = comparison_video.shape
    
    # Split horizontally: left half = GT, right half = Predicted
    mid_point = W // 2
    gt_video = comparison_video[:, :, :mid_point, :]  # Left half
    pred_video = comparison_video[:, :, mid_point:, :]  # Right half
    
    # Remove label bars if present (typically at the top)
    if remove_label_bars and H > label_bar_height:
        # Check if there's a label bar by looking at the first frame
        # Label bars are typically black or have text, but we'll just remove
        # the top portion if it's significantly different from the rest
        # For simplicity, we'll remove the top label_bar_height pixels
        gt_video = gt_video[:, label_bar_height:, :, :]
        pred_video = pred_video[:, label_bar_height:, :, :]
    
    return gt_video, pred_video


def compute_psnr_metric(gt_video: np.ndarray, pred_video: np.ndarray) -> Dict[str, float]:
    """Compute PSNR metric between GT and predicted videos.
    
    Args:
        gt_video: (T, H, W, 3) ground truth video
        pred_video: (T, H, W, 3) predicted video
        
    Returns:
        Dictionary with PSNR metrics
    """
    if psnr is None:
        return {"error": "PSNR metric not available"}
    
    # Convert to torch tensors
    # Convert from (T, H, W, 3) to (1, T, 3, H, W) for batch processing
    gt_tensor = torch.from_numpy(gt_video).float().permute(0, 3, 1, 2)  # (T, 3, H, W)
    pred_tensor = torch.from_numpy(pred_video).float().permute(0, 3, 1, 2)  # (T, 3, H, W)
    
    # Normalize to [0, 1] if needed
    if gt_tensor.max() > 1.0:
        gt_tensor = gt_tensor / 255.0
    if pred_tensor.max() > 1.0:
        pred_tensor = pred_tensor / 255.0
    
    # Add batch dimension: (1, T, 3, H, W)
    gt_tensor = gt_tensor.unsqueeze(0)
    pred_tensor = pred_tensor.unsqueeze(0)
    
    # Compute PSNR per frame
    psnr_per_frame = []
    T = gt_tensor.shape[1]
    for t in range(T):
        gt_frame = gt_tensor[:, t, :, :, :]  # (1, 3, H, W)
        pred_frame = pred_tensor[:, t, :, :, :]  # (1, 3, H, W)
        psnr_val = psnr(gt_frame, pred_frame, max_val=1.0, reduction='none')
        psnr_per_frame.append(psnr_val.item())
    
    # Compute overall PSNR
    overall_psnr = psnr(gt_tensor, pred_tensor, max_val=1.0, reduction='mean').item()
    
    return {
        "psnr_mean": float(overall_psnr),
        "psnr_per_frame": [float(p) for p in psnr_per_frame],
        "psnr_std": float(np.std(psnr_per_frame)),
        "psnr_min": float(np.min(psnr_per_frame)),
        "psnr_max": float(np.max(psnr_per_frame)),
    }


def compute_fvd_metric(
    gt_videos: List[np.ndarray],
    pred_videos: List[np.ndarray],
    device: str = "cuda",
    fvd_model_path: str = None
) -> Dict[str, float]:
    """Compute FVD metric between GT and predicted videos.
    
    Args:
        gt_videos: List of (T, H, W, 3) ground truth videos
        pred_videos: List of (T, H, W, 3) predicted videos
        device: Device to run FVD computation on
        fvd_model_path: Path to I3D model weights (optional)
        
    Returns:
        Dictionary with FVD metric
    """
    if FVDMetric is None:
        return {"error": "FVD metric not available"}
    
    if len(gt_videos) != len(pred_videos):
        return {"error": f"Mismatch in number of videos: {len(gt_videos)} vs {len(pred_videos)}"}
    
    # Convert to torch tensors: (N, T, H, W, 3)
    gt_tensor = torch.from_numpy(np.stack(gt_videos)).float()  # (N, T, H, W, 3)
    pred_tensor = torch.from_numpy(np.stack(pred_videos)).float()  # (N, T, H, W, 3)
    
    # Ensure values are in [0, 255] range
    if gt_tensor.max() <= 1.0:
        gt_tensor = gt_tensor * 255.0
    if pred_tensor.max() <= 1.0:
        pred_tensor = pred_tensor * 255.0
    
    # Initialize FVD metric
    try:
        fvd_metric = FVDMetric(model_path=fvd_model_path, device=device)
    except Exception as e:
        return {"error": f"Failed to initialize FVD metric: {str(e)}"}
    
    # Compute FVD
    try:
        fvd_value = fvd_metric.compute(gt_tensor, pred_tensor)
        return {
            "fvd": float(fvd_value),
            "num_videos": len(gt_videos),
        }
    except Exception as e:
        return {"error": f"Failed to compute FVD: {str(e)}"}


def process_video_directory(
    video_dir: Path,
    output_path: Path = None,
    device: str = "cuda",
    compute_fvd: bool = True,
    compute_psnr: bool = True,
    fvd_model_path: str = None,
    remove_label_bars: bool = True,
    label_bar_height: int = 64
) -> Dict:
    """Process all comparison videos in a directory and compute metrics.
    
    Args:
        video_dir: Directory containing comparison videos
        output_path: Path to save metrics JSON (optional)
        device: Device for FVD computation
        compute_fvd: Whether to compute FVD
        compute_psnr: Whether to compute PSNR
        fvd_model_path: Path to I3D model weights (optional)
        remove_label_bars: Whether to remove label bars from videos
        label_bar_height: Height of label bars to remove (if present)
        
    Returns:
        Dictionary with all metrics
    """
    # Find all comparison videos
    video_files = sorted(video_dir.glob("*comparison*.mp4"))
    
    if not video_files:
        print(f"ERROR: No comparison videos found in {video_dir}")
        return {}
    
    print(f"Found {len(video_files)} comparison videos")
    
    # Process each video
    all_gt_videos = []
    all_pred_videos = []
    per_video_metrics = {}
    
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        
        # Load video
        try:
            comparison_video = load_video(video_path)
            print(f"  Loaded video: shape {comparison_video.shape}")
        except Exception as e:
            print(f"  ERROR: Failed to load video: {e}")
            continue
        
        # Split into GT and predicted
        gt_video, pred_video = split_comparison_video(
            comparison_video,
            remove_label_bars=remove_label_bars,
            label_bar_height=label_bar_height
        )
        print(f"  GT shape: {gt_video.shape}, Predicted shape: {pred_video.shape}")
        
        # Store for batch FVD computation
        all_gt_videos.append(gt_video)
        all_pred_videos.append(pred_video)
        
        # Compute per-video PSNR
        if compute_psnr:
            print("  Computing PSNR...")
            psnr_metrics = compute_psnr_metric(gt_video, pred_video)
            per_video_metrics[video_path.stem] = {
                "psnr": psnr_metrics
            }
            if "error" not in psnr_metrics:
                print(f"  PSNR: {psnr_metrics['psnr_mean']:.2f} dB")
    
    # Compute FVD across all videos
    results = {
        "num_videos": len(all_gt_videos),
        "per_video_metrics": per_video_metrics,
    }
    
    if compute_fvd and len(all_gt_videos) > 0:
        print(f"\nComputing FVD across {len(all_gt_videos)} videos...")
        fvd_metrics = compute_fvd_metric(
            all_gt_videos,
            all_pred_videos,
            device=device,
            fvd_model_path=fvd_model_path
        )
        results["fvd"] = fvd_metrics
        if "error" not in fvd_metrics:
            print(f"FVD: {fvd_metrics['fvd']:.2f}")
    
    # Compute aggregate PSNR statistics
    if compute_psnr and per_video_metrics:
        all_psnr_values = []
        for video_metrics in per_video_metrics.values():
            if "psnr" in video_metrics and "error" not in video_metrics["psnr"]:
                all_psnr_values.append(video_metrics["psnr"]["psnr_mean"])
        
        if all_psnr_values:
            results["psnr_aggregate"] = {
                "mean": float(np.mean(all_psnr_values)),
                "std": float(np.std(all_psnr_values)),
                "min": float(np.min(all_psnr_values)),
                "max": float(np.max(all_psnr_values)),
            }
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute FVD and PSNR metrics from comparison videos"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing comparison videos (GT | Predicted side-by-side)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for metrics (default: video_dir/metrics.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for FVD computation (cuda or cpu)"
    )
    parser.add_argument(
        "--no_fvd",
        action="store_true",
        help="Skip FVD computation"
    )
    parser.add_argument(
        "--no_psnr",
        action="store_true",
        help="Skip PSNR computation"
    )
    parser.add_argument(
        "--fvd_model_path",
        type=str,
        default=None,
        help="Path to I3D model weights (optional, will try to find automatically)"
    )
    parser.add_argument(
        "--keep_label_bars",
        action="store_true",
        help="Keep label bars in videos (by default, label bars are removed)"
    )
    parser.add_argument(
        "--label_bar_height",
        type=int,
        default=64,
        help="Height of label bars to remove (default: 64)"
    )
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"ERROR: Video directory does not exist: {video_dir}")
        return
    
    output_path = Path(args.output) if args.output else video_dir / "metrics.json"
    
    print(f"{'='*60}")
    print(f"Computing Metrics from Comparison Videos")
    print(f"{'='*60}")
    print(f"Video directory: {video_dir}")
    print(f"Output: {output_path}")
    print(f"Device: {args.device}")
    print(f"Compute FVD: {not args.no_fvd}")
    print(f"Compute PSNR: {not args.no_psnr}")
    print(f"{'='*60}\n")
    
    results = process_video_directory(
        video_dir=video_dir,
        output_path=output_path,
        device=args.device,
        compute_fvd=not args.no_fvd,
        compute_psnr=not args.no_psnr,
        fvd_model_path=args.fvd_model_path,
        remove_label_bars=not args.keep_label_bars,
        label_bar_height=args.label_bar_height
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    if "psnr_aggregate" in results:
        psnr_agg = results["psnr_aggregate"]
        print(f"PSNR: {psnr_agg['mean']:.2f} ± {psnr_agg['std']:.2f} dB")
        print(f"  Range: [{psnr_agg['min']:.2f}, {psnr_agg['max']:.2f}] dB")
    
    if "fvd" in results and "error" not in results["fvd"]:
        print(f"FVD: {results['fvd']['fvd']:.2f}")
        print(f"  (computed over {results['fvd']['num_videos']} videos)")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
