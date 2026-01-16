#!/usr/bin/env python3
"""Debug Flow-HWM decoding and noise schedule outputs.

Generates:
1) Ground-truth video decoded from original tokens
2) Predicted video decoded from checkpoint latents
3) Side-by-side comparison
4) Noise schedule video showing X_t along the flow path
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flow_hwm.config import FlowHWMConfig, FlowHWMConfigMedium
from flow_hwm.dataset_latent import FlowHWMDataset
from flow_hwm.flow_matching import construct_flow_path, sample_noise
from flow_hwm.inference import generate_video_latents, load_model_from_checkpoint

# Import base dataset to get original tokens
sys.path.insert(0, "/media/skr/storage/robot_world/humanoid_wm/build")
from data.dataset import HumanoidWorldModelDataset

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("ERROR: imageio not available. Install with: pip install imageio imageio-ffmpeg")
    sys.exit(1)


def latents_to_factorized_tokens(latents: torch.Tensor) -> torch.Tensor:
    """Convert continuous latents to factorized tokens for Cosmos decoder.
    
    NOTE: This method (taking first 3 channels) was tested and found to be the best
    approach among multiple alternatives (averaging groups, selecting channels 0/5/10, etc.).
    Test results: Mean Absolute Error ~3859 vs ~5318 for other methods.
    See: build/helper/test_latent_to_token_conversion.py
    """
    B, C, T, H, W = latents.shape

    # Clamp to expected range before de-quantization
    latents = torch.clamp(latents, -1.0, 1.0)

    # Use first 3 channels as factor tokens (best method based on testing)
    if C >= 3:
        latents_3ch = latents[:, :3, :, :, :]
    else:
        latents_3ch = latents.repeat(1, (3 // C + 1), 1, 1, 1)[:, :3, :, :, :]

    # Reverse normalization: [-1, 1] -> [0, 65535]
    latents_normalized = (latents_3ch + 1.0) / 2.0
    latents_normalized = torch.clamp(latents_normalized, 0.0, 1.0)
    tokens = (latents_normalized * 65535.0).round().long()
    return torch.clamp(tokens, 0, 65535)


def decode_tokens_to_video(tokens: torch.Tensor, decoder, device: str) -> np.ndarray:
    """Decode factorized tokens to video frames."""
    from cosmos_tokenizer.utils import tensor2numpy

    B, num_factors, T, H, W = tokens.shape
    decoded_frames = []
    for t in range(T):
        clip_tokens = tokens[:, :, t, :, :].to(device)
        with torch.no_grad():
            decoded_clip = decoder.decode(clip_tokens).float()
            decoded_frames.append(decoded_clip)
    video = torch.cat(decoded_frames, dim=2)
    return tensor2numpy(video)


def decode_latents_to_video(latents: torch.Tensor, decoder, device: str) -> np.ndarray:
    """Decode continuous latents to video frames using Cosmos decoder."""
    tokens = latents_to_factorized_tokens(latents)
    return decode_tokens_to_video(tokens, decoder, device)


def decode_cv_latents_to_video(latents: torch.Tensor, cv_decoder, device: str) -> np.ndarray:
    """Decode continuous CV latents directly using the CV decoder.

    This is the correct way to decode latents from the CV encoder.
    The CV decoder expects latents of shape (B, C, T, H, W).
    """
    from cosmos_tokenizer.utils import tensor2numpy

    # latents: (B, C, T, H, W) - already in correct format
    target_dtype = next(cv_decoder._dec_model.parameters()).dtype
    latents = latents.to(device=device, dtype=target_dtype)

    with torch.no_grad():
        # CV decoder.decode() expects continuous latents and returns video
        decoded = cv_decoder.decode(latents).float()

    return tensor2numpy(decoded)


def add_label_bar(frame: np.ndarray, label: str, bar_height: int = 64) -> np.ndarray:
    """Add a label bar at the top of a frame."""
    from PIL import Image, ImageDraw, ImageFont

    H, W, C = frame.shape
    bar = np.zeros((bar_height, W, C), dtype=np.uint8)
    bar_img = Image.fromarray(bar)
    draw = ImageDraw.Draw(bar_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (W - text_width) // 2
    text_y = (bar_height - (bbox[3] - bbox[1])) // 2
    draw.text((text_x, text_y), label, fill="white", font=font)

    return np.concatenate([np.array(bar_img), frame], axis=0)


def pad_frame_to_size(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad frame with black pixels to reach target size."""
    h, w, c = frame.shape
    if h == target_h and w == target_w:
        return frame
    padded = np.zeros((target_h, target_w, c), dtype=frame.dtype)
    padded[:h, :w] = frame
    return padded


def save_video(frames: np.ndarray, output_path: Path, fps: int = 30) -> None:
    """Save a numpy video array (T, H, W, 3) to mp4 with high quality."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-crf", "18", "-preset", "slow"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved: {output_path}")


def create_side_by_side_video(
    left_video: np.ndarray,
    right_video: np.ndarray,
    output_path: Path,
    fps: int = 30,
    add_labels: bool = True,
) -> None:
    """Create side-by-side comparison video (left | right)."""
    T = min(left_video.shape[0], right_video.shape[0])
    combined_frames = []
    for t in range(T):
        left_frame = left_video[t]
        right_frame = right_video[t]
        if add_labels:
            left_frame = add_label_bar(left_frame, "Ground Truth")
            right_frame = add_label_bar(right_frame, "Prediction")
        target_h = max(left_frame.shape[0], right_frame.shape[0])
        target_w = max(left_frame.shape[1], right_frame.shape[1])
        left_frame = pad_frame_to_size(left_frame, target_h, target_w)
        right_frame = pad_frame_to_size(right_frame, target_h, target_w)
        combined_frames.append(np.concatenate([left_frame, right_frame], axis=1))
    save_video(np.stack(combined_frames, axis=0), output_path, fps=fps)


def create_noise_schedule_video(
    target_latents: torch.Tensor,
    cv_decoder,
    device: str,
    sigma_min: float,
    noise_std: float,
    num_timesteps: int,
    output_path: Path,
    fps: int,
    seed: int,
    add_labels: bool = True,
) -> None:
    """Create a video showing the flow path X_t at multiple timesteps.

    Uses the CV decoder to decode continuous latents directly.
    """
    torch.manual_seed(seed)
    x0 = sample_noise(target_latents.shape, device, std=noise_std)
    t_values = torch.linspace(0.0, 1.0, num_timesteps, device=device)

    videos = []
    labels = []
    for t in t_values:
        if torch.isclose(t, torch.tensor(1.0, device=device)):
            # Ensure the final frame is the clean target (no noise at t=1).
            x_t = target_latents
        else:
            x_t = construct_flow_path(x0, target_latents, t, sigma_min)
        # Use CV decoder for continuous latents
        video_np = decode_cv_latents_to_video(x_t, cv_decoder, device)[0]
        videos.append(video_np)
        labels.append(f"t={float(t):.2f}")

    # Stack all timesteps horizontally for each frame index
    T = min(v.shape[0] for v in videos)
    combined_frames = []
    for idx in range(T):
        row = []
        for vid, label in zip(videos, labels):
            frame = vid[idx]
            if add_labels:
                frame = add_label_bar(frame, label)
            row.append(frame)
        combined_frames.append(np.concatenate(row, axis=1))

    save_video(np.stack(combined_frames, axis=0), output_path, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Flow-HWM videos and noise schedule")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/checkpoints_flow_hwm_medium/model-60000.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/videos/debug_flowhwm",
        help="Output directory for videos",
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to process")
    parser.add_argument("--num_steps", type=int, default=50, help="Euler integration steps")
    parser.add_argument("--cfg_scale", type=float, default=1.5, help="CFG scale")
    parser.add_argument("--num_noise_steps", type=int, default=6, help="Timesteps for noise schedule")
    parser.add_argument("--noise_seed", type=int, default=0, help="Seed for noise schedule")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--max_shards", type=int, default=1, help="Max shards to load")
    parser.add_argument("--use_medium_config", action="store_true", help="Use FlowHWMConfigMedium")
    parser.add_argument("--no_labels", action="store_true", help="Disable label bars")
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer",
        help="Path to Cosmos DV (discrete video) tokenizer directory",
    )
    parser.add_argument(
        "--cv_tokenizer_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/cosmos_tokenizer/Continuous_video",
        help="Path to Cosmos CV (continuous video) tokenizer directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config
    use_medium = args.use_medium_config or ("medium" in args.checkpoint.lower())
    config = FlowHWMConfigMedium() if use_medium else FlowHWMConfig()
    print(f"Using config: {'FlowHWMConfigMedium' if use_medium else 'FlowHWMConfig'}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, config, device)
    print(f"Model loaded: {model.get_num_params():,} parameters")

    if config.mixed_precision == "bf16" and device == "cuda":
        inference_dtype = torch.bfloat16
    elif config.mixed_precision == "fp16" and device == "cuda":
        inference_dtype = torch.float16
    else:
        inference_dtype = torch.float32

    model = model.to(dtype=inference_dtype)

    # Load datasets
    print(f"Loading datasets from {args.data_dir}...")
    base_dataset = HumanoidWorldModelDataset(
        data_dir=args.data_dir,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        use_factored_tokens=True,
        filter_interrupts=False,
        filter_overlaps=False,
        max_shards=args.max_shards,
    )
    flow_dataset = FlowHWMDataset(
        data_dir=args.data_dir,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        latent_dim=config.latent_dim,
        max_shards=args.max_shards,
    )

    if len(base_dataset) == 0:
        print("ERROR: Dataset is empty")
        return

    # Load decoders
    print("Loading Cosmos decoders...")
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    # DV decoder for discrete tokens (ground truth from base dataset)
    dv_decoder = CausalVideoTokenizer(
        checkpoint_dec=f"{args.tokenizer_dir}/decoder.jit",
        device=device,
        dtype="bfloat16" if device == "cuda" else "float32",
    )

    # CV decoder for continuous latents (from FlowHWMDataset)
    cv_decoder = CausalVideoTokenizer(
        checkpoint_dec=f"{args.cv_tokenizer_dir}/decoder.jit",
        device=device,
        dtype="bfloat16" if device == "cuda" else "float32",
    )

    for sample_idx in range(min(args.num_samples, len(base_dataset))):
        print(f"\n{'=' * 60}")
        print(f"Processing sample {sample_idx}")
        print(f"{'=' * 60}")

        # Ground truth from original tokens
        base_sample = base_dataset[sample_idx]
        gt_tokens = base_sample["video_future"]  # (T, 3, H, W)
        gt_tokens = gt_tokens.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, 3, T, H, W)

        # Conditioning and target latents
        flow_sample = flow_dataset[sample_idx]
        v_p = flow_sample["latent_past"].unsqueeze(0).to(device, dtype=inference_dtype)
        a_p = flow_sample["actions_past"].unsqueeze(0).to(device, dtype=inference_dtype)
        a_f = flow_sample["actions_future"].unsqueeze(0).to(device, dtype=inference_dtype)
        latent_future = flow_sample["latent_future"].unsqueeze(0).to(device, dtype=inference_dtype)

        print("Decoding ground truth from original tokens...")
        gt_video = decode_tokens_to_video(gt_tokens, dv_decoder, device)[0]

        print("Generating prediction latents...")
        with torch.no_grad():
            pred_latents = generate_video_latents(
                model,
                v_p,
                a_p,
                a_f,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                verbose=True,
            )

        print("Decoding prediction latents...")
        # Use CV decoder since pred_latents are continuous latents
        pred_video = decode_cv_latents_to_video(pred_latents, cv_decoder, device)[0]

        sample_dir = output_dir / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        save_video(gt_video, sample_dir / "ground_truth.mp4", fps=args.fps)
        save_video(pred_video, sample_dir / "prediction.mp4", fps=args.fps)
        create_side_by_side_video(
            gt_video,
            pred_video,
            sample_dir / "comparison.mp4",
            fps=args.fps,
            add_labels=not args.no_labels,
        )

        print("Creating noise schedule video...")
        noise_std = getattr(config, 'noise_std', 0.5)
        create_noise_schedule_video(
            target_latents=latent_future,
            cv_decoder=cv_decoder,
            device=device,
            sigma_min=config.sigma_min,
            noise_std=noise_std,
            num_timesteps=args.num_noise_steps,
            output_path=sample_dir / "noise_schedule.mp4",
            fps=args.fps,
            seed=args.noise_seed,
            add_labels=not args.no_labels,
        )

    print(f"\nDone! Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
