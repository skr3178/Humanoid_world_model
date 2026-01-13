"""Training script for Flow-HWM.

Implements flow matching training:
1. Sample t ~ U(0, 1)
2. Sample X_0 ~ N(0, I)
3. Construct X_t = t*X_1 + (1 - (1 - sigma_min)*t)*X_0
4. Compute V_t = X_1 - (1 - sigma_min)*X_0
5. Predict velocity u_theta(X_t, P, t)
6. Loss = MSE(u_theta, V_t)

Usage:
    python train.py --train_data_dir /path/to/train --val_data_dir /path/to/val

    # With reduced model for testing
    python train.py --use_small_config --max_steps 100
"""

import argparse
import csv
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_hwm.config import FlowHWMConfig, FlowHWMConfigMedium, FlowHWMConfigSmall, FlowHWMConfigTest
from flow_hwm.model import FlowHWM, create_flow_hwm
from flow_hwm.flow_matching import (
    construct_flow_path,
    compute_target_velocity,
    flow_matching_loss,
    sample_timesteps,
    sample_noise,
)
from flow_hwm.dataset_latent import FlowHWMDataset, create_flow_hwm_dataloader


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create cosine learning rate schedule with warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(
    model: FlowHWM,
    batch: Dict[str, torch.Tensor],
    config: FlowHWMConfig,
    accelerator: Accelerator,
) -> torch.Tensor:
    """Single training step.

    Args:
        model: FlowHWM model
        batch: Dictionary with latent_past, latent_future, actions_past, actions_future
        config: Model configuration
        accelerator: Hugging Face Accelerator

    Returns:
        Loss tensor
    """
    # Unpack batch
    v_p = batch["latent_past"]  # (B, C, T_p, H, W)
    v_f = batch["latent_future"]  # (B, C, T_f, H, W) = X_1 (target)
    a_p = batch["actions_past"]  # (B, T_p_frames, action_dim)
    a_f = batch["actions_future"]  # (B, T_f_frames, action_dim)

    B = v_f.shape[0]
    device = v_f.device

    # Sample timestep t ~ U(0, 1)
    t = sample_timesteps(B, device)

    # Sample noise X_0 ~ N(0, I)
    x0 = sample_noise(v_f.shape, device)

    # Construct X_t along the flow path
    x_t = construct_flow_path(x0, v_f, t, config.sigma_min)

    # Compute target velocity V_t
    target_velocity = compute_target_velocity(x0, v_f, config.sigma_min)

    # Classifier-free guidance dropout
    # Randomly zero out conditioning for some samples
    if config.cfg_drop_prob > 0:
        drop_mask = torch.rand(B, device=device) < config.cfg_drop_prob
        if drop_mask.any():
            # Zero out past video and actions for dropped samples
            v_p = v_p.clone()
            a_p = a_p.clone()
            v_p[drop_mask] = 0
            a_p[drop_mask] = 0

    # Forward pass: predict velocity
    predicted_velocity = model(
        v_f_noisy=x_t,
        v_p=v_p,
        a_p=a_p,
        a_f=a_f,
        t=t,
    )

    # Compute loss
    loss = flow_matching_loss(predicted_velocity, target_velocity)

    return loss


@torch.no_grad()
def evaluate(
    model: FlowHWM,
    dataloader: DataLoader,
    config: FlowHWMConfig,
    accelerator: Accelerator,
    num_batches: int = 50,
) -> float:
    """Evaluate model on validation set.

    Args:
        model: FlowHWM model
        dataloader: Validation dataloader
        config: Model configuration
        accelerator: Accelerator
        num_batches: Number of batches to evaluate

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    count = 0

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        loss = train_step(model, batch, config, accelerator)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


def save_checkpoint(
    model: FlowHWM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    loss: float,
    checkpoint_dir: Path,
    accelerator: Accelerator,
):
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save with accelerator for distributed training support
    accelerator.save_state(checkpoint_dir / f"checkpoint-{step}")

    # Also save model separately for easy loading
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(
        {
            "step": step,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        },
        checkpoint_dir / f"model-{step}.pt",
    )


def main():
    parser = argparse.ArgumentParser(description="Train Flow-HWM")

    # Data paths
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/train_v2.0",
        help="Training data directory",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0",
        help="Validation data directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints_flow_hwm",
        help="Checkpoint directory",
    )

    # Model configuration
    parser.add_argument("--use_medium_config", action="store_true", help="Use medium model config (between full and small)")
    parser.add_argument("--use_small_config", action="store_true", help="Use small model config")
    parser.add_argument("--use_test_config", action="store_true", help="Use test model config")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)

    # Evaluation and logging
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=None, help="Checkpoint frequency")
    parser.add_argument("--logging_steps", type=int, default=None, help="Logging frequency")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Select configuration
    if args.use_test_config:
        config = FlowHWMConfigTest()
    elif args.use_small_config:
        config = FlowHWMConfigSmall()
    elif args.use_medium_config:
        config = FlowHWMConfigMedium()
    else:
        config = FlowHWMConfig()

    # Override config with command line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.eval_steps is not None:
        config.eval_steps = args.eval_steps
    if args.save_steps is not None:
        config.save_steps = args.save_steps
    if args.logging_steps is not None:
        config.logging_steps = args.logging_steps

    # Update data paths
    config.train_data_dir = args.train_data_dir
    config.val_data_dir = args.val_data_dir

    # Set seed
    set_seed(args.seed)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.mixed_precision == "bf16" else None,
    )

    # Print configuration
    if accelerator.is_main_process:
        print("=" * 60)
        print("Flow-HWM Training")
        print("=" * 60)
        print(f"Model: {config.num_layers} layers, {config.d_model} dim, {config.num_heads} heads")
        print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Max steps: {config.max_steps}")
        print(f"Warmup steps: {config.warmup_steps}")
        print(f"Mixed precision: {config.mixed_precision}")
        print("=" * 60)

    # Create model
    model = create_flow_hwm(config)

    if accelerator.is_main_process:
        num_params = model.get_num_params()
        print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # Create datasets
    train_dataloader = create_flow_hwm_dataloader(
        data_dir=config.train_data_dir,
        batch_size=config.batch_size,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        latent_dim=config.latent_dim,
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_dataloader = create_flow_hwm_dataloader(
        data_dir=config.val_data_dir,
        batch_size=config.batch_size,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        latent_dim=config.latent_dim,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # Enable gradient checkpointing if configured
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable() if hasattr(model, 'gradient_checkpointing_enable') else None

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        start_step = int(args.resume_from.split("-")[-1])
        print(f"Resumed from step {start_step}")

    # Setup logging
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_file = checkpoint_dir / "training_log.csv"
    if accelerator.is_main_process and not log_file.exists():
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss", "lr"])

    # Training loop
    model.train()
    train_iter = iter(train_dataloader)
    running_loss = 0.0
    log_steps = 0

    for step in range(start_step, config.max_steps):
        # Get next batch (with cycling)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        # Forward pass with gradient accumulation
        with accelerator.accumulate(model):
            loss = train_step(model, batch, config, accelerator)
            accelerator.backward(loss)

            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        log_steps += 1

        # Logging
        if (step + 1) % config.logging_steps == 0:
            avg_loss = running_loss / log_steps
            lr = scheduler.get_last_lr()[0]

            if accelerator.is_main_process:
                print(f"Step {step + 1}/{config.max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

            running_loss = 0.0
            log_steps = 0

        # Evaluation
        if (step + 1) % config.eval_steps == 0:
            val_loss = evaluate(model, val_dataloader, config, accelerator)

            if accelerator.is_main_process:
                print(f"Step {step + 1} | Validation Loss: {val_loss:.4f}")

                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step + 1, avg_loss if log_steps == 0 else running_loss / log_steps, val_loss, lr])

        # Save checkpoint
        if (step + 1) % config.save_steps == 0:
            if accelerator.is_main_process:
                save_checkpoint(
                    model, optimizer, scheduler, step + 1,
                    running_loss / max(log_steps, 1),
                    checkpoint_dir, accelerator
                )
                print(f"Saved checkpoint at step {step + 1}")

    # Final checkpoint
    if accelerator.is_main_process:
        save_checkpoint(
            model, optimizer, scheduler, config.max_steps,
            running_loss / max(log_steps, 1),
            checkpoint_dir, accelerator
        )
        print("Training complete!")


if __name__ == "__main__":
    main()
