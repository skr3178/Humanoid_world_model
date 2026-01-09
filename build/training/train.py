"""Training script for Masked-HWM."""

import argparse
import csv
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from masked_hwm.config import MaskedHWMConfig
from masked_hwm.config_test import MaskedHWMTestConfig
from masked_hwm.config_12gb import MaskedHWM12GBConfig
from masked_hwm.config_minimal import MaskedHWMMinimalConfig
from masked_hwm.model import MaskedHWM
from data.dataset import HumanoidWorldModelDataset
from data.collator import MaskedHWMCollator


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Masked-HWM model")
    
    # Data
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, default=None, help="Test data directory for final evaluation")
    
    # Model
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON (optional)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=60000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    
    # Other
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_test_config", action="store_true", help="Use reduced test config")
    parser.add_argument("--use_12gb_config", action="store_true", help="Use 12GB GPU config (10 layers, 256 dim, 8 heads)")
    parser.add_argument("--use_minimal_config", action="store_true", help="Use minimal config for testing (4 layers, 128 dim, 1+1 clips)")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine mixed precision setting
    # Will be updated after config is loaded if config specifies mixed_precision
    # Default to bf16 if CUDA available, otherwise no mixed precision
    initial_mixed_precision = "bf16" if torch.cuda.is_available() else "no"

    # Initialize accelerator (may be re-initialized after config is loaded)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=initial_mixed_precision,
    )
    
    # Setup logging
    if accelerator.is_local_main_process:
        logger.setLevel("INFO")
    else:
        logger.setLevel("ERROR")
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    if args.config and os.path.exists(args.config):
        # Load from JSON file
        import json
        with open(args.config) as f:
            config_dict = json.load(f)
        config = MaskedHWMConfig(**config_dict)
    elif args.use_12gb_config:
        # Use 12GB GPU config (increased model size for better learning)
        config = MaskedHWM12GBConfig()
        config.train_data_dir = args.train_data_dir
        config.val_data_dir = args.val_data_dir
        config.test_data_dir = args.test_data_dir
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.warmup_steps = args.warmup_steps
        config.max_steps = args.max_steps
        config.save_steps = args.save_steps
        config.eval_steps = args.eval_steps
        config.logging_steps = args.logging_steps
        config.seed = args.seed
        logger.info(f"Using 12GB GPU CONFIG ({config.num_layers} layers, {config.d_model} dim, {config.num_heads} heads, {config.mlp_hidden} MLP)")
    elif args.use_minimal_config:
        # Use minimal config for testing on small GPUs
        config = MaskedHWMMinimalConfig()
        config.train_data_dir = args.train_data_dir
        config.val_data_dir = args.val_data_dir
        config.test_data_dir = args.test_data_dir
        config.batch_size = args.batch_size
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
        config.learning_rate = args.learning_rate
        config.warmup_steps = args.warmup_steps
        config.max_steps = args.max_steps
        config.save_steps = args.save_steps
        config.eval_steps = args.eval_steps
        config.logging_steps = args.logging_steps
        config.seed = args.seed
        logger.info(f"Using MINIMAL CONFIG for testing ({config.num_layers} layers, {config.d_model} dim, {config.num_heads} heads, {config.num_past_clips}+{config.num_future_clips} clips)")
    elif args.use_test_config:
        # Use reduced test config
        config = MaskedHWMTestConfig()
        config.train_data_dir = args.train_data_dir
        config.val_data_dir = args.val_data_dir
        config.test_data_dir = args.test_data_dir
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.warmup_steps = args.warmup_steps
        config.max_steps = args.max_steps
        config.save_steps = args.save_steps
        config.eval_steps = args.eval_steps
        config.logging_steps = args.logging_steps
        config.seed = args.seed
        logger.info("Using REDUCED TEST CONFIG (4 layers, 128 dim)")
    else:
        # Use defaults with command-line overrides
        config = MaskedHWMConfig(
            train_data_dir=args.train_data_dir,
            val_data_dir=args.val_data_dir,
            test_data_dir=args.test_data_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            seed=args.seed,
        )
    
    # Create datasets
    # Use explicit clip counts if available (paper: 2 past + 1 future)
    num_past_clips = getattr(config, 'num_past_clips', None)
    num_future_clips = getattr(config, 'num_future_clips', None)

    train_dataset = HumanoidWorldModelDataset(
        data_dir=config.train_data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        num_past_clips=num_past_clips,
        num_future_clips=num_future_clips,
        filter_interrupts=True,
        filter_overlaps=False,
    )

    val_dataset = HumanoidWorldModelDataset(
        data_dir=config.val_data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        num_past_clips=num_past_clips,
        num_future_clips=num_future_clips,
        filter_interrupts=True,
        filter_overlaps=True,
    )
    
    # Create collator
    collator = MaskedHWMCollator(config)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )
    
    # Use batch_size=1 for validation to reduce memory usage
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,  # Reduced from config.batch_size to prevent OOM during eval
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    model = MaskedHWM(config)

    # Log memory optimization settings
    mixed_prec = getattr(config, 'mixed_precision', 'bf16')
    grad_ckpt = getattr(config, 'use_gradient_checkpointing', False)
    logger.info(f"Memory optimizations: mixed_precision={mixed_prec}, gradient_checkpointing={grad_ckpt}")

    # Note: Gradient checkpointing requires model-level implementation
    # The main memory savings come from: small model, small batch, mixed precision
    if grad_ckpt:
        logger.info("Note: Gradient checkpointing flag is set but requires model-level support")
        logger.info("Primary memory savings: small model size + small batch + bf16 mixed precision")

    # Create optimizer (per paper: betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,  # Added per paper specification
        weight_decay=0.01,
    )
    
    # Create scheduler - cosine decay with warmup for better convergence
    num_training_steps = config.max_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory for later use
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "training_metrics.csv"
    if accelerator.is_local_main_process:
        import json
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Initialize metrics CSV file
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss", "learning_rate"])
    
    # Training loop
    model.train()
    global_step = 0
    progress_bar = tqdm(range(config.max_steps), disable=not accelerator.is_local_main_process)
    last_train_loss = None  # Track last training loss for CSV logging
    
    for epoch in range(1000):  # Large number, will break on max_steps
        for batch in train_dataloader:
            if global_step >= config.max_steps:
                break
            
            # Clip tokens to vocab size (for test config with reduced vocab)
            # Past tokens shouldn't have mask tokens, so safe to clip
            video_past = batch["video_past"] % config.vocab_size
            video_future_input = batch["video_future"].clone()
            # Keep mask tokens as-is, modulo the rest
            is_mask = (video_future_input == config.mask_token_id)
            video_future_input = video_future_input % config.vocab_size
            video_future_input[is_mask] = config.mask_token_id
            labels = batch["video_future_labels"] % config.vocab_size
            
            # Data validation (first batch only, for debugging)
            if global_step == 0:
                logger.info(f"Data validation (first batch):")
                logger.info(f"  video_past range: [{video_past.min()}, {video_past.max()}] (expected: [0, {config.vocab_size-1}])")
                logger.info(f"  video_future_input range: [{video_future_input.min()}, {video_future_input.max()}] (expected: [0, {config.mask_token_id}])")
                logger.info(f"  labels range: [{labels.min()}, {labels.max()}] (expected: [0, {config.vocab_size-1}])")
                logger.info(f"  mask sum: {batch['mask'].sum().item()} / {batch['mask'].numel()} ({100 * batch['mask'].sum().item() / batch['mask'].numel():.2f}% masked)")
                logger.info(f"  mask tokens in input: {(video_future_input == config.mask_token_id).sum().item()}")
                
                # Check for invalid targets (should never happen, but verify)
                invalid_targets = (labels < 0) | (labels >= config.vocab_size)
                if invalid_targets.any():
                    logger.error(f"  ERROR: Found {invalid_targets.sum().item()} invalid target tokens!")
                    logger.error(f"    Invalid target range: [{labels[invalid_targets].min()}, {labels[invalid_targets].max()}]")
                
                # Check logits shape
                logger.info(f"  Expected logits shape: (num_factors={config.num_factored_vocabs}, B, T_f, H, W, vocab_size={config.vocab_size})")
            
            # Forward pass
            logits = model(
                v_p_tokens=video_past,
                v_f_tokens=video_future_input,
                a_p=batch["actions_past"],
                a_f=batch["actions_future"],
            )
            
            # Validate logits shape and range
            if global_step == 0:
                logger.info(f"  Actual logits shape: {logits.shape}")
                logger.info(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                logger.info(f"  Logits vocab size: {logits.shape[-1]} (expected: {config.vocab_size})")
            
            # Ensure targets are in valid range for cross-entropy
            labels_clamped = torch.clamp(labels, 0, config.vocab_size - 1)
            if not torch.equal(labels, labels_clamped):
                logger.warning(f"Step {global_step}: Clamped {((labels != labels_clamped).sum().item())} target tokens to valid range")
                labels = labels_clamped
            
            # Compute loss
            loss = model.compute_loss(
                logits=logits,
                targets=labels,
                mask=batch["mask"],
            )
            
            # Additional loss diagnostics (first batch)
            if global_step == 0:
                # Compute per-factor losses for debugging
                num_factors = logits.shape[0]
                for i in range(num_factors):
                    factor_logits = logits[i]  # (B, T_f, H, W, vocab_size)
                    factor_targets = labels[:, i]  # (B, T_f, H, W)
                    mask_flat = batch["mask"].flatten()
                    logits_flat = factor_logits.flatten(0, -2)  # (B*T*H*W, vocab_size)
                    targets_flat = factor_targets.flatten()
                    loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                    masked_loss = (loss_flat * mask_flat.flatten()).sum() / mask_flat.sum().clamp(min=1)
                    logger.info(f"  Factor {i} loss: {masked_loss.item():.4f}")
                
                # Check if loss is reasonable (should be around log(vocab_size) for random)
                expected_random_loss = torch.log(torch.tensor(float(config.vocab_size)))
                logger.info(f"  Expected random baseline loss: {expected_random_loss.item():.4f}")
                logger.info(f"  Actual loss: {loss.item():.4f} ({loss.item() / expected_random_loss.item():.2f}x baseline)")
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at step {global_step}! Loss = {loss.item()}")
                logger.error("Skipping this batch...")
                continue
            
            # Backward pass
            accelerator.backward(loss)

            # ALWAYS compute gradient norm for diagnostics (before clipping)
            grad_norm_pre_clip = 0.0
            grad_norm_post_clip = 0.0
            max_grad = 0.0
            min_grad = float('inf')

            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm_pre_clip += param_norm.item() ** 2
                    max_grad = max(max_grad, p.grad.data.abs().max().item())
                    min_grad = min(min_grad, p.grad.data.abs().min().item())
            grad_norm_pre_clip = grad_norm_pre_clip ** (1. / 2)

            # Apply gradient updates
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping for training stability
                if hasattr(accelerator, 'clip_grad_norm_'):
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Compute gradient norm AFTER clipping
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm_post_clip += param_norm.item() ** 2
                grad_norm_post_clip = grad_norm_post_clip ** (1. / 2)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Enhanced logging at EVERY step for gradient diagnostics
            if global_step % config.logging_steps == 0:
                current_lr = lr_scheduler.get_last_lr()[0]

                # Log detailed gradient statistics
                logger.info(f"Step {global_step}: Loss = {loss.item():.4f}, LR = {current_lr:.6e}")
                logger.info(f"  Gradient Stats: norm_pre={grad_norm_pre_clip:.4f}, norm_post={grad_norm_post_clip:.4f}, max={max_grad:.4e}, min={min_grad:.4e}")

                # Check for gradient issues
                if grad_norm_pre_clip > 100:
                    logger.warning(f"  WARNING: Large gradient norm detected! ({grad_norm_pre_clip:.2f}) - may indicate instability")
                if grad_norm_pre_clip < 0.001:
                    logger.warning(f"  WARNING: Very small gradient norm! ({grad_norm_pre_clip:.6f}) - may indicate vanishing gradients")
                if torch.isnan(torch.tensor(grad_norm_pre_clip)):
                    logger.error(f"  ERROR: NaN gradient detected!")

                progress_bar.set_description(f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}, Grad: {grad_norm_pre_clip:.4f}")
                last_train_loss = loss.item()
            
            # Evaluation
            if global_step % config.eval_steps == 0 and global_step > 0:
                # Clear GPU cache before evaluation to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                model.eval()
                val_loss = 0.0
                val_count = 0
                
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        # Clip tokens to vocab size
                        val_video_past = val_batch["video_past"] % config.vocab_size
                        val_video_future_input = val_batch["video_future"].clone()
                        val_is_mask = (val_video_future_input == config.mask_token_id)
                        val_video_future_input = val_video_future_input % config.vocab_size
                        val_video_future_input[val_is_mask] = config.mask_token_id
                        val_labels = val_batch["video_future_labels"] % config.vocab_size
                        
                        val_logits = model(
                            v_p_tokens=val_video_past,
                            v_f_tokens=val_video_future_input,
                            a_p=val_batch["actions_past"],
                            a_f=val_batch["actions_future"],
                        )
                        val_loss_batch = model.compute_loss(
                            logits=val_logits,
                            targets=val_labels,
                            mask=val_batch["mask"],
                        )
                        val_loss += val_loss_batch.item()
                        val_count += 1
                
                val_loss /= val_count
                logger.info(f"Step {global_step}: Val Loss = {val_loss:.4f}")
                
                # Save metrics to CSV (both train and val if available)
                if accelerator.is_local_main_process:
                    train_loss_str = f"{last_train_loss:.6f}" if last_train_loss is not None else ""
                    with open(metrics_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            global_step, 
                            train_loss_str, 
                            f"{val_loss:.6f}", 
                            f"{lr_scheduler.get_last_lr()[0]:.2e}"
                        ])
                
                model.train()
            
            # Save checkpoint
            if global_step % config.save_steps == 0 and global_step > 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "global_step": global_step,
                    "config": config,
                }, checkpoint_dir / "pytorch_model.bin")
                
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            global_step += 1
            progress_bar.update(1)
        
        if global_step >= config.max_steps:
            break
    
    # Save final checkpoint
    logger.info("Training completed! Saving final checkpoint...")
    final_checkpoint_dir = output_dir / "checkpoint-final"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save({
        "model_state_dict": unwrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "global_step": global_step,
        "config": config,
    }, final_checkpoint_dir / "pytorch_model.bin")
    
    logger.info(f"Final checkpoint saved to {final_checkpoint_dir}")
    
    # Run test set evaluation if test_data_dir is provided
    test_loss = None
    if hasattr(config, 'test_data_dir') and config.test_data_dir and os.path.exists(config.test_data_dir):
        logger.info(f"Running final evaluation on test set: {config.test_data_dir}")
        
        test_dataset = HumanoidWorldModelDataset(
            data_dir=config.test_data_dir,
            num_past_frames=config.num_past_frames,
            num_future_frames=config.num_future_frames,
            filter_interrupts=True,
            filter_overlaps=True,
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,  # Reduced from config.batch_size to prevent OOM during eval
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )
        
        test_dataloader = accelerator.prepare(test_dataloader)
        
        # Clear GPU cache before test evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model.eval()
        test_loss = 0.0
        test_count = 0
        
        with torch.no_grad():
            for test_batch in tqdm(test_dataloader, desc="Test Evaluation", disable=not accelerator.is_local_main_process):
                # Clip tokens to vocab size
                test_video_past = test_batch["video_past"] % config.vocab_size
                test_video_future_input = test_batch["video_future"].clone()
                test_is_mask = (test_video_future_input == config.mask_token_id)
                test_video_future_input = test_video_future_input % config.vocab_size
                test_video_future_input[test_is_mask] = config.mask_token_id
                test_labels = test_batch["video_future_labels"] % config.vocab_size
                
                test_logits = model(
                    v_p_tokens=test_video_past,
                    v_f_tokens=test_video_future_input,
                    a_p=test_batch["actions_past"],
                    a_f=test_batch["actions_future"],
                )
                test_loss_batch = model.compute_loss(
                    logits=test_logits,
                    targets=test_labels,
                    mask=test_batch["mask"],
                )
                test_loss += test_loss_batch.item()
                test_count += 1
        
        test_loss /= test_count
        logger.info(f"TEST SET LOSS: {test_loss:.4f}")
        
        # Save test metrics to CSV
        if accelerator.is_local_main_process:
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "FINAL_TEST", 
                    "", 
                    f"{test_loss:.6f}", 
                    ""
                ])
            
            # Also save to a separate test results file
            test_results_path = output_dir / "test_results.json"
            import json
            test_results = {
                "test_data_dir": config.test_data_dir,
                "test_loss": test_loss,
                "test_samples": test_count,
                "final_step": global_step,
            }
            with open(test_results_path, "w") as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Test results saved to {test_results_path}")
    
    # Run video generation test if requested
    if accelerator.is_local_main_process:
        logger.info("Running video generation test...")
        try:
            import subprocess
            test_script = Path(__file__).parent.parent / "test_generate_video.py"
            if test_script.exists():
                cmd = [
                    sys.executable,
                    str(test_script),
                    "--checkpoint", str(final_checkpoint_dir / "pytorch_model.bin"),
                    "--data_dir", config.val_data_dir,
                    "--output_dir", str(output_dir / "test_videos"),
                    "--tokenizer_dir", config.tokenizer_checkpoint_dir,
                    "--num_samples", "3",
                ]
                if isinstance(config, MaskedHWMTestConfig):
                    cmd.append("--use_test_config")
                
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("Video generation test completed successfully!")
                    logger.info(f"Videos saved to: {output_dir / 'test_videos'}")
                else:
                    logger.warning(f"Video generation test failed: {result.stderr}")
            else:
                logger.warning(f"Test script not found: {test_script}")
        except Exception as e:
            logger.warning(f"Failed to run video generation test: {e}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
