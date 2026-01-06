"""Training script for Masked-HWM."""

import argparse
import os
import random
from pathlib import Path

import torch
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from masked_hwm.config import MaskedHWMConfig
from masked_hwm.config_test import MaskedHWMTestConfig
from masked_hwm.config_12gb import MaskedHWM12GBConfig
from masked_hwm.model import MaskedHWM
from data.dataset import HumanoidWorldModelDataset
from data.collator import MaskedHWMCollator


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Masked-HWM model")
    
    # Data
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, required=True)
    
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
    parser.add_argument("--use_12gb_config", action="store_true", help="Use 12GB GPU config (12 layers, 256 dim)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
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
        # Use 12GB GPU config (reduced model size)
        config = MaskedHWM12GBConfig()
        config.train_data_dir = args.train_data_dir
        config.val_data_dir = args.val_data_dir
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.warmup_steps = args.warmup_steps
        config.max_steps = args.max_steps
        config.save_steps = args.save_steps
        config.eval_steps = args.eval_steps
        config.logging_steps = args.logging_steps
        config.seed = args.seed
        logger.info("Using 12GB GPU CONFIG (12 layers, 256 dim, 4 heads)")
    elif args.use_test_config:
        # Use reduced test config
        config = MaskedHWMTestConfig()
        config.train_data_dir = args.train_data_dir
        config.val_data_dir = args.val_data_dir
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
    train_dataset = HumanoidWorldModelDataset(
        data_dir=config.train_data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        filter_interrupts=True,
        filter_overlaps=False,
    )
    
    val_dataset = HumanoidWorldModelDataset(
        data_dir=config.val_data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
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
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    model = MaskedHWM(config)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    # Create scheduler - linear decay (per paper)
    num_training_steps = config.max_steps
    lr_scheduler = get_linear_schedule_with_warmup(
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
    if accelerator.is_local_main_process:
        import json
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    # Training loop
    model.train()
    global_step = 0
    progress_bar = tqdm(range(config.max_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(1000):  # Large number, will break on max_steps
        for batch in train_dataloader:
            if global_step >= config.max_steps:
                break
            
            # Clip tokens to vocab size (for test config with reduced vocab)
            video_past = batch["video_past"] % config.vocab_size
            video_future_input = batch["video_future"].clone()
            # Keep mask tokens as-is, modulo the rest
            is_mask = (video_future_input == config.mask_token_id)
            video_future_input = video_future_input % config.vocab_size
            video_future_input[is_mask] = config.mask_token_id
            labels = batch["video_future_labels"] % config.vocab_size
            
            # Forward pass
            logits = model(
                v_p_tokens=video_past,
                v_f_tokens=video_future_input,
                a_p=batch["actions_past"],
                a_f=batch["actions_future"],
            )
            
            # Compute loss
            loss = model.compute_loss(
                logits=logits,
                targets=labels,
                mask=batch["mask"],
            )
            
            # Backward pass
            accelerator.backward(loss)
            
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if global_step % config.logging_steps == 0:
                progress_bar.set_description(f"Loss: {loss.item():.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                logger.info(f"Step {global_step}: Loss = {loss.item():.4f}")
            
            # Evaluation
            if global_step % config.eval_steps == 0 and global_step > 0:
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
