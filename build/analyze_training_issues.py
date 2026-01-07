#!/usr/bin/env python3
"""Analyze training log and diagnose loss plateau issues."""

import re
import sys
from pathlib import Path

def analyze_log(log_path):
    """Extract and analyze loss progression from training log."""
    with open(log_path) as f:
        content = f.read()
    
    # Extract loss values
    loss_pattern = r'Loss: ([\d.]+)'
    losses = [float(x) for x in re.findall(loss_pattern, content)]
    
    if not losses:
        print("No loss values found in log!")
        return
    
    # Extract step numbers
    step_pattern = r'\| (\d+)/60000'
    steps = [int(x) for x in re.findall(step_pattern, content)]
    
    # Extract learning rates
    lr_pattern = r'LR: ([\d.e+-]+)'
    lrs = [float(x) for x in re.findall(lr_pattern, content)]
    
    print("=" * 80)
    print("TRAINING ANALYSIS")
    print("=" * 80)
    print(f"\nTotal loss values extracted: {len(losses)}")
    print(f"Total steps extracted: {len(steps)}")
    print(f"Total LR values extracted: {len(lrs)}")
    
    if losses:
        print(f"\nLoss Statistics:")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Total improvement: {losses[0] - losses[-1]:.4f}")
        print(f"  Improvement rate: {(losses[0] - losses[-1]) / len(losses) * 100:.4f} per 100 steps")
        
        # Check for plateau
        if len(losses) > 100:
            recent_losses = losses[-100:]
            recent_std = sum((x - sum(recent_losses)/len(recent_losses))**2 for x in recent_losses) / len(recent_losses)
            recent_std = recent_std ** 0.5
            print(f"  Recent 100 steps std dev: {recent_std:.4f}")
            if recent_std < 0.5:
                print("  ⚠️  WARNING: Loss appears to be plateauing (low variance)")
        
        # Check improvement rate
        if len(losses) > 10:
            early_avg = sum(losses[:10]) / 10
            late_avg = sum(losses[-10:]) / 10
            improvement = early_avg - late_avg
            print(f"  Early average (first 10): {early_avg:.4f}")
            print(f"  Late average (last 10): {late_avg:.4f}")
            print(f"  Improvement: {improvement:.4f}")
            if improvement < 1.0:
                print("  ⚠️  WARNING: Very slow improvement - model may be too small or LR too low")
    
    if lrs:
        print(f"\nLearning Rate Statistics:")
        # Find first non-zero LR (after warmup)
        non_zero_lrs = [lr for lr in lrs if lr > 0]
        if non_zero_lrs:
            print(f"  First non-zero LR: {non_zero_lrs[0]:.2e}")
            print(f"  Final LR: {lrs[-1]:.2e}")
            if len(non_zero_lrs) > 1 and non_zero_lrs[0] > 0:
                lr_decay = (non_zero_lrs[0] - lrs[-1]) / non_zero_lrs[0] * 100
                print(f"  LR decay: {lr_decay:.2f}%")
                if lr_decay > 50:
                    print("  ⚠️  WARNING: Learning rate has decayed significantly")
        else:
            print(f"  Initial LR: {lrs[0]:.2e}")
            print(f"  Final LR: {lrs[-1]:.2e}")
    
    if steps:
        print(f"\nTraining Progress:")
        print(f"  Steps completed: {steps[-1]}")
        print(f"  Progress: {steps[-1] / 60000 * 100:.1f}%")
    
    # Extract model config from log
    config_match = re.search(r'Model config.*?\n(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    if config_match:
        print(f"\nModel Configuration (from log):")
        print(config_match.group(1))
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if losses and (losses[0] - losses[-1]) < 2.0:
        print("\n1. MODEL CAPACITY:")
        print("   - Current model appears too small (2 layers, 64 dim)")
        print("   - Recommendation: Use config_12gb.py (6 layers, 128 dim)")
        print("   - If still OOM, try gradient checkpointing")
    
    if lrs and lrs[-1] < 1e-5:
        print("\n2. LEARNING RATE:")
        print("   - LR has decayed too much")
        print("   - Recommendation: Increase base LR to 5e-5 or use cosine schedule")
        print("   - Consider longer warmup (500-1000 steps)")
    
    if losses and len(losses) > 100:
        recent_improvement = losses[-100] - losses[-1]
        if recent_improvement < 0.1:
            print("\n3. LOSS PLATEAU:")
            print("   - Loss has stopped improving")
            print("   - Recommendations:")
            print("     a) Increase model capacity")
            print("     b) Increase learning rate")
            print("     c) Check for gradient clipping issues")
            print("     d) Verify data quality and preprocessing")
    
    print("\n4. MEMORY ISSUES:")
    print("   - Training was killed (exit code 137 = OOM)")
    print("   - Recommendations:")
    print("     a) Reduce batch size or increase gradient accumulation")
    print("     b) Enable gradient checkpointing")
    print("     c) Use smaller model (but not too small!)")
    print("     d) Reduce eval batch size")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_training_issues.py <log_file>")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    analyze_log(log_path)
