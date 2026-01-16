#!/usr/bin/env python3
"""Plot loss curves from flow matching training log CSV file."""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_loss_curves(csv_path, output_path=None):
    """Plot training and validation loss curves from CSV log file."""
    # Read the CSV file
    steps = []
    train_losses = []
    val_losses = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
    
    steps = np.array(steps)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training loss
    ax1.plot(steps, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Flow Matching HWM - Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add statistics for training loss
    min_train_loss = np.min(train_losses)
    min_train_step = steps[np.argmin(train_losses)]
    final_train_loss = train_losses[-1]
    
    stats_text = f'Min: {min_train_loss:.4f} at step {min_train_step}\n'
    stats_text += f'Final: {final_train_loss:.4f} at step {steps[-1]}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Plot validation loss
    ax2.plot(steps, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Flow Matching HWM - Validation Loss Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add statistics for validation loss
    min_val_loss = np.min(val_losses)
    min_val_step = steps[np.argmin(val_losses)]
    final_val_loss = val_losses[-1]
    
    stats_text = f'Min: {min_val_loss:.4f} at step {min_val_step}\n'
    stats_text += f'Final: {final_val_loss:.4f} at step {steps[-1]}'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Loss curves saved to: {output_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print(f"\nTraining Summary:")
    print(f"  Total steps: {len(steps)}")
    print(f"  Step range: {steps[0]} - {steps[-1]}")
    print(f"  Training loss: {train_losses.min():.4f} (min) -> {final_train_loss:.4f} (final)")
    print(f"  Validation loss: {val_losses.min():.4f} (min) -> {final_val_loss:.4f} (final)")

def main():
    checkpoint_dir = Path("/media/skr/storage/robot_world/humanoid_wm/checkpoints_flow_hwm_medium")
    csv_path = checkpoint_dir / "training_log.csv"
    output_path = checkpoint_dir / "loss_curves.png"
    
    if not csv_path.exists():
        print(f"Error: Training log not found at {csv_path}")
        return
    
    print(f"Reading training log from: {csv_path}")
    plot_loss_curves(csv_path, output_path)

if __name__ == "__main__":
    main()
