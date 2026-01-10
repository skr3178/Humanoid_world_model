#!/usr/bin/env python3
"""Plot loss curve from training log file."""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """Parse training log to extract step numbers and loss values."""
    steps = []
    losses = []
    
    # Pattern to match: "Loss: 10.4759, LR: 1.00e-04, Grad: 0.0223:   4%|▍         | 2637/60000"
    loss_pattern = re.compile(r'Loss: ([\d.]+).*?\| (\d+)/\d+')
    
    with open(log_path, 'r') as f:
        for line in f:
            match = loss_pattern.search(line)
            if match:
                loss = float(match.group(1))
                step = int(match.group(2))
                steps.append(step)
                losses.append(loss)
    
    return np.array(steps), np.array(losses)

def plot_loss_curve(steps, losses, output_path=None):
    """Plot loss curve."""
    plt.figure(figsize=(12, 6))
    
    # Plot all points
    plt.plot(steps, losses, alpha=0.3, linewidth=0.5, label='Loss (all steps)', color='blue')
    
    # Plot smoothed curve (moving average)
    if len(losses) > 50:
        window_size = min(50, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = steps[window_size//2:len(smoothed)+window_size//2]
        plt.plot(smoothed_steps, smoothed, linewidth=2, label=f'Loss (smoothed, window={window_size})', color='red')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add statistics
    min_loss = np.min(losses)
    min_step = steps[np.argmin(losses)]
    final_loss = losses[-1]
    final_step = steps[-1]
    
    stats_text = f'Min Loss: {min_loss:.4f} at step {min_step}\n'
    stats_text += f'Final Loss: {final_loss:.4f} at step {final_step}\n'
    stats_text += f'Total Steps: {len(steps)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Loss curve saved to: {output_path}")
    else:
        plt.show()

def main():
    log_path = Path("/media/skr/storage/robot_world/humanoid_wm/build/train_quick_v2.log")
    output_path = Path("/media/skr/storage/robot_world/humanoid_wm/build/training_loss_curve.png")
    
    print(f"Parsing log file: {log_path}")
    steps, losses = parse_log_file(log_path)
    
    print(f"Extracted {len(steps)} data points")
    print(f"Step range: {steps[0]} - {steps[-1]}")
    print(f"Loss range: {losses.min():.4f} - {losses.max():.4f}")
    
    plot_loss_curve(steps, losses, output_path)

if __name__ == "__main__":
    main()
