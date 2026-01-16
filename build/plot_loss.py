#!/usr/bin/env python3
"""Real-time loss curve plotting from training log.

Reads training.log and plots loss vs step in real-time without stopping training.
"""

import re
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from pathlib import Path

# Configuration
LOG_FILE = Path(__file__).parent / "training.log"
OUTPUT_PLOT = Path(__file__).parent / "loss_curve.png"
UPDATE_INTERVAL = 2  # seconds between plot updates

# Regex to parse log lines
LOSS_PATTERN = re.compile(r'Loss:\s+([\d.]+).*?(\d+)/100000')

def parse_log_file(log_path):
    """Parse log file and extract (step, loss) pairs."""
    steps = []
    losses = []
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = LOSS_PATTERN.search(line)
                if match:
                    loss = float(match.group(1))
                    step = int(match.group(2))
                    steps.append(step)
                    losses.append(loss)
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
        return [], []
    
    return steps, losses

def update_plot(frame):
    """Update the plot with latest data from log file."""
    steps, losses = parse_log_file(LOG_FILE)
    
    if len(steps) == 0:
        return
    
    # Clear and replot
    plt.clf()
    
    # Plot loss curve
    plt.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.7, label='Training Loss')
    
    # Add moving average for smoother view
    if len(losses) > 50:
        window = min(100, len(losses) // 10)
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        moving_steps = steps[window-1:]
        plt.plot(moving_steps, moving_avg, 'r-', linewidth=2, alpha=0.8, label=f'Moving Avg ({window})')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss Curve (Step {steps[-1] if steps else 0}/100000)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set reasonable y-axis limits
    if len(losses) > 0:
        y_min = min(losses) * 0.95
        y_max = max(losses) * 1.05
        plt.ylim(y_min, y_max)
    
    # Save current plot
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    
    # Print status
    if len(losses) > 0:
        print(f"\rLatest: Step {steps[-1]}, Loss {losses[-1]:.4f} (saved to {OUTPUT_PLOT})", end='', flush=True)

def main():
    """Main function to run real-time plotting."""
    print(f"Monitoring log file: {LOG_FILE}")
    print(f"Plot will be saved to: {OUTPUT_PLOT}")
    print(f"Update interval: {UPDATE_INTERVAL} seconds")
    print("Press Ctrl+C to stop\n")
    
    # Create initial plot
    fig = plt.figure(figsize=(12, 6))
    
    # Animate plot updates
    ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL * 1000, blit=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n\nStopped monitoring. Final plot saved to:", OUTPUT_PLOT)

if __name__ == "__main__":
    main()
