#!/usr/bin/env python3
"""Simple loss curve plotting from training log (no GUI, saves plot periodically).

Usage:
    python plot_loss_simple.py [--interval SECONDS] [--output OUTPUT.png]
"""

import re
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Try to import scipy for Gaussian filter (smoother than moving average)
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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

def plot_loss_curve(steps, losses, output_path, smooth_window=None):
    """Create and save loss curve plot with ultra-smooth line."""
    if len(steps) == 0:
        print("No data to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Calculate optimal smoothing for ultra-smooth line (no bumps)
    if smooth_window is None:
        # Use very large window: ~5-10% of data points for maximum smoothness
        # Gaussian filter sigma is roughly window/3 for equivalent smoothness
        smooth_window = max(1000, len(losses) // 20)
    
    if len(losses) >= 50:
        if SCIPY_AVAILABLE:
            # Use Gaussian filter for ultra-smooth results (no bumps)
            # Sigma controls smoothness: larger = smoother
            sigma = smooth_window / 3.0  # Gaussian filter parameter
            smoothed_losses = ndimage.gaussian_filter1d(losses, sigma=sigma)
            smooth_label = f'Smoothed Loss (σ={sigma:.0f})'
        else:
            # Fallback: Very large moving average for smoothness
            # Use larger window for ultra-smooth line
            window = smooth_window
            smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='same')
            smooth_label = f'Smoothed Loss (window={window})'
        
        # Plot raw loss curve (very faint, optional)
        plt.plot(steps, losses, 'b-', linewidth=0.3, alpha=0.15, label='Raw Loss')
        
        # Plot ultra-smooth line as the main line
        plt.plot(steps, smoothed_losses, 'r-', linewidth=3.5, alpha=0.95, 
                label=smooth_label)
    else:
        # If not enough data, just plot raw
        plt.plot(steps, losses, 'r-', linewidth=3.5, alpha=0.95, label='Training Loss')
        smoothed_losses = losses
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss Curve (Step {steps[-1]}/100000)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set reasonable y-axis limits based on smoothed data
    y_min = min(smoothed_losses) * 0.95
    y_max = max(smoothed_losses) * 1.05
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if len(losses) >= 50 and SCIPY_AVAILABLE:
        sigma = smooth_window / 3.0
        print(f"Plot saved: {output_path} (Step {steps[-1]}, Loss {losses[-1]:.4f}, σ={sigma:.0f})")
    else:
        print(f"Plot saved: {output_path} (Step {steps[-1]}, Loss {losses[-1]:.4f}, window={smooth_window})")

def main():
    parser = argparse.ArgumentParser(description='Plot training loss from log file')
    parser.add_argument('--log', type=str, default='training.log',
                        help='Path to training log file')
    parser.add_argument('--output', type=str, default='loss_curve.png',
                        help='Output plot file')
    parser.add_argument('--interval', type=int, default=10,
                        help='Update interval in seconds (0 = run once and exit)')
    parser.add_argument('--smooth', type=int, default=None,
                        help='Smoothing window size (default: auto-calculated for smooth line)')
    args = parser.parse_args()
    
    log_path = Path(args.log)
    output_path = Path(args.output)
    
    if args.interval == 0:
        # Run once and exit
        steps, losses = parse_log_file(log_path)
        plot_loss_curve(steps, losses, output_path, smooth_window=args.smooth)
    else:
        # Continuous monitoring
        print(f"Monitoring: {log_path}")
        print(f"Output: {output_path}")
        print(f"Update interval: {args.interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                steps, losses = parse_log_file(log_path)
                plot_loss_curve(steps, losses, output_path, smooth_window=args.smooth)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped monitoring")

if __name__ == "__main__":
    main()
