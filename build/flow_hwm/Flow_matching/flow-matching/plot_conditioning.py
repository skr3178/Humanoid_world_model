import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from utils import generate_dynamic_conditioning

# Set parameters
total_steps = 1500
steps = np.arange(total_steps)

# Generate conditioning for each step
x_values = []
y_values = []

for step in steps:
    conditioning = generate_dynamic_conditioning(step, total_steps)
    x_values.append(conditioning[0].item())
    y_values.append(conditioning[1].item())

# Create the plot
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot x component (linear)
axes[0].plot(steps, x_values, 'b-', linewidth=2)
axes[0].set_title('Dynamic Conditioning: X Component', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Training Step', fontsize=12)
axes[0].set_ylabel('X Value', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-1.2, 1.2)

# Plot y component (sinusoidal)
axes[1].plot(steps, y_values, 'r-', linewidth=2)
axes[1].set_title('Dynamic Conditioning: Y Component', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Training Step', fontsize=12)
axes[1].set_ylabel('Y Value', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.6, 0.6)

plt.tight_layout()
plt.savefig('images/conditioning_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved to: images/conditioning_plot.png")

# Also create a 2D trajectory plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_values, y_values, 'o-', markersize=1, linewidth=0.5, alpha=0.6, color='purple')
ax.set_title('Dynamic Conditioning: Trajectory in (X, Y) Space', fontsize=14, fontweight='bold')
ax.set_xlabel('X Component', fontsize=12)
ax.set_ylabel('Y Component', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('images/conditioning_trajectory.png', dpi=300, bbox_inches='tight')
plt.close()
print("Trajectory plot saved to: images/conditioning_trajectory.png")

