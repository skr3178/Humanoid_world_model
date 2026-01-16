"""Visualize RoPE (Rotary Position Embedding) in 1D vs 2D space."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from build.masked_hwm.rope import RoPE1D, RoPE2D

# Set up parameters
head_dim = 64
max_seq_len = 100
max_h, max_w = 20, 20
base = 10000.0

# Initialize RoPE modules
rope_1d = RoPE1D(head_dim=head_dim, max_seq_len=max_seq_len, base=base)
rope_2d = RoPE2D(head_dim=head_dim, max_h=max_h, max_w=max_w, base=base)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# ========== 1D RoPE Visualization ==========
# Compute frequencies for 1D
positions_1d = torch.arange(max_seq_len, dtype=torch.float32)
freqs_1d = torch.outer(positions_1d, rope_1d.inv_freq)  # (seq_len, head_dim // 2)

# Plot 1: Frequency patterns across sequence
ax1 = plt.subplot(2, 3, 1)
# Show first few frequency dimensions
for dim_idx in range(min(4, freqs_1d.shape[1])):
    ax1.plot(positions_1d.numpy(), freqs_1d[:, dim_idx].numpy(), 
             label=f'Freq dim {dim_idx}', alpha=0.7)
ax1.set_xlabel('Position in Sequence')
ax1.set_ylabel('Frequency (radians)')
ax1.set_title('1D RoPE: Frequency vs Position')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cosine/Sine values for first frequency dimension
ax2 = plt.subplot(2, 3, 2)
cos_vals = freqs_1d[:, 0].cos()
sin_vals = freqs_1d[:, 0].sin()
ax2.plot(positions_1d.numpy(), cos_vals.numpy(), label='cos(θ)', linewidth=2)
ax2.plot(positions_1d.numpy(), sin_vals.numpy(), label='sin(θ)', linewidth=2)
ax2.set_xlabel('Position in Sequence')
ax2.set_ylabel('Value')
ax2.set_title('1D RoPE: Rotation Components (first dim)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: 2D rotation visualization for 1D RoPE
ax3 = plt.subplot(2, 3, 3)
# Show how a unit vector rotates at different positions
unit_vec = torch.tensor([1.0, 0.0])  # Start with unit vector along x-axis
positions_to_show = [0, 10, 20, 30, 40, 50]
colors = plt.cm.viridis(np.linspace(0, 1, len(positions_to_show)))

for i, pos in enumerate(positions_to_show):
    if pos < max_seq_len:
        theta = freqs_1d[pos, 0].item()
        # Rotate the vector
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotated_vec = np.array([
            unit_vec[0] * cos_theta - unit_vec[1] * sin_theta,
            unit_vec[0] * sin_theta + unit_vec[1] * cos_theta
        ])
        ax3.arrow(0, 0, rotated_vec[0], rotated_vec[1], 
                 head_width=0.05, head_length=0.05, 
                 fc=colors[i], ec=colors[i], 
                 length_includes_head=True, linewidth=2,
                 label=f'Pos {pos}')
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_xlabel('Real component')
ax3.set_ylabel('Imaginary component')
ax3.set_title('1D RoPE: Vector Rotation at Different Positions')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

# ========== 2D RoPE Visualization ==========
# Compute frequencies for 2D
positions_h = torch.arange(max_h, dtype=torch.float32)
positions_w = torch.arange(max_w, dtype=torch.float32)
freqs_h = torch.outer(positions_h, rope_2d.inv_freq)  # (H, head_dim // 4)
freqs_w = torch.outer(positions_w, rope_2d.inv_freq)  # (W, head_dim // 4)

# Plot 4: Frequency heatmap for height dimension
ax4 = plt.subplot(2, 3, 4)
# Show first frequency dimension across height
freq_h_2d = freqs_h[:, 0].unsqueeze(1).expand(-1, max_w).numpy()
im4 = ax4.imshow(freq_h_2d, cmap='viridis', aspect='auto')
ax4.set_xlabel('Width')
ax4.set_ylabel('Height')
ax4.set_title('2D RoPE: Height Frequency (first dim)')
plt.colorbar(im4, ax=ax4)

# Plot 5: Frequency heatmap for width dimension
ax5 = plt.subplot(2, 3, 5)
# Show first frequency dimension across width
freq_w_2d = freqs_w[:, 0].unsqueeze(0).expand(max_h, -1).numpy()
im5 = ax5.imshow(freq_w_2d, cmap='plasma', aspect='auto')
ax5.set_xlabel('Width')
ax5.set_ylabel('Height')
ax5.set_title('2D RoPE: Width Frequency (first dim)')
plt.colorbar(im5, ax=ax5)

# Plot 6: Combined 2D rotation visualization
ax6 = plt.subplot(2, 3, 6)
# Show rotation angles at different positions in 2D space
theta_h = freqs_h[:, 0].numpy()
theta_w = freqs_w[:, 0].numpy()

# Create a grid showing rotation magnitude
rotation_magnitude = np.zeros((max_h, max_w))
for h in range(max_h):
    for w in range(max_w):
        # Combine height and width rotations
        rot_h = np.sqrt(np.cos(theta_h[h])**2 + np.sin(theta_h[h])**2)
        rot_w = np.sqrt(np.cos(theta_w[w])**2 + np.sin(theta_w[w])**2)
        rotation_magnitude[h, w] = rot_h + rot_w

im6 = ax6.imshow(rotation_magnitude, cmap='coolwarm', aspect='auto')
ax6.set_xlabel('Width Position')
ax6.set_ylabel('Height Position')
ax6.set_title('2D RoPE: Combined Rotation Magnitude')
plt.colorbar(im6, ax=ax6)

plt.tight_layout()
plt.savefig('rope_1d_vs_2d_visualization.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'rope_1d_vs_2d_visualization.png'")
plt.show()
