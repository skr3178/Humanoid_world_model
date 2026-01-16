"""Plot the cosine schedule function from collator.py"""

import math
import numpy as np
import matplotlib.pyplot as plt

def cosine_schedule(r: float) -> float:
    """MaskGIT cosine schedule for masking threshold.
    
    γ(r) = cos(r * π/2)
    - r=0 → γ=1.0 (mask ~100% of tokens)
    - r=1 → γ=0.0 (mask ~0% of tokens)
    """
    return math.cos(r * math.pi / 2)

def cosine_schedule_opposite_phase(r: float) -> float:
    """Opposite phase cosine schedule for masking threshold.
    
    γ(r) = sin(r * π/2) = 1 - cos(r * π/2)
    - r=0 → γ=0.0 (mask ~0% of tokens)
    - r=1 → γ=1.0 (mask ~100% of tokens)
    
    This is the inverse behavior: starts with low masking, increases to high masking.
    """
    return math.sin(r * math.pi / 2)

# Generate r values from 0 to 1
r_values = np.linspace(0, 1, 1000)
gamma_values = np.array([cosine_schedule(r) for r in r_values])
gamma_values_opposite = np.array([cosine_schedule_opposite_phase(r) for r in r_values])

# Create the plot
plt.figure(figsize=(12, 7))
plt.plot(r_values, gamma_values, 'b-', linewidth=2.5, label='Standard: γ(r) = cos(r · π/2)', alpha=0.8)
plt.plot(r_values, gamma_values_opposite, 'r--', linewidth=2.5, label='Opposite Phase: γ(r) = sin(r · π/2)', alpha=0.8)
plt.xlabel('r (random value from U(0,1))', fontsize=12)
plt.ylabel('γ(r) (masking threshold)', fontsize=12)
plt.title('Cosine Schedule Comparison: Standard vs Opposite Phase', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc='upper right')

# Add annotations for standard schedule
plt.plot(0, 1, 'bo', markersize=8)
plt.plot(1, 0, 'go', markersize=8)
plt.annotate('Standard: r=0 → γ=1.0\n(mask ~100%)', 
             xy=(0, 1), xytext=(0.15, 0.85),
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
             fontsize=9, color='blue')
plt.annotate('Standard: r=1 → γ=0.0\n(mask ~0%)', 
             xy=(1, 0), xytext=(0.75, 0.15),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
             fontsize=9, color='green')

# Add annotations for opposite phase schedule
plt.plot(0, 0, 'ro', markersize=8)
plt.plot(1, 1, 'mo', markersize=8)
plt.annotate('Opposite: r=0 → γ=0.0\n(mask ~0%)', 
             xy=(0, 0), xytext=(0.15, 0.15),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=9, color='red')
plt.annotate('Opposite: r=1 → γ=1.0\n(mask ~100%)', 
             xy=(1, 1), xytext=(0.75, 0.85),
             arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5),
             fontsize=9, color='magenta')

# Add horizontal line at y=0.5 for reference
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
plt.text(0.5, 0.52, 'γ=0.5', fontsize=9, color='gray', ha='center')

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.tight_layout()

# Save the plot
output_path = 'cosine_schedule_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
print("\nSchedule Comparison:")
print("  Standard (cos):    r=0 → γ=1.0 (high mask), r=1 → γ=0.0 (low mask)")
print("  Opposite (sin):   r=0 → γ=0.0 (low mask),  r=1 → γ=1.0 (high mask)")

# Display the plot
plt.show()
