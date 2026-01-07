#!/usr/bin/env python3
"""Inspect model architecture and input/output shapes.

This script helps you understand:
- Model architecture and parameters
- Input shapes expected by the model
- Output shapes from the model
- Memory requirements
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from masked_hwm.config_minimal import MaskedHWMMinimalConfig
from masked_hwm.config_12gb import MaskedHWM12GBConfig
from masked_hwm.model import MaskedHWM


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def inspect_config(config, config_name):
    """Inspect a model configuration."""
    print(f"\n{'='*70}")
    print(f"{config_name} Configuration")
    print(f"{'='*70}")
    
    # Calculate sequence dimensions
    num_past_clips = config.num_past_clips
    num_future_clips = config.num_future_clips
    total_clips = num_past_clips + num_future_clips
    spatial_tokens = config.spatial_size * config.spatial_size  # 1024
    seq_length = total_clips * spatial_tokens
    
    print(f"\nModel Architecture:")
    print(f"  - Layers: {config.num_layers}")
    print(f"  - Heads: {config.num_heads}")
    print(f"  - d_model: {config.d_model}")
    print(f"  - MLP hidden: {config.mlp_hidden}")
    print(f"  - Head dimension: {config.d_model // config.num_heads}")
    
    print(f"\nSequence Configuration:")
    print(f"  - Past clips: {num_past_clips}")
    print(f"  - Future clips: {num_future_clips}")
    print(f"  - Total clips: {total_clips}")
    print(f"  - Spatial tokens per clip: {spatial_tokens} (32×32)")
    print(f"  - Total sequence length: {seq_length} tokens")
    print(f"  - Past frames: {config.num_past_frames}")
    print(f"  - Future frames: {config.num_future_frames}")
    
    print(f"\nOther Settings:")
    print(f"  - vocab_size: {config.vocab_size} (per factor)")
    print(f"  - num_factors: {config.num_factored_vocabs}")
    print(f"  - action_dim: {config.action_dim}")
    
    # Create model
    print(f"\n{'─'*70}")
    print("Creating model...")
    model = MaskedHWM(config)
    total_params = count_parameters(model)
    param_size = total_params * 2 / (1024**3)  # bf16 = 2 bytes
    
    print(f"\nModel Parameters:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size (bf16): {param_size:.2f} GB")
    
    # Estimate memory for batch_size=1
    batch_size = 1
    print(f"\n{'─'*70}")
    print(f"Memory Estimate (batch_size={batch_size}, training):")
    print(f"{'─'*70}")
    
    # Model + optimizer + gradients
    optimizer_memory = total_params * 4 * 2 / (1024**3)  # AdamW: 2x fp32
    gradients_memory = total_params * 2 / (1024**3)  # bf16
    print(f"  1. Model params (bf16):     {param_size:.2f} GB")
    print(f"  2. Optimizer states (fp32): {optimizer_memory:.2f} GB")
    print(f"  3. Gradients (bf16):        {gradients_memory:.2f} GB")
    
    # Activations
    head_dim = config.d_model // config.num_heads
    # Attention scores: (B, num_heads, seq_len, seq_len)
    attn_scores = batch_size * config.num_heads * seq_length * seq_length * 2 / (1024**3)
    # Q, K, V: (B, num_heads, seq_len, head_dim) × 3
    qkv = batch_size * config.num_heads * seq_length * head_dim * 3 * 2 / (1024**3)
    # MLP: (B, seq_len, d_model + mlp_hidden)
    mlp = batch_size * seq_length * (config.d_model + config.mlp_hidden) * 2 / (1024**3)
    activation_per_layer = attn_scores + qkv + mlp
    total_activations = activation_per_layer * config.num_layers
    
    print(f"  4. Activations (all layers): {total_activations:.2f} GB")
    print(f"     (per layer: ~{activation_per_layer:.3f} GB)")
    
    # Cross-entropy logits (THE BOTTLENECK!)
    logits_per_factor = batch_size * seq_length * config.vocab_size * 2 / (1024**3)
    logits_total = logits_per_factor * config.num_factored_vocabs
    print(f"  5. Cross-entropy logits:    {logits_total:.2f} GB")
    print(f"     (per factor: {logits_per_factor:.2f} GB)")
    
    overhead = 0.5  # GB
    total_memory = param_size + optimizer_memory + gradients_memory + total_activations + logits_total + overhead
    print(f"  6. CUDA overhead:           {overhead:.2f} GB")
    print(f"\n  TOTAL ESTIMATED:           {total_memory:.2f} GB")
    
    # Test forward pass
    print(f"\n{'─'*70}")
    print("Testing Forward Pass (batch_size=1):")
    print(f"{'─'*70}")
    
    # Create dummy inputs
    B = 1
    video_past = torch.randint(0, config.vocab_size, (B, config.num_factored_vocabs, num_past_clips, config.spatial_size, config.spatial_size))
    video_future = torch.randint(0, config.vocab_size, (B, config.num_factored_vocabs, num_future_clips, config.spatial_size, config.spatial_size))
    actions_past = torch.randn(B, config.num_past_frames, config.action_dim)
    actions_future = torch.randn(B, config.num_future_frames, config.action_dim)
    
    print(f"\nInput Shapes:")
    print(f"  video_past:      {video_past.shape} (B, num_factors, T_p_clips, H, W)")
    print(f"  video_future:    {video_future.shape} (B, num_factors, T_f_clips, H, W)")
    print(f"  actions_past:    {actions_past.shape} (B, T_p_frames, action_dim)")
    print(f"  actions_future:  {actions_future.shape} (B, T_f_frames, action_dim)")
    
    model.eval()
    with torch.no_grad():
        logits = model(
            v_p_tokens=video_past,
            v_f_tokens=video_future,
            a_p=actions_past,
            a_f=actions_future,
        )
    
    print(f"\nOutput Shape:")
    print(f"  logits:          {logits.shape} (num_factors, B, T_f_clips, H, W, vocab_size)")
    print(f"  Expected:        ({config.num_factored_vocabs}, {B}, {num_future_clips}, {config.spatial_size}, {config.spatial_size}, {config.vocab_size})")
    
    # Test loss computation
    targets = torch.randint(0, config.vocab_size, (B, config.num_factored_vocabs, num_future_clips, config.spatial_size, config.spatial_size))
    mask = torch.ones(B, num_future_clips, config.spatial_size, config.spatial_size, dtype=torch.bool)
    
    loss = model.compute_loss(logits, targets, mask)
    print(f"\nLoss Computation:")
    print(f"  targets shape:   {targets.shape} (B, num_factors, T_f_clips, H, W)")
    print(f"  mask shape:      {mask.shape} (B, T_f_clips, H, W)")
    print(f"  loss value:      {loss.item():.4f}")
    print(f"  Expected baseline (random): {torch.log(torch.tensor(float(config.vocab_size))).item():.4f}")
    
    print(f"\n{'='*70}")
    print("✓ Model inspection complete!")
    print(f"{'='*70}\n")
    
    return model, total_memory


if __name__ == "__main__":
    print("Model Architecture Inspector")
    print("=" * 70)
    
    # Inspect minimal config
    minimal_config = MaskedHWMMinimalConfig()
    minimal_model, minimal_memory = inspect_config(minimal_config, "MINIMAL (Testing)")
    
    # Inspect 12GB config
    config_12gb = MaskedHWM12GBConfig()
    config_12gb_model, config_12gb_memory = inspect_config(config_12gb, "12GB (Production)")
    
    print("\n" + "=" * 70)
    print("Comparison:")
    print("=" * 70)
    print(f"Minimal config memory:  {minimal_memory:.2f} GB")
    print(f"12GB config memory:     {config_12gb_memory:.2f} GB")
    print(f"Difference:            {config_12gb_memory - minimal_memory:.2f} GB")
    print("\nRecommendation:")
    if minimal_memory < 6:
        print("  ✓ Minimal config should fit in 4-6GB GPU")
    else:
        print("  ⚠️  Minimal config may need further reduction")
    
    if config_12gb_memory < 12:
        print("  ✓ 12GB config should fit in 12GB GPU")
    else:
        print("  ⚠️  12GB config may not fit - reduce batch_size or model size")
