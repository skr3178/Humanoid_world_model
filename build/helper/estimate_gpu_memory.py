"""Estimate GPU memory usage for Masked-HWM model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from masked_hwm.config import MaskedHWMConfig
from masked_hwm.config_test import MaskedHWMTestConfig
from masked_hwm.model import MaskedHWM


def estimate_memory(config, config_name, batch_size=128, training=True):
    """Estimate GPU memory usage."""
    print(f"\n{'='*70}")
    print(f"{config_name} Configuration - Memory Estimate")
    print(f"{'='*70}")
    
    # Create model
    model = MaskedHWM(config)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate sequence dimensions
    num_past_clips = max(1, config.num_past_frames // config.frames_per_clip + 1)
    num_future_clips = max(1, config.num_future_frames // config.frames_per_clip + 1)
    spatial_tokens = config.spatial_size * config.spatial_size  # 32×32 = 1024
    
    print(f"\nModel Configuration:")
    print(f"  - Parameters: {total_params:,}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Past clips: {num_past_clips}, Future clips: {num_future_clips}")
    print(f"  - Spatial tokens per clip: {spatial_tokens}")
    print(f"  - d_model: {config.d_model}")
    
    # Memory breakdown
    print(f"\n{'─'*70}")
    print("Memory Breakdown (Training with bf16):")
    print(f"{'─'*70}")
    
    # 1. Model parameters (bf16 = 2 bytes per param)
    model_memory = total_params * 2 / (1024**3)  # GB
    print(f"  1. Model parameters (bf16):     {model_memory:.2f} GB")
    
    if training:
        # 2. Optimizer states (AdamW: momentum + variance, both in fp32)
        # Each is same size as model, so 2x model size in fp32
        optimizer_memory = total_params * 4 * 2 / (1024**3)  # GB (fp32 = 4 bytes)
        print(f"  2. Optimizer states (AdamW):   {optimizer_memory:.2f} GB")
        
        # 3. Gradients (bf16)
        gradients_memory = total_params * 2 / (1024**3)  # GB
        print(f"  3. Gradients (bf16):           {gradients_memory:.2f} GB")
        
        # 4. Activations (rough estimate)
        # For each layer, we store activations
        # Input: batch_size × (past_clips + future_clips) × spatial_tokens × d_model
        total_clips = num_past_clips + num_future_clips
        seq_length = total_clips * spatial_tokens
        
        # Activation memory per layer (rough estimate)
        # Attention: batch_size × num_heads × seq_length × head_dim (for Q, K, V, output)
        # MLP: batch_size × seq_length × d_model (input) + batch_size × seq_length × mlp_hidden (hidden)
        head_dim = config.d_model // config.num_heads
        
        # Attention activations (Q, K, V, attention scores, output)
        attn_activation = batch_size * config.num_heads * seq_length * head_dim * 5  # Q, K, V, scores, output
        attn_activation_mb = attn_activation * 2 / (1024**2)  # bf16 = 2 bytes
        
        # MLP activations
        mlp_activation = batch_size * seq_length * (config.d_model + config.mlp_hidden)
        mlp_activation_mb = mlp_activation * 2 / (1024**2)
        
        # Per layer activation (rough)
        activation_per_layer_gb = (attn_activation_mb + mlp_activation_mb) / 1024  # GB
        
        # Total activations (all layers, with gradient checkpointing this could be reduced)
        # Without gradient checkpointing: store all activations
        # With gradient checkpointing: only store activations for current layer
        total_activation_memory = activation_per_layer_gb * config.num_layers  # GB
        print(f"  4. Activations (all layers):    {total_activation_memory:.2f} GB")
        print(f"     (per layer: ~{activation_per_layer_gb:.3f} GB)")
        
        # 5. Other overhead (CUDA, PyTorch, etc.)
        overhead = 0.5  # GB
        print(f"  5. CUDA/PyTorch overhead:      {overhead:.2f} GB")
        
        total_memory = model_memory + optimizer_memory + gradients_memory + total_activation_memory + overhead
        print(f"\n{'─'*70}")
        print(f"TOTAL ESTIMATED MEMORY:         {total_memory:.2f} GB")
        print(f"{'─'*70}")
        
        # Recommendations
        print(f"\nRecommendations for 12GB GPU:")
        if total_memory > 12:
            print(f"  ⚠️  Model may not fit with batch_size={batch_size}")
            print(f"  💡 Suggestions:")
            print(f"     - Reduce batch_size to 32-64")
            print(f"     - Use gradient accumulation (e.g., 4 steps with batch_size=32)")
            print(f"     - Enable gradient checkpointing (if available)")
            print(f"     - Use DeepSpeed ZeRO (if available)")
        else:
            print(f"  ✅ Model should fit with batch_size={batch_size}")
            if total_memory < 10:
                print(f"  💡 You may be able to increase batch_size slightly")
    else:
        # Inference only
        total_clips = num_past_clips + num_future_clips
        seq_length = total_clips * spatial_tokens
        
        # Activations for inference (only current layer, not all layers)
        head_dim = config.d_model // config.num_heads
        attn_activation = batch_size * config.num_heads * seq_length * head_dim * 5
        mlp_activation = batch_size * seq_length * (config.d_model + config.mlp_hidden)
        activation_memory = (attn_activation + mlp_activation) * 2 / (1024**3)  # GB (bf16)
        
        overhead = 0.3  # GB
        total_memory = model_memory + activation_memory + overhead
        
        print(f"  2. Activations (inference):     {activation_memory:.2f} GB")
        print(f"  3. CUDA/PyTorch overhead:      {overhead:.2f} GB")
        print(f"\n{'─'*70}")
        print(f"TOTAL ESTIMATED MEMORY:         {total_memory:.2f} GB")
        print(f"{'─'*70}")
        
        if total_memory > 12:
            print(f"\n  ⚠️  May not fit with batch_size={batch_size}")
            print(f"  💡 Reduce batch_size for inference")
        else:
            print(f"\n  ✅ Should fit comfortably for inference")
    
    return total_memory


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Estimate GPU memory usage")
    parser.add_argument("--config", choices=["default", "test", "both"], default="both",
                       help="Which config to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--inference", action="store_true", help="Estimate for inference only")
    args = parser.parse_args()
    
    if args.config in ["default", "both"]:
        default_config = MaskedHWMConfig()
        estimate_memory(default_config, "DEFAULT (Full)", batch_size=args.batch_size, training=not args.inference)
    
    if args.config in ["test", "both"]:
        test_config = MaskedHWMTestConfig()
        estimate_memory(test_config, "TEST (Reduced)", batch_size=args.batch_size, training=not args.inference)
