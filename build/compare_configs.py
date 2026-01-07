"""Compare parameter counts between original and 12GB configs."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from masked_hwm.config import MaskedHWMConfig
from masked_hwm.config_12gb import MaskedHWM12GBConfig
from masked_hwm.model import MaskedHWM


def count_parameters(config, config_name):
    """Count and display model parameters."""
    print(f"\n{'='*70}")
    print(f"{config_name} Configuration")
    print(f"{'='*70}")
    
    # Create model
    model = MaskedHWM(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by component
    video_embed_params = sum(p.numel() for p in model.video_token_embed.parameters())
    action_embed_params = sum(p.numel() for p in model.action_embedding.parameters())
    transformer_params = sum(p.numel() for p in model.transformer.parameters())
    output_proj_params = sum(p.numel() for p in model.output_projs.parameters())
    video_pos_params = model.video_pos_embed.numel()
    action_pos_params = model.action_pos_embed.numel()
    
    print(f"\nModel Architecture:")
    print(f"  - Layers: {config.num_layers}")
    print(f"  - Heads: {config.num_heads}")
    print(f"  - d_model: {config.d_model}")
    print(f"  - MLP hidden: {config.mlp_hidden}")
    print(f"  - Vocab size (per factor): {config.vocab_size:,}")
    print(f"  - Num factors: {config.num_factored_vocabs}")
    print(f"  - Spatial size: {config.spatial_size}×{config.spatial_size}")
    print(f"  - Action dim: {config.action_dim}")
    print(f"  - Shared layers start: {config.shared_layers_start}")
    
    print(f"\nParameter Breakdown:")
    print(f"  - Video token embeddings: {video_embed_params:,}")
    print(f"  - Action embeddings: {action_embed_params:,}")
    print(f"  - Transformer: {transformer_params:,}")
    print(f"  - Output projections: {output_proj_params:,}")
    print(f"  - Video position embeddings: {video_pos_params:,}")
    print(f"  - Action position embeddings: {action_pos_params:,}")
    
    print(f"\n{'─'*70}")
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Size (MB, float32): {total_params * 4 / (1024**2):.2f}")
    print(f"Size (MB, float16): {total_params * 2 / (1024**2):.2f}")
    print(f"{'='*70}\n")
    
    return {
        'total': total_params,
        'video_embed': video_embed_params,
        'action_embed': action_embed_params,
        'transformer': transformer_params,
        'output_proj': output_proj_params,
        'video_pos': video_pos_params,
        'action_pos': action_pos_params,
    }


if __name__ == "__main__":
    # Count parameters for both configs
    original_config = MaskedHWMConfig()
    reduced_config = MaskedHWM12GBConfig()
    
    original_params = count_parameters(original_config, "ORIGINAL (Full Model)")
    reduced_params = count_parameters(reduced_config, "12GB (Reduced Model)")
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"\nTotal Parameters:")
    print(f"  Original:  {original_params['total']:,}")
    print(f"  12GB:      {reduced_params['total']:,}")
    print(f"  Reduction: {original_params['total'] - reduced_params['total']:,}")
    print(f"  Ratio:     {reduced_params['total'] / original_params['total']:.4f} ({reduced_params['total'] / original_params['total'] * 100:.2f}%)")
    
    print(f"\nComponent-wise Comparison:")
    print(f"{'Component':<30} {'Original':>15} {'12GB':>15} {'Ratio':>10}")
    print(f"{'-'*70}")
    for key in ['video_embed', 'action_embed', 'transformer', 'output_proj', 'video_pos', 'action_pos']:
        orig = original_params[key]
        red = reduced_params[key]
        ratio = red / orig if orig > 0 else 0.0
        print(f"{key:<30} {orig:>15,} {red:>15,} {ratio:>9.4f}")
    
    print(f"\n{'='*70}\n")
