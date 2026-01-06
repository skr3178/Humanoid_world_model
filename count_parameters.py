"""Count total number of parameters in Masked-HWM model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from masked_hwm.config import MaskedHWMConfig
from masked_hwm.config_test import MaskedHWMTestConfig
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
    
    return total_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument("--config", choices=["default", "test", "both"], default="both",
                       help="Which config to use (default: both)")
    args = parser.parse_args()
    
    if args.config in ["default", "both"]:
        default_config = MaskedHWMConfig()
        count_parameters(default_config, "DEFAULT (Full)")
    
    if args.config in ["test", "both"]:
        test_config = MaskedHWMTestConfig()
        count_parameters(test_config, "TEST (Reduced)")
