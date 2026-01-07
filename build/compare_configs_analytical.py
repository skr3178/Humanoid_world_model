"""Analytically compare parameter counts between original and 12GB configs.

This script calculates parameters without instantiating the model.
"""

from masked_hwm.config import MaskedHWMConfig
from masked_hwm.config_12gb import MaskedHWM12GBConfig


def count_transformer_params(config):
    """Count transformer parameters accounting for parameter sharing.
    
    Parameter sharing:
    - Layers < shared_layers_start: No sharing (separate params per stream)
      - 4 streams: v_p, v_f, a_p, a_f
      - Each stream has: spatial_attn (video only), temporal_attn (shared), mlp
    - Layers >= shared_layers_start: Modality sharing
      - Video streams (v_p, v_f) share spatial_attn and MLP
      - Action streams (a_p, a_f) share MLP
      - Temporal attention is always shared across all streams
    
    Per layer components:
    - Spatial attention (video only):
      - QKV: d_model * (3 * d_model) = 3 * d_model^2
      - Proj: d_model * d_model = d_model^2
      - Total: 4 * d_model^2 (if qkv_bias=False, proj_bias=True)
    - Temporal attention (shared):
      - QKV: d_model * (3 * d_model) = 3 * d_model^2
      - Proj: d_model * d_model = d_model^2
      - Total: 4 * d_model^2
      - Optional stream_type_emb: 4 * d_model (if use_stream_type_emb=True)
    - MLP:
      - FC1: d_model * mlp_hidden
      - FC2: mlp_hidden * d_model
      - Total: 2 * d_model * mlp_hidden (if bias=True)
    - Layer norms: 3 per layer (spatial, temporal, mlp)
      - Each: 2 * d_model (weight + bias)
      - Total: 6 * d_model
    
    RoPE parameters are negligible (just precomputed cos/sin tables).
    """
    d_model = config.d_model
    num_heads = config.num_heads
    mlp_hidden = config.mlp_hidden
    num_layers = config.num_layers
    shared_layers_start = config.shared_layers_start
    
    # Attention parameters (no bias in QKV, bias in proj)
    qkv_params = 3 * d_model * d_model  # QKV projection
    proj_params = d_model * d_model + (d_model if config.proj_bias else 0)  # Output projection
    spatial_attn_params = qkv_params + proj_params
    temporal_attn_params = qkv_params + proj_params
    
    # Optional stream type embeddings
    if config.use_stream_type_emb:
        temporal_attn_params += 4 * d_model
    
    # MLP parameters (with bias)
    mlp_params = 2 * d_model * mlp_hidden + mlp_hidden + d_model  # FC1 + FC2 with biases
    
    # Layer norm parameters (3 per layer: spatial, temporal, mlp)
    ln_params = 3 * (2 * d_model)  # weight + bias per norm
    
    total_params = 0
    
    # Unshared layers (layers < shared_layers_start)
    num_unshared = min(shared_layers_start, num_layers)
    for _ in range(num_unshared):
        # Video streams (v_p, v_f): each has spatial_attn + mlp
        # Action streams (a_p, a_f): each has mlp (no spatial_attn)
        # Temporal attention is shared
        layer_params = (
            2 * spatial_attn_params +  # v_p, v_f spatial attention
            temporal_attn_params +      # shared temporal attention
            4 * mlp_params +            # separate MLP for each stream
            ln_params                   # layer norms
        )
        total_params += layer_params
    
    # Shared layers (layers >= shared_layers_start)
    num_shared = max(0, num_layers - shared_layers_start)
    for _ in range(num_shared):
        # Video streams share spatial_attn and MLP
        # Action streams share MLP
        # Temporal attention is shared
        layer_params = (
            spatial_attn_params +       # shared spatial attention for video
            temporal_attn_params +      # shared temporal attention
            2 * mlp_params +            # video_mlp + action_mlp
            ln_params                   # layer norms
        )
        total_params += layer_params
    
    # Final layer norm (4 streams)
    total_params += 4 * (2 * d_model)
    
    return total_params


def count_embedding_params(config):
    """Count embedding parameters."""
    # Video token embeddings: num_factored_vocabs * (vocab_size + 1) * d_model
    # +1 for mask token
    video_embed_params = config.num_factored_vocabs * (config.vocab_size + 1) * config.d_model
    
    # Action embeddings: Linear(25, d_model) + Linear(d_model, d_model) with biases
    action_embed_params = (
        config.action_dim * config.d_model + config.d_model +  # FC1
        config.d_model * config.d_model + config.d_model        # FC2
    )
    
    return video_embed_params, action_embed_params


def count_position_embedding_params(config):
    """Count position embedding parameters."""
    # Calculate number of clips
    num_past_clips = max(1, config.num_past_frames // config.frames_per_clip + 1)
    num_future_clips = max(1, config.num_future_frames // config.frames_per_clip + 1)
    total_clips = num_past_clips + num_future_clips
    
    # Video position embeddings: (1, total_clips, H, W, d_model)
    video_pos_params = total_clips * config.spatial_size * config.spatial_size * config.d_model
    
    # Action position embeddings: (1, total_clips, d_model)
    action_pos_params = total_clips * config.d_model
    
    return video_pos_params, action_pos_params


def count_output_proj_params(config):
    """Count output projection parameters."""
    # Separate head for each factor: Linear(d_model, vocab_size)
    output_proj_params = config.num_factored_vocabs * (
        config.d_model * config.vocab_size + config.vocab_size  # weight + bias
    )
    return output_proj_params


def count_all_params(config):
    """Count all model parameters."""
    transformer_params = count_transformer_params(config)
    video_embed_params, action_embed_params = count_embedding_params(config)
    video_pos_params, action_pos_params = count_position_embedding_params(config)
    output_proj_params = count_output_proj_params(config)
    
    total_params = (
        transformer_params +
        video_embed_params +
        action_embed_params +
        video_pos_params +
        action_pos_params +
        output_proj_params
    )
    
    return {
        'total': total_params,
        'transformer': transformer_params,
        'video_embed': video_embed_params,
        'action_embed': action_embed_params,
        'video_pos': video_pos_params,
        'action_pos': action_pos_params,
        'output_proj': output_proj_params,
    }


if __name__ == "__main__":
    original_config = MaskedHWMConfig()
    reduced_config = MaskedHWM12GBConfig()
    
    print(f"\n{'='*70}")
    print("ORIGINAL Configuration")
    print(f"{'='*70}")
    print(f"\nModel Architecture:")
    print(f"  - Layers: {original_config.num_layers}")
    print(f"  - Heads: {original_config.num_heads}")
    print(f"  - d_model: {original_config.d_model}")
    print(f"  - MLP hidden: {original_config.mlp_hidden}")
    print(f"  - Shared layers start: {original_config.shared_layers_start}")
    
    original_params = count_all_params(original_config)
    
    print(f"\nParameter Breakdown:")
    print(f"  - Video token embeddings: {original_params['video_embed']:,}")
    print(f"  - Action embeddings: {original_params['action_embed']:,}")
    print(f"  - Transformer: {original_params['transformer']:,}")
    print(f"  - Output projections: {original_params['output_proj']:,}")
    print(f"  - Video position embeddings: {original_params['video_pos']:,}")
    print(f"  - Action position embeddings: {original_params['action_pos']:,}")
    print(f"\n{'─'*70}")
    print(f"TOTAL PARAMETERS: {original_params['total']:,}")
    print(f"Size (MB, float32): {original_params['total'] * 4 / (1024**2):.2f}")
    print(f"Size (MB, float16): {original_params['total'] * 2 / (1024**2):.2f}")
    
    print(f"\n{'='*70}")
    print("12GB Configuration")
    print(f"{'='*70}")
    print(f"\nModel Architecture:")
    print(f"  - Layers: {reduced_config.num_layers}")
    print(f"  - Heads: {reduced_config.num_heads}")
    print(f"  - d_model: {reduced_config.d_model}")
    print(f"  - MLP hidden: {reduced_config.mlp_hidden}")
    print(f"  - Shared layers start: {reduced_config.shared_layers_start}")
    
    reduced_params = count_all_params(reduced_config)
    
    print(f"\nParameter Breakdown:")
    print(f"  - Video token embeddings: {reduced_params['video_embed']:,}")
    print(f"  - Action embeddings: {reduced_params['action_embed']:,}")
    print(f"  - Transformer: {reduced_params['transformer']:,}")
    print(f"  - Output projections: {reduced_params['output_proj']:,}")
    print(f"  - Video position embeddings: {reduced_params['video_pos']:,}")
    print(f"  - Action position embeddings: {reduced_params['action_pos']:,}")
    print(f"\n{'─'*70}")
    print(f"TOTAL PARAMETERS: {reduced_params['total']:,}")
    print(f"Size (MB, float32): {reduced_params['total'] * 4 / (1024**2):.2f}")
    print(f"Size (MB, float16): {reduced_params['total'] * 2 / (1024**2):.2f}")
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"\nTotal Parameters:")
    print(f"  Original:  {original_params['total']:,}")
    print(f"  12GB:      {reduced_params['total']:,}")
    print(f"  Reduction: {original_params['total'] - reduced_params['total']:,}")
    ratio = reduced_params['total'] / original_params['total']
    print(f"  Ratio:     {ratio:.4f} ({ratio * 100:.2f}%)")
    
    print(f"\nComponent-wise Comparison:")
    print(f"{'Component':<30} {'Original':>15} {'12GB':>15} {'Ratio':>10}")
    print(f"{'-'*70}")
    for key in ['video_embed', 'action_embed', 'transformer', 'output_proj', 'video_pos', 'action_pos']:
        orig = original_params[key]
        red = reduced_params[key]
        ratio = red / orig if orig > 0 else 0.0
        print(f"{key:<30} {orig:>15,} {red:>15,} {ratio:>9.4f}")
    
    print(f"\n{'='*70}\n")
