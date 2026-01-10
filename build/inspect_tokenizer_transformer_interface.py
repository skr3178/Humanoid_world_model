#!/usr/bin/env python3
"""Inspect the interface between Cosmos tokenizer output and transformer input.

This script traces the data flow:
1. Cosmos tokenizer output (from dataset)
2. Dataset loading
3. Collator processing (corruption + masking)
4. Model embedding
5. Transformer input
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add build directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import HumanoidWorldModelDataset
from data.collator import MaskedHWMCollator
from masked_hwm.config_minimal import MaskedHWMMinimalConfig
from masked_hwm.model import MaskedHWM

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_shape_info(name, tensor, expected=None):
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Range: [{tensor.min().item()}, {tensor.max().item()}]")
    if expected:
        print(f"  Expected: {expected}")
        if tensor.shape != expected:
            print(f"  ⚠️  Shape mismatch!")

def inspect_tokenizer_to_transformer():
    """Trace data flow from tokenizer output to transformer input."""
    
    config = MaskedHWMMinimalConfig()
    
    print_section("COSMOS TOKENIZER OUTPUT (Dataset Format)")
    print("\nThe Cosmos tokenizer produces factorized tokens stored in the dataset:")
    print("  - Format: [num_clips, 3, 32, 32] where:")
    print("    * num_clips = number of temporally-compressed clips (17 frames each)")
    print("    * 3 = number of factorized tokens per spatial position")
    print("    * 32×32 = spatial dimensions (compressed from 256×256)")
    print("  - Each token is an integer in range [0, 65535] (vocab_size)")
    print("  - These are FSQ quantization indices from Cosmos DV 8×8×8")
    
    # Load dataset
    print_section("STEP 1: DATASET LOADING")
    dataset = HumanoidWorldModelDataset(
        data_dir=config.train_data_dir,
        num_past_frames=config.num_past_frames,
        num_future_frames=config.num_future_frames,
        num_past_clips=config.num_past_clips,
        num_future_clips=config.num_future_clips,
        filter_interrupts=True,
        filter_overlaps=False,
    )
    
    sample = dataset[0]
    
    print("\nDataset output (single sample, no batch):")
    print_shape_info("video_past", sample['video_past'], 
                     f"({config.num_past_clips}, 3, 32, 32)")
    print_shape_info("video_future", sample['video_future'],
                     f"({config.num_future_clips}, 3, 32, 32)")
    print_shape_info("actions_past", sample['actions_past'],
                     f"({config.num_past_frames}, 25)")
    print_shape_info("actions_future", sample['actions_future'],
                     f"({config.num_future_frames}, 25)")
    
    print("\nKey points:")
    print("  ✓ Video tokens are factorized: 3 separate token maps per position")
    print("  ✓ Each token is an integer index into vocab (0-65535)")
    print("  ✓ Actions are at frame level (not clip level)")
    
    # Collator processing
    print_section("STEP 2: COLLATOR PROCESSING (Corruption + Masking)")
    collator = MaskedHWMCollator(config)
    batch = collator([sample])  # Create batch of size 1
    
    print("\nCollator output (batched):")
    print_shape_info("video_past (input)", batch['video_past'],
                     f"(B, 3, {config.num_past_clips}, 32, 32)")
    print_shape_info("video_future (input)", batch['video_future'],
                     f"(B, 3, {config.num_future_clips}, 32, 32)")
    print_shape_info("video_future_labels", batch['video_future_labels'],
                     f"(B, 3, {config.num_future_clips}, 32, 32)")
    print_shape_info("mask", batch['mask'],
                     f"(B, {config.num_future_clips}, 32, 32)")
    
    print("\nTransformations applied:")
    print("  1. Batching: Stacked samples into batch dimension")
    print("  2. Permutation: (B, T_clips, 3, H, W) → (B, 3, T_clips, H, W)")
    print("     * Factors moved to dimension 1 for easier processing")
    print("  3. Corruption: Random token replacement at rate U(0, 0.2)")
    print("     * Applied to BOTH past and future tokens")
    print("     * Past tokens: corrupted but NOT masked")
    print("     * Future tokens: corrupted AND masked")
    print("  4. Masking: Applied ONLY to future tokens using cosine schedule")
    print("     * Mask token ID = 65536 (vocab_size)")
    print("     * Same mask applied to all 3 factors at each position")
    
    # Check token ranges
    print("\nToken ranges after collator:")
    v_p = batch['video_past']
    v_f = batch['video_future']
    labels = batch['video_future_labels']
    mask = batch['mask']
    
    print(f"  video_past: [{v_p.min().item()}, {v_p.max().item()}]")
    print(f"  video_future (input): [{v_f.min().item()}, {v_f.max().item()}]")
    print(f"  video_future_labels: [{labels.min().item()}, {labels.max().item()}]")
    print(f"  Masked tokens: {(v_f == config.mask_token_id).sum().item()} / {v_f.numel()}")
    print(f"  Mask ratio: {mask.float().mean().item():.2%}")
    
    # Model embedding
    print_section("STEP 3: MODEL EMBEDDING")
    model = MaskedHWM(config)
    model.eval()
    
    with torch.no_grad():
        # Embed tokens
        v_p_emb = model.video_token_embed(batch['video_past'])
        v_f_emb = model.video_token_embed(batch['video_future'])
    
    print("\nEmbedding output:")
    print_shape_info("video_past embeddings", v_p_emb,
                     f"(B, {config.num_past_clips}, 32, 32, {config.d_model})")
    print_shape_info("video_future embeddings", v_f_emb,
                     f"(B, {config.num_future_clips}, 32, 32, {config.d_model})")
    
    print("\nEmbedding process:")
    print("  1. FactorizedEmbedding receives: (B, 3, T, H, W)")
    print("  2. Each of 3 factors is embedded separately:")
    print("     * Factor 0: embed(v_p[:, 0]) → (B, T, H, W, d_model)")
    print("     * Factor 1: embed(v_p[:, 1]) → (B, T, H, W, d_model)")
    print("     * Factor 2: embed(v_p[:, 2]) → (B, T, H, W, d_model)")
    print("  3. All 3 embeddings are SUMMED: sum(emb_0, emb_1, emb_2)")
    print("  4. Result: (B, T, H, W, d_model) - single embedding per position")
    
    # Position embeddings
    print("\nPosition embeddings added:")
    v_p_emb = v_p_emb + model.video_pos_embed[:, :config.num_past_clips]
    v_f_emb = v_f_emb + model.video_pos_embed[:, config.num_past_clips:config.num_past_clips + config.num_future_clips]
    
    print_shape_info("video_past (with pos)", v_p_emb,
                     f"(B, {config.num_past_clips}, 32, 32, {config.d_model})")
    
    # Action embedding
    print("\nAction embedding:")
    a_p_emb = model.action_embedding(batch['actions_past'])
    a_f_emb = model.action_embedding(batch['actions_future'])
    
    print_shape_info("actions_past (embedded)", a_p_emb,
                     f"(B, {config.num_past_frames}, {config.d_model})")
    print_shape_info("actions_future (embedded)", a_f_emb,
                     f"(B, {config.num_future_frames}, {config.d_model})")
    
    print("\nAction downsampling (frame → clip level):")
    a_p_clip = model._downsample_actions_to_clips(a_p_emb, config.num_past_clips)
    a_f_clip = model._downsample_actions_to_clips(a_f_emb, config.num_future_clips)
    
    print_shape_info("actions_past (clips)", a_p_clip,
                     f"(B, {config.num_past_clips}, {config.d_model})")
    print_shape_info("actions_future (clips)", a_f_clip,
                     f"(B, {config.num_future_clips}, {config.d_model})")
    
    # Transformer input
    print_section("STEP 4: TRANSFORMER INPUT")
    
    print("\nTransformer receives:")
    print(f"  v_p_emb: (B={1}, T={config.num_past_clips}, H=32, W=32, d_model={config.d_model})")
    print(f"  v_f_emb: (B={1}, T={config.num_future_clips}, H=32, W=32, d_model={config.d_model})")
    print(f"  a_p_emb: (B={1}, T={config.num_past_clips}, d_model={config.d_model})")
    print(f"  a_f_emb: (B={1}, T={config.num_future_clips}, d_model={config.d_model})")
    
    print("\nTransformer processing:")
    print("  1. Spatial mean-pooling: (B, T, H, W, d) → (B, T, d)")
    print("     * Reduces sequence length from (T × H×W) to T")
    print("     * Applied to video embeddings only")
    print("  2. Concatenation: [v_p, v_f, a_p, a_f] → (B, T_total, d)")
    print("     * T_total = T_p_clips + T_f_clips + T_p_clips + T_f_clips")
    print("  3. Temporal attention across all streams")
    print("  4. Broadcast back to spatial dimensions for video")
    
    # Full forward pass
    print_section("STEP 5: FULL FORWARD PASS")
    
    with torch.no_grad():
        logits = model(
            v_p_tokens=batch['video_past'],
            v_f_tokens=batch['video_future'],
            a_p=batch['actions_past'],
            a_f=batch['actions_future'],
        )
    
    print("\nModel output (logits):")
    print(f"  Type: List of {len(logits)} tensors (one per factor)")
    for i, logit in enumerate(logits):
        print_shape_info(f"  Factor {i} logits", logit,
                         f"(B, {config.num_future_clips}, 32, 32, {config.vocab_size})")
    
    print("\nKey points:")
    print("  ✓ Each factor gets separate logits (3 separate predictions)")
    print("  ✓ Logits predict tokens for FUTURE clips only")
    print("  ✓ Shape: (B, T_f_clips, H, W, vocab_size)")
    print("  ✓ Loss computed only on masked positions")
    
    # Summary
    print_section("DATA FLOW SUMMARY")
    
    print("\nComplete pipeline:")
    print("""
    Cosmos Tokenizer Output
    ↓ [num_clips, 3, 32, 32] - factorized tokens
    Dataset
    ↓ (num_past_clips, 3, 32, 32) + (num_future_clips, 3, 32, 32)
    Collator
    ↓ Batching + Corruption + Masking
    ↓ (B, 3, T_p, H, W) + (B, 3, T_f, H, W) - corrupted & masked
    Model Embedding
    ↓ FactorizedEmbedding: embed each factor, then SUM
    ↓ (B, T_p, H, W, d_model) + (B, T_f, H, W, d_model)
    Position Embeddings
    ↓ Add learnable position embeddings
    Transformer
    ↓ Spatial pooling + Temporal attention
    ↓ (B, T_f, H, W, vocab_size) × 3 factors
    Loss
    ↓ Cross-entropy on masked positions only
    """)
    
    print("\nCritical interface points:")
    print("  1. Tokenizer → Dataset: [num_clips, 3, 32, 32] format")
    print("  2. Dataset → Collator: Factorized tokens preserved")
    print("  3. Collator → Model: (B, 3, T, H, W) with mask tokens")
    print("  4. Model Embedding: 3 factors → summed embeddings")
    print("  5. Transformer: Processes (B, T, H, W, d) video embeddings")
    print("  6. Output: 3 separate logits (one per factor)")

if __name__ == "__main__":
    inspect_tokenizer_to_transformer()
