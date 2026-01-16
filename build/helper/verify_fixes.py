#!/usr/bin/env python3
"""Verify that all training fixes have been applied correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from masked_hwm.config_12gb import MaskedHWM12GBConfig
from masked_hwm.model import MaskedHWM, FactorizedEmbedding
import torch
import math

def verify_config():
    """Verify config changes."""
    config = MaskedHWM12GBConfig()
    
    print("✓ Verifying Config Changes:")
    print(f"  - Batch size: {config.batch_size} (expected: 4)")
    assert config.batch_size == 4, "Batch size should be 4"
    
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps} (expected: 4)")
    assert config.gradient_accumulation_steps == 4, "Grad accum should be 4"
    
    print(f"  - Effective batch size: {config.batch_size * config.gradient_accumulation_steps} (expected: 16)")
    assert config.batch_size * config.gradient_accumulation_steps == 16, "Effective batch should be 16"
    
    print(f"  - Learning rate: {config.learning_rate} (expected: 5e-5)")
    assert config.learning_rate == 5e-5, "LR should be 5e-5"
    
    print(f"  - Warmup steps: {config.warmup_steps} (expected: 500)")
    assert config.warmup_steps == 500, "Warmup should be 500"
    
    print("✓ Config verification passed!\n")
    return config

def verify_scheduler():
    """Verify scheduler change."""
    print("✓ Verifying Scheduler:")
    try:
        from training.train import get_cosine_schedule_with_warmup
        print("  - Cosine schedule imported successfully")
        print("✓ Scheduler verification passed!\n")
    except ImportError as e:
        print(f"✗ Failed to import cosine scheduler: {e}")
        sys.exit(1)

def verify_factorized_embedding():
    """Verify factorized embedding initialization scaling."""
    print("✓ Verifying Factorized Embedding:")
    
    config = MaskedHWM12GBConfig()
    num_factors = config.num_factored_vocabs
    init_std = config.init_std
    
    # Create embedding
    embed = FactorizedEmbedding(
        num_factored_vocabs=num_factors,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        init_std=init_std,
    )
    
    expected_scaled_std = init_std / math.sqrt(num_factors)
    print(f"  - Number of factors: {num_factors}")
    print(f"  - Base init_std: {init_std}")
    print(f"  - Scaled init_std: {embed.scaled_init_std:.6f}")
    print(f"  - Expected: {expected_scaled_std:.6f}")
    
    assert abs(embed.scaled_init_std - expected_scaled_std) < 1e-6, "Scaled std mismatch"
    print("✓ Factorized embedding scaling verified!\n")

def verify_model_initialization():
    """Verify model can be initialized with fixes."""
    print("✓ Verifying Model Initialization:")
    
    config = MaskedHWM12GBConfig()
    model = MaskedHWM(config)
    
    print(f"  - Model created successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check factorized embedding has scaled initialization
    assert hasattr(model.video_token_embed, 'scaled_init_std'), "Missing scaled_init_std"
    print(f"  - Factorized embedding scaled_init_std: {model.video_token_embed.scaled_init_std:.6f}")
    
    # Quick forward pass test
    batch_size = 2
    T_p = 1
    T_f = 1
    
    v_p = torch.randint(0, config.vocab_size, (batch_size, config.num_factored_vocabs, T_p, 32, 32))
    v_f = torch.randint(0, config.vocab_size, (batch_size, config.num_factored_vocabs, T_f, 32, 32))
    a_p = torch.randn(batch_size, T_p * 17, config.action_dim)
    a_f = torch.randn(batch_size, T_f * 17, config.action_dim)
    
    with torch.no_grad():
        logits = model(v_p, v_f, a_p, a_f)
    
    print(f"  - Forward pass successful: {logits.shape}")
    print("✓ Model initialization verified!\n")

def main():
    print("=" * 60)
    print("Training Fixes Verification")
    print("=" * 60 + "\n")
    
    try:
        verify_config()
        verify_scheduler()
        verify_factorized_embedding()
        verify_model_initialization()
        
        print("=" * 60)
        print("✓ ALL VERIFICATIONS PASSED!")
        print("=" * 60)
        print("\nYou can now start training with:")
        print("  cd build && ./train_10pct_v2_12gb.sh")
        print("\nExpected results:")
        print("  - Initial loss: ~11.1 (log(65536))")
        print("  - After 100 steps: ~9-10 range")
        print("  - After 1000 steps: ~7-8 range")
        print("  - Gradient norms: 0.1-10 range (healthy)")
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
