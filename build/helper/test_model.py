"""Test script to verify Masked-HWM implementation (v2.0 format)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from masked_hwm.config import MaskedHWMConfig
from masked_hwm.config_test import MaskedHWMTestConfig
from masked_hwm.model import MaskedHWM


def test_model(config, name=""):
    print(f"\n{'='*50}")
    print(f"Testing {name} config...")
    print(f"{'='*50}")
    
    model = MaskedHWM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Action dim: {config.action_dim}")
    print(f"Spatial size: {config.spatial_size}")
    print(f"Vocab size (per factor): {config.vocab_size}")
    print(f"Num factors: {config.num_factored_vocabs}")
    
    B = 2
    num_factors = config.num_factored_vocabs
    num_past_clips = model.num_past_clips
    num_future_clips = model.num_future_clips
    H = W = config.spatial_size
    
    # Video tokens: (B, num_factors, num_clips, H, W)
    v_p = torch.randint(0, config.vocab_size, (B, num_factors, num_past_clips, H, W))
    v_f = torch.randint(0, config.vocab_size, (B, num_factors, num_future_clips, H, W))
    
    # Actions: (B, num_frames, action_dim)
    a_p = torch.randn(B, config.num_past_frames, config.action_dim)
    a_f = torch.randn(B, config.num_future_frames, config.action_dim)
    
    print(f"v_p shape: {v_p.shape}")
    print(f"v_f shape: {v_f.shape}")
    print(f"a_p shape: {a_p.shape}")
    print(f"a_f shape: {a_f.shape}")
    
    logits = model(v_p, v_f, a_p, a_f)
    print(f"Output shape: {logits.shape}")
    
    # Test loss computation
    mask = torch.ones(B, num_future_clips, H, W, dtype=torch.bool)
    loss = model.compute_loss(logits, v_f, mask)
    print(f"Loss: {loss.item():.4f}")
    print("PASSED!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Test full config (requires lots of RAM)")
    args = parser.parse_args()
    
    if args.full:
        test_model(MaskedHWMConfig(), "Full")
    test_model(MaskedHWMTestConfig(), "Test (reduced)")
