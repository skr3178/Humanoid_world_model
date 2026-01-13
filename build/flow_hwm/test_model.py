"""Quick test script to verify Flow-HWM model works correctly."""

import torch
import sys
sys.path.insert(0, "/media/skr/storage/robot_world/humanoid_wm/build")

from flow_hwm import FlowHWMConfigTest, create_flow_hwm
from flow_hwm.flow_matching import sample_timesteps, sample_noise, construct_flow_path, compute_target_velocity

def test_model():
    """Test Flow-HWM model with test config."""
    print("=" * 60)
    print("Flow-HWM Model Test")
    print("=" * 60)
    
    # Create test config
    config = FlowHWMConfigTest()
    print(f"Config: {config.num_layers} layers, {config.d_model} dim, {config.num_heads} heads")
    
    # Create model
    model = create_flow_hwm(config)
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    
    # Create dummy inputs
    B = 2
    C = config.latent_dim
    T_p = config.num_past_clips
    T_f = config.num_future_clips
    H = W = config.latent_spatial
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Past video latent (conditioning)
    v_p = torch.randn(B, C, T_p, H, W, device=device)
    
    # Future video latent (target X_1)
    v_f = torch.randn(B, C, T_f, H, W, device=device)
    
    # Actions
    a_p = torch.randn(B, T_p * config.frames_per_clip, config.action_dim, device=device)
    a_f = torch.randn(B, T_f * config.frames_per_clip, config.action_dim, device=device)
    
    # Timestep
    t = sample_timesteps(B, device)
    
    # Sample noise and construct X_t
    x0 = sample_noise(v_f.shape, device)
    x_t = construct_flow_path(x0, v_f, t, config.sigma_min)
    
    print(f"\nInput shapes:")
    print(f"  v_p (past video): {v_p.shape}")
    print(f"  v_f_noisy (X_t): {x_t.shape}")
    print(f"  a_p (past actions): {a_p.shape}")
    print(f"  a_f (future actions): {a_f.shape}")
    print(f"  t (timesteps): {t.shape}")
    
    # Forward pass
    print(f"\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        velocity = model(x_t, v_p, a_p, a_f, t)
    
    print(f"Output shape: {velocity.shape}")
    print(f"Expected: (B={B}, C={C}, T_f={T_f}, H={H}, W={W})")
    
    # Check output shape
    assert velocity.shape == (B, C, T_f, H, W), f"Shape mismatch: {velocity.shape} != {(B, C, T_f, H, W)}"
    
    # Test loss computation
    target_velocity = compute_target_velocity(x0, v_f, config.sigma_min)
    from flow_hwm.flow_matching import flow_matching_loss
    loss = flow_matching_loss(velocity, target_velocity)
    print(f"\nLoss: {loss.item():.4f}")
    
    # Test with CFG
    print(f"\nTesting classifier-free guidance...")
    with torch.no_grad():
        velocity_cfg = model.forward_with_cfg(x_t, v_p, a_p, a_f, t, cfg_scale=1.5)
    print(f"CFG output shape: {velocity_cfg.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_model()
