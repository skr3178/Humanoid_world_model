"""Example usage of Flow-HWM model programmatically."""

import torch
import sys
sys.path.insert(0, "/media/skr/storage/robot_world/humanoid_wm/build")

from flow_hwm import FlowHWMConfig, FlowHWM, create_flow_hwm
from flow_hwm.flow_matching import (
    sample_timesteps,
    sample_noise,
    construct_flow_path,
    compute_target_velocity,
    flow_matching_loss,
)
from flow_hwm.inference import generate_video_latents


def example_training_step():
    """Example of a single training step."""
    print("=" * 60)
    print("Example: Training Step")
    print("=" * 60)
    
    # Create model
    config = FlowHWMConfig()
    model = create_flow_hwm(config)
    model.train()
    
    # Create dummy batch
    B = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    v_p = torch.randn(B, config.latent_dim, config.past_video_tokens, 
                     config.latent_spatial, config.latent_spatial, device=device)
    v_f = torch.randn(B, config.latent_dim, config.future_video_tokens,
                     config.latent_spatial, config.latent_spatial, device=device)
    a_p = torch.randn(B, config.past_frames, config.action_dim, device=device)
    a_f = torch.randn(B, config.future_frames, config.action_dim, device=device)
    
    # Sample timestep and noise
    t = sample_timesteps(B, device)
    x0 = sample_noise(v_f.shape, device)
    x_t = construct_flow_path(x0, v_f, t, config.sigma_min)
    
    # Forward pass
    predicted_velocity = model(x_t, v_p, a_p, a_f, t)
    
    # Compute loss
    target_velocity = compute_target_velocity(x0, v_f, config.sigma_min)
    loss = flow_matching_loss(predicted_velocity, target_velocity)
    
    # Backward pass
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print("Training step complete!")


def example_inference():
    """Example of inference/generation."""
    print("=" * 60)
    print("Example: Inference")
    print("=" * 60)
    
    # Create model
    config = FlowHWMConfig()
    model = create_flow_hwm(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load checkpoint (if available)
    # checkpoint = torch.load("checkpoints_flow_hwm/model-10000.pt", map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create conditioning context
    B = 1
    v_p = torch.randn(B, config.latent_dim, config.past_video_tokens,
                     config.latent_spatial, config.latent_spatial, device=device)
    a_p = torch.randn(B, config.past_frames, config.action_dim, device=device)
    a_f = torch.randn(B, config.future_frames, config.action_dim, device=device)
    
    # Generate future video latents
    print("Generating future video latents...")
    with torch.no_grad():
        generated_latents = generate_video_latents(
            model,
            v_p=v_p,
            a_p=a_p,
            a_f=a_f,
            num_steps=50,
            cfg_scale=1.5,
            verbose=True,
        )
    
    print(f"Generated latents shape: {generated_latents.shape}")
    print("Inference complete!")


def example_with_dataset():
    """Example using the dataset."""
    print("=" * 60)
    print("Example: Using Dataset")
    print("=" * 60)
    
    from flow_hwm.dataset_latent import FlowHWMDataset
    
    # Create dataset
    dataset = FlowHWMDataset(
        data_dir="/media/skr/storage/robot_world/humanoid_wm/1xgpt/data/val_v2.0",
        num_past_clips=2,
        num_future_clips=1,
        latent_dim=16,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"  latent_past: {sample['latent_past'].shape}")
    print(f"  latent_future: {sample['latent_future'].shape}")
    print(f"  actions_past: {sample['actions_past'].shape}")
    print(f"  actions_future: {sample['actions_future'].shape}")
    
    # Use in model
    config = FlowHWMConfig()
    model = create_flow_hwm(config)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Prepare inputs
    v_p = sample['latent_past'].unsqueeze(0).to(device)
    v_f = sample['latent_future'].unsqueeze(0).to(device)
    a_p = sample['actions_past'].unsqueeze(0).to(device)
    a_f = sample['actions_future'].unsqueeze(0).to(device)
    
    # Sample timestep and noise
    t = sample_timesteps(1, device)
    x0 = sample_noise(v_f.shape, device)
    x_t = construct_flow_path(x0, v_f, t, config.sigma_min)
    
    # Forward pass
    with torch.no_grad():
        velocity = model(x_t, v_p, a_p, a_f, t)
    
    print(f"\nPredicted velocity shape: {velocity.shape}")
    print("Dataset example complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flow-HWM examples")
    parser.add_argument(
        "--example",
        type=str,
        choices=["training", "inference", "dataset"],
        default="training",
        help="Which example to run",
    )
    
    args = parser.parse_args()
    
    if args.example == "training":
        example_training_step()
    elif args.example == "inference":
        example_inference()
    elif args.example == "dataset":
        example_with_dataset()
